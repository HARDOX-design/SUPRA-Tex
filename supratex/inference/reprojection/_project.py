"""Project observed albedo at UV texel centres with measured 19-phase gates.
"""
from __future__ import annotations
import argparse, copy, io, json, os, sys, time
from pathlib import Path
from ._common import OUTSIDE_COLOR_MAX_ABS, outside_color_passes, load_renderer, bind_file
from . import _glb as fixed
SIZE = 4096
RES = 1024
TRAIN = tuple(((x / 4, y / 4) for x in range(4) for y in range(4)))
HELD = ((0.125, 0.375), (0.375, 0.625), (0.875, 0.125))

def put(p, x):
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False) + '\n')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=('B',), required=True)
    ap.add_argument('--baseline-glb', type=Path, required=True)
    ap.add_argument('--source-albedo', type=Path, required=True)
    ap.add_argument('--source-rgba', type=Path, required=True)
    ap.add_argument('--camera', type=Path, required=True)
    ap.add_argument('--evaluator', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    for p in (a.baseline_glb, a.source_albedo, a.source_rgba, a.camera, a.evaluator):
        if not p.is_file():
            raise FileNotFoundError(p)
    a.output.mkdir(parents=True)
    put(a.output / 'PLAN.json', {'arm': a.arm, 'baseline_glb': str(a.baseline_glb.absolute()), 'source_albedo': str(a.source_albedo.absolute()), 'source_rgba': str(a.source_rgba.absolute()), 'camera': str(a.camera.absolute()), 'evaluator': str(a.evaluator.absolute()), 'size': SIZE, 'resolution': RES, 'train_phases': TRAIN, 'heldout_phases': HELD, 'source': 'predicted albedo only; no GT read', 'support': 'common sixteen-phase union, intersected with covered source-hard-trusted UV-triangle centres; alpha>=254 source support eroded 2 pixels; two-pixel unsafe-surface footprint blacklist', 'B': 'atlas-centre UV-triangle final-surface projection then bilinear source albedo sample mixed as T0+s*(source-T0)', 'validation': 'actual GLB rerender at 16 train + 3 heldout phases'})
    os.environ['FLEX_GEMM_USE_AUTOTUNE_CACHE'] = '0'
    os.environ['FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE'] = '0'
    import cv2, numpy as np, torch
    from PIL import Image
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    began = time.monotonic()
    rm = load_renderer(a.evaluator)
    rt = rm._runtime_imports()
    dr = rt['dr']
    camera = json.loads(a.camera.read_text())
    (doc, bin0) = fixed._read_glb(a.baseline_glb)
    fixed._assert_fixed_scene(doc)
    (ii, vi) = fixed._base_color_image_ref(doc)
    (im, _, off, n) = fixed._decode_base_image(doc, bin0, ii, vi)
    pix = np.asarray(im).copy()
    if pix.shape[:2] != (SIZE, SIZE):
        raise ValueError('requires 4096 atlas')
    vals = rt['_load_textured_mesh'](a.baseline_glb, rt)
    (vertices, faces, _, uv, _, _, factor, _, _) = vals
    if not np.allclose(factor, 1):
        raise RuntimeError('baseColorFactor')
    exported = torch.as_tensor(vertices, dtype=torch.float32, device='cuda')
    native = exported * torch.tensor((-1.0, 1.0, -1.0), device='cuda')
    faces = torch.as_tensor(faces, dtype=torch.int32, device='cuda').contiguous()
    uv = torch.as_tensor(uv, dtype=torch.float32, device='cuda')
    uv = torch.stack((uv[:, 0], 1 - uv[:, 1]), -1).contiguous()
    base = torch.as_tensor(pix[..., :3].copy(), dtype=torch.float32, device='cuda').reshape(-1, 3) / 255
    rgba = np.asarray(Image.open(a.source_rgba).convert('RGBA'))
    hard = cv2.erode((rgba[..., 3] >= 254).astype('uint8'), np.ones((5, 5), np.uint8)).astype(bool)
    albedo = np.asarray(Image.open(a.source_albedo).convert('RGB'), np.float32) / 255
    if albedo.shape[:2] != (RES, RES) or hard.shape != (RES, RES):
        raise ValueError('aligned source must be 1024')
    trust = np.clip(cv2.distanceTransform(hard.astype('uint8'), cv2.DIST_L2, 3) / 4.0, 0, 1)
    src = torch.as_tensor(albedo, device='cuda').permute(2, 0, 1)[None]
    trust_t = torch.as_tensor(trust, dtype=torch.float32, device='cuda')[None, None]
    hard_t = torch.as_tensor(hard, device='cuda')
    (E, K) = rm._native_camera(camera, 'cuda', rt)
    original = rm._exported_glb_camera

    def pc(cam, device, runtime):
        (ex, inn) = original(cam, device, runtime)
        (dx, dy) = cam.get('front_atlas_phase', (0.0, 0.0))
        inn = inn.clone()
        inn[0, 2] += dx / RES
        inn[1, 2] += dy / RES
        return (ex, inn)
    rm._exported_glb_camera = pc

    def op(phase, need_all=False):
        cam = copy.deepcopy(camera)
        cam['front_atlas_phase'] = phase
        (ee, kk) = pc(cam, 'cuda', rt)
        dist = rm._camera_distance(cam)
        proj = rt['intrinsics_to_projection'](kk, max(0.2, dist - 1), dist + 1)
        h = torch.cat((exported, torch.ones_like(exported[:, :1])), 1)
        (rast, _) = dr.rasterize(dr.RasterizeCudaContext(device='cuda'), (h @ (proj @ ee).T)[None].contiguous(), faces, (RES, RES))
        (uvs, _) = dr.interpolate(uv[None], rast, faces)
        (points, _) = dr.interpolate(native[None], rast, faces)
        mask = (rast[0, ..., -1] > 0).cpu().numpy()
        rowsall = torch.as_tensor(np.flatnonzero(mask.reshape(-1)), device='cuda')
        xyz = points[0].reshape(-1, 3)[rowsall]
        q = torch.cat((xyz, torch.ones_like(xyz[:, :1])), 1) @ E.T
        q = q[:, :3] @ K.T
        suv = q[:, :2] / q[:, 2:3].clamp_min(1e-08)
        near = torch.floor(suv * RES).long()
        inside = (near[:, 0] >= 0) & (near[:, 0] < RES) & (near[:, 1] >= 0) & (near[:, 1] < RES) & (q[:, 2] > 0)
        trusted = torch.zeros(len(rowsall), dtype=torch.bool, device='cuda')
        trusted[inside] = hard_t[near[inside, 1], near[inside, 0]]
        rows = rowsall[trusted]
        tsuv = suv[trusted]

        def fp(rr):
            t = uvs[0].reshape(-1, 2)[rr] * SIZE - 0.5
            xy = t.floor().long()
            f = t - xy
            ids = []
            ws = []
            for (dx, dy) in ((0, 0), (1, 0), (0, 1), (1, 1)):
                ids.append((xy[:, 1] + dy) % SIZE * SIZE + (xy[:, 0] + dx) % SIZE)
                ws.append((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]))
            return (torch.stack(ids, 1), torch.stack(ws, 1))
        (idx, w) = fp(rows)
        unsafe = np.zeros(RES * RES, np.uint8)
        unsafe[rowsall[~trusted].cpu().numpy()] = 1
        unsafe = cv2.dilate(unsafe.reshape(RES, RES), np.ones((5, 5), np.uint8)) & mask
        ur = torch.as_tensor(np.flatnonzero(unsafe.reshape(-1)), device='cuda')
        outside = fp(ur)[0] if len(ur) else torch.empty((0, 4), device='cuda', dtype=torch.long)
        target = rt['F'].grid_sample(src, (tsuv * 2 - 1).reshape(1, 1, -1, 2), align_corners=False, padding_mode='zeros')[0, :, 0].T
        strength = rt['F'].grid_sample(trust_t, (tsuv * 2 - 1).reshape(1, 1, -1, 2), align_corners=False, padding_mode='zeros')[0, 0, 0]
        screen = np.zeros(RES * RES, bool)
        screen[rows.cpu().numpy()] = True
        return {'phase': phase, 'idx': idx, 'w': w, 'outside': outside, 'target': target, 'strength': strength, 'rows': rows, 'screen': screen.reshape(RES, RES), 'uvs': uvs, 'points': points, 'rast': rast, 'trusted': trusted, 'rowsall': rowsall, 'suv': suv}
    ops = [op(p) for p in TRAIN]
    active = torch.unique(torch.cat([o['idx'].reshape(-1) for o in ops]))
    (rastuv, _) = dr.rasterize(dr.RasterizeCudaContext(device='cuda'), torch.cat((uv * 2 - 1, torch.zeros_like(uv[:, :1]), torch.ones_like(uv[:, :1])), 1)[None], faces, (SIZE, SIZE))
    (pxyz, _) = dr.interpolate(native[None], rastuv, faces)
    xy = torch.stack((active // SIZE, active % SIZE), 1)
    covered = rastuv[0, xy[:, 0], xy[:, 1], -1] > 0
    xyz = pxyz[0, xy[:, 0], xy[:, 1]]
    q = torch.cat((xyz, torch.ones_like(xyz[:, :1])), 1) @ E.T
    q = q[:, :3] @ K.T
    centre_suv = q[:, :2] / q[:, 2:3].clamp_min(1e-08)
    ne = torch.floor(centre_suv * RES).long()
    inside = (ne[:, 0] >= 0) & (ne[:, 0] < RES) & (ne[:, 1] >= 0) & (ne[:, 1] < RES) & (q[:, 2] > 0)
    centre_trust = torch.zeros(len(active), dtype=torch.bool, device='cuda')
    centre_trust[inside] = hard_t[ne[inside, 1], ne[inside, 0]]
    centre_keep = covered & centre_trust
    active = active[centre_keep]
    centre_suv = centre_suv[centre_keep]
    look = torch.full((SIZE * SIZE,), -1, dtype=torch.long, device='cuda')
    look[active] = torch.arange(len(active), device='cuda')
    outids = torch.unique(torch.cat([o['outside'].reshape(-1) for o in ops if o['outside'].numel()])) if any((o['outside'].numel() for o in ops)) else torch.empty(0, dtype=torch.long, device='cuda')
    blocked = torch.zeros(len(active), dtype=torch.bool, device='cuda')
    if len(outids):
        where = torch.searchsorted(active, outids)
        hit = (where < len(active)) & (active[where.clamp_max(len(active) - 1)] == outids)
        blocked[where[hit]] = True
    if active.numel() == 0 or not bool((~blocked).any()):
        raise RuntimeError('no unblocked trusted atlas texels remain')
    np.savez_compressed(a.output / 'COMMON_DOMAIN.npz', active_texel_ids=active.cpu().numpy(), blocked_texel_ids=active[blocked].cpu().numpy())
    for o in ops:
        compact = look[o['idx']]
        keep = (compact >= 0).all(1)
        for key in ('idx', 'w'):
            o[key] = o[key][keep]
        for key in ('target', 'strength', 'rows'):
            o[key] = o[key][keep]
        o['compact'] = compact[keep]
        o['allowed'] = (~blocked[o['compact']]).float()
        o['base'] = (base[o['idx']] * o['w'][..., None]).sum(1)
        o['screen'] = np.zeros((RES, RES), bool)
        o['screen'].reshape(-1)[o['rows'].cpu().numpy()] = True
    mode = 'uv_texel_centre_projection'
    corrected = base[active].clone()
    s = centre_suv
    sample = rt['F'].grid_sample(src, (s * 2 - 1).reshape(1, 1, -1, 2), align_corners=False, padding_mode='zeros')[0, :, 0].T
    strength = rt['F'].grid_sample(trust_t, (s * 2 - 1).reshape(1, 1, -1, 2), align_corners=False, padding_mode='zeros')[0, 0, 0]
    take = ~blocked
    corrected[take] = base[active][take] + strength[take, None] * (sample[take] - base[active][take])
    out = pix.copy()
    out.reshape(-1, out.shape[-1])[active.cpu().numpy(), :3] = (corrected.cpu().numpy() * 255 + 0.5).astype(np.uint8)
    blob = io.BytesIO()
    Image.fromarray(out).save(blob, format='PNG')
    od = copy.deepcopy(doc)
    ob = fixed._replace_image_payload(od, bin0, vi, off, n, blob.getvalue())
    candidate = a.output / 'candidate'
    candidate.mkdir()
    glb = candidate / 'mesh.glb'
    fixed._write_glb(glb, od, ob)
    pres = fixed._nonbase_identity_check(doc, bin0, glb, vi)

    def render(p, split):
        folder = a.output / 'renders' / split / f'x{p[0]:.3f}_y{p[1]:.3f}'
        folder.mkdir(parents=True)
        cam = copy.deepcopy(camera)
        cam['front_atlas_phase'] = p
        o = op(p)
        ca = rm.render_glb(glb, cam, folder / 'candidate.png', resolution=RES, rt=rt)
        ba = rm.render_glb(a.baseline_glb, cam, folder / 'baseline.png', resolution=RES, rt=rt)
        with np.load(ca['attrs_npz']) as z:
            (ci, cm, cmra) = (z['albedo'], z['mask'].astype(bool), z['mra'])
        with np.load(ba['attrs_npz']) as z:
            (bi, bm, bmra) = (z['albedo'], z['mask'].astype(bool), z['mra'])
        if not np.array_equal(cm, bm) or not np.array_equal(cmra, bmra):
            raise RuntimeError('mask/MRA changed')
        target = np.zeros_like(ci)
        target.reshape(-1, 3)[o['rows'].cpu().numpy()] = o['target'].cpu().numpy()
        sel = o['screen'] & cm
        Image.fromarray((target * 255 + 0.5).astype(np.uint8)).save(folder / 'target_projected.png')
        quant = torch.as_tensor(out[..., :3], dtype=torch.float32, device='cuda').reshape(-1, 3) / 255
        replay = (quant[o['idx']] * o['w'][..., None]).sum(1).cpu().numpy()
        actual = ci.reshape(-1, 3)[o['rows'].cpu().numpy()]
        mx = float(np.abs(replay - actual).max())
        outside = cm & ~o['screen']
        oo = float(np.abs(ci - bi)[outside].max()) if outside.any() else 0.0
        r = {'phase': p, 'pixels': int(sel.sum()), 'candidate_vs_source': rm._masked_metrics(ci, target, sel), 'baseline_vs_source': rm._masked_metrics(bi, target, sel), 'quantized_atlas_replay_max': mx, 'outside_source_hardtrust_max_abs_threshold': OUTSIDE_COLOR_MAX_ABS, 'outside_source_hardtrust_delta_max_abs': oo, 'passes': bool(mx <= 1e-05 and outside_color_passes(oo))}
        r.update(mask_exact=True, mra_exact=True, evidence={'candidate': bind_file(Path(ca['attrs_npz'])), 'baseline': bind_file(Path(ba['attrs_npz']))})
        put(folder / 'PHASE_RECEIPT.json', r)
        return r
    train = [render(x, 'train') for x in TRAIN]
    held = [render(x, 'heldout') for x in HELD]
    changed = int(np.any(out[..., :3] != pix[..., :3], axis=-1).sum())
    conv = True
    ok = conv and all((x['passes'] for x in train + held))
    result = {'status': 'valid' if ok else 'invalid', 'arm': a.arm, 'mode': mode, 'seconds': time.monotonic() - began, 'candidate_glb': str(glb), 'gates': {'accepted': ok, 'common_support_and_protection': True, 'nonbase_identity': pres, 'all_19_phase_renders': len(train) + len(held) == 19}, 'atlas': {'active_texels': int(len(active)), 'blocked_texels': int(blocked.sum()), 'changed_texels': changed, 'alpha_exact': bool(np.array_equal(out[..., 3], pix[..., 3])) if out.shape[-1] == 4 else True}, 'phase_results': {'train': train, 'heldout': held}, 'limits': 'Fixed-shape, observed-source front control. It does not establish novel-view quality or a model improvement.'}
    put(a.output / 'RECEIPT.json', result)
    put(a.output / 'RESULT.json', result)
    if not ok:
        raise RuntimeError('acceptance failed')
    print(json.dumps({'status': 'valid', 'candidate': str(glb)}))
if __name__ == '__main__':
    main()
