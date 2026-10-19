"""Repair one-texel UV gutters with observed continuous albedo gradients.

Each uncovered gutter texel next to an unblocked projected texel is anchored
to that nearest covered donor's projected RGB, then shifted by the predicted
albedo difference between the donor centre and the gutter centre extrapolated
on the donor's UV face. A gutter is written only if both samples have positive
camera depth, a finite non-degenerate UV face, bounded barycentrics, and fully
hard-trusted bilinear source footprints, and if its bilinear atlas taps avoid
untrusted pixels in all nineteen phases. Every UV-covered texel, blocked texel,
alpha, and all non-base GLB content stay exact.
"""
from __future__ import annotations
import argparse, copy, io, json, os, shutil, sys, time
from pathlib import Path
from ._common import GRID_PHASES, OFFGRID_PHASES, OUTSIDE_COLOR_MAX_ABS, IDENTITY_GATES, outside_color_passes, load_renderer, bind_file
from . import _glb as fixed
from ._project import SCHEMA as PROJECT_SCHEMA
SCHEMA = 'supratex.reprojection.continuous'
SIZE = 4096
RES = 1024
REJECT_UNSAFE = 1
REJECT_CHART = 2
REJECT_DEGENERATE_UV = 4
REJECT_SOURCE = 8

def write_once(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def require(path, label):
    path = Path(path).absolute()
    if not path.is_file():
        raise FileNotFoundError(f'{label}: {path}')
    return path

def selected_case(manifest, label):
    found = [x for x in json.loads(require(manifest, 'manifest').read_text()) if x.get('label') == label]
    if len(found) != 1:
        raise ValueError(f'expected one manifest case for {label!r}')
    return found[0]

def assert_skew_barycentric_oracle(np):
    """Catch row/column and signed-area errors on a non-axis-aligned triangle."""
    tri = np.asarray(((0.11, 0.17), (0.83, 0.29), (0.26, 0.91)), np.float64)
    expected = np.asarray((0.19, 0.34, 0.47), np.float64)
    point = (tri * expected[:, None]).sum(0)
    basis = tri[:2] - tri[2]
    solved = np.linalg.solve(basis.T, point - tri[2])
    recovered = np.asarray((solved[0], solved[1], 1 - solved.sum()))
    if not np.allclose(recovered, expected, rtol=0, atol=1e-12):
        raise RuntimeError('skew barycentric oracle failed')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--renderer', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gutter-radius', type=int, choices=(1,), default=1, help='Only one-texel gutter repair is supported.')
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    case = selected_case(a.manifest, a.case)
    renderer = require(a.renderer, 'renderer')
    (work, pdir) = (Path(case['work']).absolute(), Path(case['project_output']).absolute())
    baseline = require(case['baseline_glb'], 'baseline GLB')
    pglb = require(pdir / 'candidate/mesh.glb', 'accepted projection GLB')
    camera_path = require(work / 'camera.json', 'camera')
    rgba_path = require(work / 'condition_rgba.png', 'condition RGBA')
    albedo_path = require(work / 'albedo.png', 'predicted albedo')
    domain_path = require(pdir / 'COMMON_DOMAIN.npz', 'projection common domain')
    pplan = json.loads(require(pdir / 'PLAN.json', 'projection PLAN').read_text())
    presult = json.loads(require(pdir / 'RESULT.json', 'projection RESULT').read_text())
    expected = {
        'schema': PROJECT_SCHEMA,
        'baseline_glb': str(baseline),
        'camera': str(camera_path),
        'source_albedo': str(albedo_path),
        'source_rgba': str(rgba_path),
        'renderer': str(renderer),
    }
    if [k for (k, v) in expected.items() if pplan.get(k) != v]:
        raise RuntimeError('projection PLAN binding failed')
    if presult.get('status') != 'valid' or not presult.get('gates', {}).get('accepted'):
        raise RuntimeError('projection source is not an accepted projection result')
    if Path(presult.get('candidate_glb', '')).absolute() != pglb:
        raise RuntimeError('projection result candidate binding failed')
    identity = presult.get('gates', {}).get('nonbase_identity', {})
    if not all((identity.get(k) for k in IDENTITY_GATES)):
        raise RuntimeError('projection non-base identity gate failed')
    phase_results = presult.get('phase_results', {})
    phases = phase_results.get('grid', []) + phase_results.get('offgrid', [])
    if (not presult.get('gates', {}).get('all_19_phase_renders')
            or not presult.get('atlas', {}).get('alpha_exact')
            or len(phases) != 19
            or not all((x.get('passes') for x in phases))):
        raise RuntimeError('projection alpha or 19-phase receipt binding failed')
    a.output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), a.output / 'CODE_SNAPSHOT.py')
    write_once(a.output / 'PLAN.json', {
        'schema': SCHEMA,
        'case': a.case,
        'manifest': str(a.manifest.absolute()),
        'project_glb': str(pglb),
        'baseline_glb': str(baseline),
        'camera': str(camera_path),
        'renderer': str(renderer),
        'source_albedo': str(albedo_path),
        'source_rgba': str(rgba_path),
        'common_domain': str(domain_path),
        'grid_phases': GRID_PHASES,
        'offgrid_phases': OFFGRID_PHASES,
        'gutter_radius': a.gutter_radius,
    })
    os.environ['FLEX_GEMM_USE_AUTOTUNE_CACHE'] = '0'
    os.environ['FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE'] = '0'
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    assert_skew_barycentric_oracle(np)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    started = time.monotonic()
    rm = load_renderer(renderer)
    rt = rm._runtime_imports()
    dr = rt['dr']
    camera = json.loads(camera_path.read_text())
    (pdoc, pbin) = fixed._read_glb(pglb)
    fixed._assert_fixed_scene(pdoc)
    (ii, vi) = fixed._base_color_image_ref(pdoc)
    (pimg, _, off, n) = fixed._decode_base_image(pdoc, pbin, ii, vi)
    before = np.asarray(pimg).copy()
    (basedoc, basebin) = fixed._read_glb(baseline)
    fixed._assert_fixed_scene(basedoc)
    (baseii, basevi) = fixed._base_color_image_ref(basedoc)
    (baseimg, _, _, _) = fixed._decode_base_image(basedoc, basebin, baseii, basevi)
    base_before = np.asarray(baseimg).copy()
    if before.shape[:2] != (SIZE, SIZE):
        raise ValueError('requires 4096 atlas')
    if base_before.shape != before.shape:
        raise RuntimeError('baseline/projection baseColor image shape mismatch')
    with np.load(domain_path, allow_pickle=False) as z:
        (active_ids, blocked_ids) = (z['active_texel_ids'].astype(np.int64), z['blocked_texel_ids'].astype(np.int64))
    project_atlas = presult.get('atlas', {})
    if (len(active_ids) != int(project_atlas.get('active_texels', -1))
            or len(blocked_ids) != int(project_atlas.get('blocked_texels', -1))
            or len(np.unique(active_ids)) != len(active_ids)
            or len(np.unique(blocked_ids)) != len(blocked_ids)):
        raise RuntimeError('projection common-domain count or uniqueness binding failed')
    (active, blocked) = (np.zeros(SIZE * SIZE, bool), np.zeros(SIZE * SIZE, bool))
    (active[active_ids], blocked[blocked_ids]) = (True, True)
    if np.any(blocked & ~active):
        raise RuntimeError('blocked IDs must be in the projection active domain')
    (vertices, faces_np, _, uv_np, _, _, factor, _, _) = rt['_load_textured_mesh'](pglb, rt)
    if not np.allclose(factor, 1):
        raise RuntimeError('unexpected baseColorFactor')
    exported = torch.as_tensor(vertices, dtype=torch.float32, device='cuda')
    native = exported * torch.tensor((-1.0, 1.0, -1.0), device='cuda')
    faces = torch.as_tensor(faces_np, dtype=torch.int32, device='cuda').contiguous()
    uv = torch.as_tensor(uv_np, dtype=torch.float32, device='cuda')
    uv = torch.stack((uv[:, 0], 1 - uv[:, 1]), -1).contiguous()
    rgba = np.asarray(Image.open(rgba_path).convert('RGBA'))
    hard = cv2.erode((rgba[..., 3] >= 254).astype('uint8'), np.ones((5, 5), np.uint8)).astype(bool)
    source = np.asarray(Image.open(albedo_path).convert('RGB'), np.float32) / 255
    if hard.shape != (RES, RES) or source.shape[:2] != (RES, RES):
        raise ValueError('source inputs must be aligned 1024')
    hard_t = torch.as_tensor(hard, device='cuda')
    src = torch.as_tensor(source, device='cuda').permute(2, 0, 1)[None]
    trust = np.clip(cv2.distanceTransform(hard.astype('uint8'), cv2.DIST_L2, 3) / 4.0, 0, 1)
    trust_t = torch.as_tensor(trust, dtype=torch.float32, device='cuda')[None, None]
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

    def footprint(rows, uvs):
        tex = uvs[0].reshape(-1, 2)[rows] * SIZE - 0.5
        (xy, frac) = (tex.floor().long(), tex - tex.floor())
        (ids, weights) = ([], [])
        for (dx, dy) in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ids.append((xy[:, 1] + dy) % SIZE * SIZE + (xy[:, 0] + dx) % SIZE)
            weights.append((frac[:, 0] if dx else 1 - frac[:, 0]) * (frac[:, 1] if dy else 1 - frac[:, 1]))
        return (torch.stack(ids, 1), torch.stack(weights, 1))

    def operation(phase):
        cam = copy.deepcopy(camera)
        cam['front_atlas_phase'] = phase
        (ee, kk) = pc(cam, 'cuda', rt)
        dist = rm._camera_distance(cam)
        proj = rt['intrinsics_to_projection'](kk, max(0.2, dist - 1), dist + 1)
        h = torch.cat((exported, torch.ones_like(exported[:, :1])), 1)
        (rast, _) = dr.rasterize(dr.RasterizeCudaContext(device='cuda'), (h @ (proj @ ee).T)[None].contiguous(), faces, (RES, RES))
        (uvs, _) = dr.interpolate(uv[None], rast, faces)
        (pts, _) = dr.interpolate(native[None], rast, faces)
        mask = (rast[0, ..., -1] > 0).cpu().numpy()
        rowsall = torch.as_tensor(np.flatnonzero(mask.reshape(-1)), device='cuda')
        xyz = pts[0].reshape(-1, 3)[rowsall]
        q = torch.cat((xyz, torch.ones_like(xyz[:, :1])), 1) @ E.T
        q = q[:, :3] @ K.T
        suv = q[:, :2] / q[:, 2:3].clamp_min(1e-08)
        near = torch.floor(suv * RES).long()
        inside = (near[:, 0] >= 0) & (near[:, 0] < RES) & (near[:, 1] >= 0) & (near[:, 1] < RES) & (q[:, 2] > 0)
        trusted = torch.zeros(len(rowsall), dtype=torch.bool, device='cuda')
        trusted[inside] = hard_t[near[inside, 1], near[inside, 0]]
        rows = rowsall[trusted]
        (idx, weights) = footprint(rows, uvs)
        unsafe_screen = np.zeros(RES * RES, np.uint8)
        unsafe_screen[rowsall[~trusted].cpu().numpy()] = 1
        unsafe_screen = cv2.dilate(unsafe_screen.reshape(RES, RES), np.ones((5, 5), np.uint8)) & mask
        unsafe_rows = torch.as_tensor(np.flatnonzero(unsafe_screen.reshape(-1)), device='cuda')
        unsafe = footprint(unsafe_rows, uvs)[0] if len(unsafe_rows) else torch.empty((0, 4), dtype=torch.long, device='cuda')
        target = rt['F'].grid_sample(src, (suv[trusted] * 2 - 1).reshape(1, 1, -1, 2), align_corners=False, padding_mode='zeros')[0, :, 0].T
        screen = np.zeros(RES * RES, bool)
        screen[rows.cpu().numpy()] = True
        return {
            'phase': phase,
            'idx': idx,
            'w': weights,
            'unsafe': unsafe,
            'rows': rows,
            'screen': screen.reshape(RES, RES),
            'target': target,
        }
    operations = [operation(p) for p in GRID_PHASES + OFFGRID_PHASES]
    if any(x['unsafe'].numel() for x in operations):
        unsafe_ids = torch.unique(torch.cat([x['unsafe'].reshape(-1) for x in operations if x['unsafe'].numel()]))
    else:
        unsafe_ids = torch.empty(0, dtype=torch.long, device='cuda')
    unsafe = np.zeros(SIZE * SIZE, bool)
    unsafe[unsafe_ids.cpu().numpy()] = True
    uvclip = torch.cat((uv * 2 - 1, torch.zeros_like(uv[:, :1]), torch.ones_like(uv[:, :1])), 1)[None]
    (urast, _) = dr.rasterize(dr.RasterizeCudaContext(device='cuda'), uvclip, faces, (SIZE, SIZE))
    covered = (urast[0, ..., -1] > 0).cpu().numpy()
    (fcpu, uvcpu) = (faces.cpu().numpy(), uv.cpu().numpy())
    parent = np.arange(len(fcpu), dtype=np.int32)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        (i, j) = (root(i), root(j))
        if i != j:
            parent[j] = i
    edges = {}
    for (fi, tri) in enumerate(fcpu):
        for (u, v) in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(min(u, v)), int(max(u, v)))
            if key in edges:
                join(fi, edges[key])
            else:
                edges[key] = fi
    face_chart = np.asarray([root(i) for i in range(len(fcpu))], np.int32)
    (_, face_chart) = np.unique(face_chart, return_inverse=True)
    primitive = urast[0, ..., -1].long().cpu().numpy() - 1
    chart = np.full((SIZE, SIZE), -1, np.int32)
    chart[covered] = face_chart[primitive[covered]]
    core = (active & ~blocked).reshape(SIZE, SIZE) & covered
    if np.any(active.reshape(SIZE, SIZE) & ~covered):
        raise RuntimeError('projection active domain includes uncovered texels')
    base = ~covered & ~blocked.reshape(SIZE, SIZE)
    rejection = np.zeros((SIZE, SIZE), np.uint8)
    rejection[base & unsafe.reshape(SIZE, SIZE)] |= REJECT_UNSAFE
    candidate = np.zeros_like(core)
    (donor_y, donor_x) = (np.full(core.shape, -1, np.int32), np.full(core.shape, -1, np.int32))
    donor_chart = np.full_like(chart, -1)
    for (dy, dx) in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        shifted = np.zeros_like(core)
        (sy, sx) = (slice(max(0, -dy), SIZE - max(0, dy)), slice(max(0, -dx), SIZE - max(0, dx)))
        (ty, tx) = (slice(max(0, dy), SIZE - max(0, -dy)), slice(max(0, dx), SIZE - max(0, -dx)))
        shifted[ty, tx] = core[sy, sx]
        add = shifted & base & ~unsafe.reshape(SIZE, SIZE)
        (ay, ax) = np.nonzero(add)
        labels = chart[ay - dy, ax - dx]
        for (y, x, label) in zip(ay.tolist(), ax.tolist(), labels.tolist()):
            if not candidate[y, x]:
                candidate[y, x] = True
                donor_y[y, x] = y - dy
                donor_x[y, x] = x - dx
                donor_chart[y, x] = label
            elif donor_chart[y, x] != label:
                donor_chart[y, x] = -1
    prechart = candidate.copy()
    for (dy, dx) in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        neigh = np.full_like(chart, -1)
        neigh[max(0, dy):SIZE - max(0, -dy), max(0, dx):SIZE - max(0, -dx)] = chart[max(0, -dy):SIZE - max(0, dy), max(0, -dx):SIZE - max(0, dx)]
        candidate &= (neigh < 0) | (neigh == donor_chart)
    candidate &= donor_chart >= 0
    rejection[prechart & ~candidate] |= REJECT_CHART
    (cy, cx) = np.nonzero(candidate)
    did = donor_y[cy, cx].astype(np.int64) * SIZE + donor_x[cy, cx]
    dfi = primitive.reshape(-1)[did]
    tri_uv = uvcpu[fcpu[dfi]].astype(np.float64)
    point_uv = np.stack(((cx + 0.5) / SIZE, (cy + 0.5) / SIZE), 1).astype(np.float64)
    basis = tri_uv[:, :2] - tri_uv[:, 2:3]
    det = basis[:, 0, 0] * basis[:, 1, 1] - basis[:, 0, 1] * basis[:, 1, 0]
    finite_face = np.isfinite(tri_uv).all((1, 2)) & (np.abs(det) > 1e-12)
    bary = np.full((len(cy), 3), np.nan, np.float32)
    good = finite_face
    delta = point_uv[good] - tri_uv[good, 2]
    bary[good, :2] = np.linalg.solve(np.swapaxes(basis[good], 1, 2), delta[..., None])[..., 0]
    bary[good, 2] = 1 - bary[good, 0] - bary[good, 1]
    good &= np.isfinite(bary).all(1) & (np.abs(bary).max(1) <= 8.0)
    rejection[cy[~good], cx[~good]] |= REJECT_DEGENERATE_UV
    tri_native = vertices[fcpu[dfi]].astype(np.float64) * np.array((-1.0, 1.0, -1.0), np.float64)
    xyz = (tri_native * bary[:, :, None]).sum(1)
    donor_uv = np.stack(((donor_x[cy, cx] + 0.5) / SIZE, (donor_y[cy, cx] + 0.5) / SIZE), 1)
    donor_delta = donor_uv - tri_uv[:, 2]
    donor_bary = np.full((len(cy), 3), np.nan, np.float32)
    donor_bary[good, :2] = np.linalg.solve(np.swapaxes(basis[good], 1, 2), donor_delta[good, ..., None])[..., 0]
    donor_bary[good, 2] = 1 - donor_bary[good, 0] - donor_bary[good, 1]
    good &= np.isfinite(donor_bary).all(1) & (np.abs(donor_bary).max(1) <= 8.0)
    donor_contained = (donor_bary >= -1e-05).all(1) & (donor_bary <= 1 + 1e-05).all(1)
    good &= donor_contained
    rejection[cy[~donor_contained], cx[~donor_contained]] |= REJECT_DEGENERATE_UV
    reconstructed_donor_uv = (tri_uv * donor_bary[:, :, None]).sum(1)
    donor_uv_error = np.abs(reconstructed_donor_uv - donor_uv).max(1)
    good &= donor_uv_error <= 1e-06
    rejection[cy[~good], cx[~good]] |= REJECT_DEGENERATE_UV
    donor_xyz = (tri_native * donor_bary[:, :, None]).sum(1)
    (atlas_native, _) = dr.interpolate(native[None], urast, faces)
    donor_surface = atlas_native[0, torch.as_tensor(donor_y[cy, cx], device='cuda'), torch.as_tensor(donor_x[cy, cx], device='cuda')].cpu().numpy()
    del atlas_native
    donor_surface_error = np.abs(donor_surface - donor_xyz).max(1)
    surface_checked = good.copy()
    surface_ok = donor_surface_error <= 1e-05
    good &= surface_ok
    rejection[cy[~surface_ok], cx[~surface_ok]] |= REJECT_DEGENERATE_UV

    def project_native(points):
        points_t = torch.as_tensor(points, dtype=torch.float32, device='cuda')
        q = torch.cat((points_t, torch.ones_like(points_t[:, :1])), 1) @ E.T
        q = q[:, :3] @ K.T
        return ((q[:, :2] / q[:, 2:3]).cpu().numpy(), q[:, 2].cpu().numpy())
    (source_uv, source_depth) = project_native(xyz)
    (donor_source_uv, donor_depth) = project_native(donor_xyz)
    cell = np.floor(np.where(np.isfinite(source_uv), source_uv, -2.0) * RES - 0.5).astype(np.int64)
    donor_cell = np.floor(np.where(np.isfinite(donor_source_uv), donor_source_uv, -2.0) * RES - 0.5).astype(np.int64)
    footprint_ok = good & np.isfinite(source_uv).all(1) & np.isfinite(donor_source_uv).all(1) & (source_depth > 0) & (donor_depth > 0)
    footprint_ok &= (cell[:, 0] >= 0) & (cell[:, 0] + 1 < RES) & (cell[:, 1] >= 0) & (cell[:, 1] + 1 < RES)
    footprint_ok &= (donor_cell[:, 0] >= 0) & (donor_cell[:, 0] + 1 < RES) & (donor_cell[:, 1] >= 0) & (donor_cell[:, 1] + 1 < RES)
    fi = np.flatnonzero(footprint_ok)
    if len(fi):
        footprint_ok[fi] &= (hard[cell[fi, 1], cell[fi, 0]] & hard[cell[fi, 1], cell[fi, 0] + 1]
                             & hard[cell[fi, 1] + 1, cell[fi, 0]] & hard[cell[fi, 1] + 1, cell[fi, 0] + 1]
                             & hard[donor_cell[fi, 1], donor_cell[fi, 0]] & hard[donor_cell[fi, 1], donor_cell[fi, 0] + 1]
                             & hard[donor_cell[fi, 1] + 1, donor_cell[fi, 0]] & hard[donor_cell[fi, 1] + 1, donor_cell[fi, 0] + 1])
    donor_strength = rt['F'].grid_sample(trust_t, (torch.as_tensor(donor_source_uv, dtype=torch.float32, device='cuda') * 2 - 1).reshape(1, 1, -1, 2), align_corners=False, padding_mode='zeros')[0, 0, 0].cpu().numpy()
    endpoint_strength = rt['F'].grid_sample(trust_t, (torch.as_tensor(source_uv, dtype=torch.float32, device='cuda') * 2 - 1).reshape(1, 1, -1, 2), align_corners=False, padding_mode='zeros')[0, 0, 0].cpu().numpy()
    footprint_ok &= (donor_strength >= 1 - 1e-06) & (endpoint_strength >= 1 - 1e-06)
    sample_uv = torch.as_tensor(source_uv[footprint_ok], dtype=torch.float32, device='cuda')
    donor_sample_uv = torch.as_tensor(donor_source_uv[footprint_ok], dtype=torch.float32, device='cuda')
    sampled = rt['F'].grid_sample(src, (sample_uv * 2 - 1).reshape(1, 1, -1, 2), align_corners=False, padding_mode='zeros')[0, :, 0].T.cpu().numpy()
    donor_sampled = rt['F'].grid_sample(src, (donor_sample_uv * 2 - 1).reshape(1, 1, -1, 2), align_corners=False, padding_mode='zeros')[0, :, 0].T.cpu().numpy()
    source_rows = np.flatnonzero(footprint_ok)
    donor_base = base_before[donor_y[cy[source_rows], cx[source_rows]], donor_x[cy[source_rows], cx[source_rows]], :3].astype(np.float32) / 255
    donor_actual = before[donor_y[cy[source_rows], cx[source_rows]], donor_x[cy[source_rows], cx[source_rows]], :3].astype(np.float32) / 255
    donor_rebuilt = donor_base + donor_strength[source_rows, None] * (donor_sampled - donor_base)
    donor_project_byte_error = np.full(len(cy), np.inf, np.float32)
    donor_project_byte_error[source_rows] = np.abs(donor_rebuilt - donor_actual).max(1) * 255
    donor_project_ok = donor_project_byte_error <= 1.0
    source_rejected = ~footprint_ok | footprint_ok & ~donor_project_ok
    rejection[cy[source_rejected], cx[source_rejected]] |= REJECT_SOURCE
    continuous_ok = footprint_ok & donor_project_ok
    eligible = np.zeros_like(base)
    eligible[cy[continuous_ok], cx[continuous_ok]] = True
    after = before.copy()
    after[cy, cx, :3] = before[donor_y[cy, cx], donor_x[cy, cx], :3]
    if not np.array_equal(after[cy, cx, :3], before[donor_y[cy, cx], donor_x[cy, cx], :3]):
        raise RuntimeError('projection donor byte fallback failed')
    (ey, ex) = np.nonzero(eligible)
    anchor = before[donor_y[ey, ex], donor_x[ey, ex], :3].astype(np.float32) / 255
    selected_rows = np.flatnonzero(continuous_ok)
    sample_indices = np.searchsorted(source_rows, selected_rows)
    after[ey, ex, :3] = np.clip((anchor + sampled[sample_indices] - donor_sampled[sample_indices]) * 255 + 0.5, 0, 255).astype(np.uint8)
    if not np.array_equal(after[covered], before[covered]):
        raise RuntimeError('UV covered preservation failed')
    if after.shape[-1] == 4 and (not np.array_equal(after[..., 3], before[..., 3])):
        raise RuntimeError('alpha changed')
    if not np.array_equal(after.reshape(-1, after.shape[-1])[active | blocked | unsafe], before.reshape(-1, before.shape[-1])[active | blocked | unsafe]):
        raise RuntimeError('active, blocked, or unsafe texels changed')
    eligible_ids = np.flatnonzero(eligible.reshape(-1))
    np.savez_compressed(
        a.output / 'GUTTER_DIAGNOSTICS.npz',
        covered=covered,
        unsafe=unsafe.reshape(SIZE, SIZE),
        candidate=candidate,
        eligible=eligible,
        rejection=rejection,
        eligible_texel_ids=eligible_ids,
        donor_texel_ids=did[continuous_ok],
        donor_charts=donor_chart.reshape(-1)[eligible_ids],
        source_uv=source_uv[continuous_ok],
        donor_source_uv=donor_source_uv[continuous_ok],
        barycentric=bary[continuous_ok],
        donor_barycentric=donor_bary[continuous_ok],
        donor_uv_reconstruction_error=donor_uv_error[continuous_ok],
        donor_surface_error=donor_surface_error[continuous_ok],
        donor_strength=donor_strength[continuous_ok],
        endpoint_strength=endpoint_strength[continuous_ok],
        donor_project_byte_error=donor_project_byte_error[continuous_ok],
        gutter_radius=np.asarray(a.gutter_radius, dtype=np.int32),
    )
    blob = io.BytesIO()
    Image.fromarray(after).save(blob, format='PNG')
    odoc = copy.deepcopy(pdoc)
    obin = fixed._replace_image_payload(odoc, pbin, vi, off, n, blob.getvalue())
    glb = a.output / 'candidate/mesh.glb'
    glb.parent.mkdir()
    fixed._write_glb(glb, odoc, obin)
    preservation = fixed._nonbase_identity_check(pdoc, pbin, glb, vi)

    def render(o, split):
        p = o['phase']
        folder = a.output / 'renders' / split / f'x{p[0]:.3f}_y{p[1]:.3f}'
        folder.mkdir(parents=True)
        cam = copy.deepcopy(camera)
        cam['front_atlas_phase'] = p
        ca = rm.render_glb(glb, cam, folder / 'repair.png', resolution=RES, rt=rt)
        pa = rm.render_glb(pglb, cam, folder / 'project.png', resolution=RES, rt=rt)
        with np.load(ca['attrs_npz']) as z:
            (ci, cm, cmra) = (z['albedo'], z['mask'].astype(bool), z['mra'])
        with np.load(pa['attrs_npz']) as z:
            (bi, bm, bmra) = (z['albedo'], z['mask'].astype(bool), z['mra'])
        if not np.array_equal(cm, bm) or not np.array_equal(cmra, bmra):
            raise RuntimeError('mask/MRA changed')
        target = np.zeros_like(ci)
        target.reshape(-1, 3)[o['rows'].cpu().numpy()] = o['target'].cpu().numpy()
        sel = o['screen'] & cm
        quant = torch.as_tensor(after[..., :3], dtype=torch.float32, device='cuda').reshape(-1, 3) / 255
        replay = (quant[o['idx']] * o['w'][..., None]).sum(1).cpu().numpy()
        actual = ci.reshape(-1, 3)[o['rows'].cpu().numpy()]
        outside = cm & ~o['screen']
        outdelta = float(np.abs(ci - bi)[outside].max()) if outside.any() else 0.0
        changed = cm & (np.abs(ci - bi).max(2) > 0)
        r = {
            'phase': p,
            'hardtrust_pixels': int(sel.sum()),
            'repair_vs_source': rm._masked_metrics(ci, target, sel),
            'project_vs_source': rm._masked_metrics(bi, target, sel),
            'repair_vs_project_changed_pixels': int(changed.sum()),
            'repair_vs_project': rm._masked_metrics(ci, bi, changed) if changed.any() else {'valid_pixels': 0},
            'quantized_atlas_replay_max': float(np.abs(replay - actual).max()),
            'outside_source_hardtrust_max_abs_threshold': OUTSIDE_COLOR_MAX_ABS,
            'outside_source_hardtrust_delta_max_abs': outdelta,
        }
        r['passes'] = bool(r['quantized_atlas_replay_max'] <= 1e-05 and outside_color_passes(outdelta))
        r.update(mask_exact=True, mra_exact=True, evidence={'repair': bind_file(Path(ca['attrs_npz'])), 'project': bind_file(Path(pa['attrs_npz']))})
        write_once(folder / 'PHASE_RECEIPT.json', r)
        return r
    results = [render(x, 'grid' if i < len(GRID_PHASES) else 'offgrid') for (i, x) in enumerate(operations)]
    changed = np.any(after[..., :3] != before[..., :3], axis=-1)
    accepted = all((x['passes'] for x in results))
    gate_ok = accepted and all((preservation.get(k) for k in IDENTITY_GATES))
    receipt = {
        'status': 'valid' if gate_ok else 'invalid',
        'schema': SCHEMA,
        'candidate_glb': str(glb),
        'seconds': time.monotonic() - started,
        'gates': {
            'accepted': gate_ok,
            'UV_covered_exact': True,
            'project_active_exact': True,
            'blocked_exact': True,
            'alpha_exact': True,
            'nonbase_identity': preservation,
            'all_19_phase_renders': len(results) == 19,
        },
        'atlas': {
            'UV_covered_texels': int(covered.sum()),
            'UV_face_charts': int(face_chart.max() + 1),
            'project_active_texels': int(active.sum()),
            'blocked_texels': int(blocked.sum()),
            'unsafe_all_phase_texels': int(unsafe.sum()),
            'gutter_radius': a.gutter_radius,
            'nearest_donor_uncovered_gutter_texels': int(candidate.sum()),
            'continuous_eligible_uncovered_gutter_texels': int(eligible.sum()),
            'changed_texels': int(changed.sum()),
            'rejected_unsafe': int((rejection & REJECT_UNSAFE != 0).sum()),
            'rejected_chart': int((rejection & REJECT_CHART != 0).sum()),
            'rejected_degenerate_uv': int((rejection & REJECT_DEGENERATE_UV != 0).sum()),
            'rejected_source_footprint_or_feather': int((rejection & REJECT_SOURCE != 0).sum()),
            'donor_surface_oracle': {
                'threshold': 1e-05,
                'checked': int(surface_checked.sum()),
                'rejected': int((surface_checked & ~surface_ok).sum()),
                'pre_filter_max': float(donor_surface_error[surface_checked].max()) if surface_checked.any() else 0.0,
                'pre_filter_p99': float(np.quantile(donor_surface_error[surface_checked], 0.99)) if surface_checked.any() else 0.0,
                'accepted_max': float(donor_surface_error[footprint_ok].max()) if footprint_ok.any() else 0.0,
            },
        },
        'phase_results': {'grid': results[:len(GRID_PHASES)], 'offgrid': results[len(GRID_PHASES):]},
        'diagnostics': str(a.output / 'GUTTER_DIAGNOSTICS.npz'),
        'code_snapshot': str(a.output / 'CODE_SNAPSHOT.py'),
    }
    write_once(a.output / 'RECEIPT.json', receipt)
    write_once(a.output / 'RESULT.json', receipt)
    if not gate_ok:
        raise RuntimeError('acceptance failed')
    print(json.dumps({'status': 'valid', 'candidate': str(glb), 'changed_texels': int(changed.sum())}))
if __name__ == '__main__':
    main()
