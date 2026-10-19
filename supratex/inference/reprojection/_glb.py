"""Strict fixed-layout GLB atlas replacement with binary preservation checks."""
from __future__ import annotations
import json
import os
import struct
from pathlib import Path
from typing import Any
GLB_JSON = 0x4E4F534A
GLB_BIN = 0x004E4942
ATLAS_SIZE = 4096

def _round4(data: bytes, fill: bytes=b'\x00') -> bytes:
    return data + fill * (-len(data) % 4)

def _read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError(f'truncated GLB: {path}')
    (magic, version, declared) = struct.unpack_from('<4sII', raw, 0)
    if magic != b'glTF' or version != 2 or declared != len(raw):
        raise ValueError(f'invalid GLB header: {path}')
    (offset, chunks) = (12, [])
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ValueError(f'truncated GLB chunk header: {path}')
        (length, kind) = struct.unpack_from('<II', raw, offset)
        offset += 8
        if length % 4 or offset + length > len(raw):
            raise ValueError(f'invalid GLB chunk bounds: {path}')
        chunks.append((kind, raw[offset:offset + length]))
        offset += length
    if offset != len(raw) or [kind for (kind, _) in chunks] != [GLB_JSON, GLB_BIN]:
        raise ValueError('fixed UV bake requires exactly JSON then BIN GLB chunks')
    document = json.loads(chunks[0][1].rstrip(b' \t\r\n\x00'))
    return (document, chunks[1][1])

def _write_glb(path: Path, document: dict[str, Any], binary: bytes) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite output: {path}')
    json_bytes = _round4(json.dumps(document, separators=(',', ':'), ensure_ascii=True).encode('utf-8'), b' ')
    binary = _round4(binary)
    payload = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(json_bytes) + 8 + len(binary)) + struct.pack('<II', len(json_bytes), GLB_JSON) + json_bytes + struct.pack('<II', len(binary), GLB_BIN) + binary
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_bytes(payload)
    os.replace(temporary, path)

def _base_color_image_ref(document: dict[str, Any]) -> tuple[int, int]:
    meshes = document.get('meshes')
    materials = document.get('materials')
    textures = document.get('textures')
    images = document.get('images')
    if not all((isinstance(value, list) for value in (meshes, materials, textures, images))):
        raise ValueError('GLB lacks mesh/material/image arrays')
    refs: set[int] = set()
    for mesh in meshes:
        for primitive in mesh.get('primitives', []):
            material_index = primitive.get('material')
            if not isinstance(material_index, int) or not 0 <= material_index < len(materials):
                raise ValueError('every fixed-UV primitive must have a valid material')
            reference = materials[material_index].get('pbrMetallicRoughness', {}).get('baseColorTexture', {})
            texture_index = reference.get('index')
            if not isinstance(texture_index, int) or not 0 <= texture_index < len(textures):
                raise ValueError('primitive material lacks a valid baseColorTexture')
            refs.add(texture_index)
    if len(refs) != 1:
        raise ValueError('fixed-UV bake currently requires one shared baseColorTexture')
    texture_index = refs.pop()
    image_index = textures[texture_index].get('source')
    if not isinstance(image_index, int) or not 0 <= image_index < len(images):
        raise ValueError('baseColorTexture source is invalid')
    view_index = images[image_index].get('bufferView')
    if not isinstance(view_index, int):
        raise ValueError('baseColor image must be embedded in a bufferView')
    return (image_index, view_index)

def _assert_fixed_scene(document: dict[str, Any]) -> None:
    """Reject transforms rather than silently baking through an unknown scene graph."""
    (meshes, nodes, scenes) = (document.get('meshes'), document.get('nodes'), document.get('scenes'))
    if not (isinstance(meshes, list) and len(meshes) == 1 and isinstance(nodes, list) and isinstance(scenes, list)):
        raise ValueError('fixed-UV bake requires exactly one mesh and explicit nodes/scenes')
    for node in nodes:
        if any((key in node for key in ('matrix', 'translation', 'rotation', 'scale'))):
            raise ValueError('fixed-UV bake rejects node transforms; apply/validate scene transform first')
    primitives = meshes[0].get('primitives', [])
    if len(primitives) != 1 or primitives[0].get('mode', 4) != 4:
        raise ValueError('reprojection requires one indexed triangle primitive')
    if len(document.get('buffers', [])) != 1 or document['buffers'][0].get('uri'):
        raise ValueError('reprojection requires one embedded binary buffer')

def _decode_base_image(document: dict[str, Any], binary: bytes, image_index: int, view_index: int) -> tuple[Any, bytes, int, int]:
    import io
    from PIL import Image
    view = document['bufferViews'][view_index]
    (offset, length) = (int(view.get('byteOffset', 0)), int(view['byteLength']))
    payload = binary[offset:offset + length]
    if len(payload) != length or document['images'][image_index].get('mimeType') != 'image/png':
        raise ValueError('baseColorTexture must be an embedded PNG')
    image = Image.open(io.BytesIO(payload))
    if image.mode not in ('RGB', 'RGBA'):
        raise ValueError(f'baseColor PNG must be RGB or RGBA for this control, got {image.mode}')
    if image.size != (ATLAS_SIZE, ATLAS_SIZE):
        raise ValueError(f'fixed-UV control requires a {ATLAS_SIZE}px base-color atlas, got {image.size}')
    return (image, payload, offset, length)

def _nonbase_identity_check(source_document: dict[str, Any], source_binary: bytes, baked_glb: Path, base_view: int) -> dict[str, Any]:
    """Prove the repack did not alter non-base material or binary payloads."""
    (baked_document, baked_binary) = _read_glb(baked_glb)
    if source_document.get('materials') != baked_document.get('materials'):
        raise RuntimeError('fixed-UV bake changed GLB materials')
    if source_document.get('meshes') != baked_document.get('meshes') or source_document.get('nodes') != baked_document.get('nodes'):
        raise RuntimeError('fixed-UV bake changed GLB mesh or node structure')
    for key in set(source_document) | set(baked_document):
        if key not in ('buffers', 'bufferViews') and source_document.get(key) != baked_document.get(key):
            raise RuntimeError(f'fixed-UV bake changed GLB document field {key}')
    (source_views, baked_views) = (source_document['bufferViews'], baked_document['bufferViews'])
    if len(source_views) != len(baked_views):
        raise RuntimeError('fixed-UV bake changed bufferView count')
    checked = 0
    for (index, (source_view, baked_view)) in enumerate(zip(source_views, baked_views)):
        if index == base_view:
            continue
        if {k: v for k, v in source_view.items() if k != 'byteOffset'} != {k: v for k, v in baked_view.items() if k != 'byteOffset'}:
            raise RuntimeError(f'fixed-UV bake changed bufferView metadata {index}')
        source_start = int(source_view.get('byteOffset', 0))
        source_end = source_start + int(source_view['byteLength'])
        baked_start = int(baked_view.get('byteOffset', 0))
        baked_end = baked_start + int(baked_view['byteLength'])
        if source_binary[source_start:source_end] != baked_binary[baked_start:baked_end]:
            raise RuntimeError(f'fixed-UV bake changed non-base bufferView {index}')
        checked += 1
    return {'materials_exact': True, 'mesh_structure_exact': True, 'node_structure_exact': True, 'nonbase_buffer_views_exact': True, 'nonbase_buffer_view_count': checked}

def _replace_image_payload(document: dict[str, Any], binary: bytes, view_index: int, old_offset: int, old_length: int, png: bytes) -> bytes:
    """Replace one image while preserving every non-image binary byte exactly."""
    padded = _round4(png)
    old_padded_length = old_length + 3 & ~3
    if binary[old_offset + old_length:old_offset + old_padded_length] not in (b'', b'\x00' * (old_padded_length - old_length)):
        raise ValueError('base-color bufferView has non-zero alignment bytes')
    delta = len(padded) - old_padded_length
    rebuilt = binary[:old_offset] + padded + binary[old_offset + old_padded_length:]
    for (index, view) in enumerate(document['bufferViews']):
        offset = int(view.get('byteOffset', 0))
        if index == view_index:
            view['byteLength'] = len(png)
        elif offset > old_offset:
            view['byteOffset'] = offset + delta
    document['buffers'][0]['byteLength'] = len(rebuilt)
    return rebuilt
