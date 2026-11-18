"""Encode Blender vertex weights into a KFC4 RenderModel skinning stream.

Stream layout (verified on vanilla helmets, build 1287369): one 8-byte record per
exported vertex, in vertexData order: 4 x uint8 index into ``skinningMap`` (the
model's bone palette), then 4 x uint8 weights summing to exactly 255.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_INFLUENCES = 4
WEIGHT_TOTAL = 255

_ADDON_NAME = re.compile(r'joint_(\d+)_([0-9a-fA-F]{8})')   # this add-on's KFC import
_GLTF_NAME = re.compile(r'j_([0-9a-fA-F]{8})')              # Embervale / glTF armatures
IGNORED_GROUPS = frozenset({'root'})


@dataclass(frozen=True)
class SkinningStream:
    skinning_map: tuple[int, ...]   # skeleton joint index per palette slot
    data: bytes                     # 8 bytes per vertex


def resolve_joint(name: str, hashes) -> int | None:
    """Map a vertex-group name to a skeleton joint index.

    Returns None for groups that never carry skin weight (``root``) and for
    names that are not joint names at all. Raises if a joint-style name does
    not exist in the target skeleton.
    """
    if name in IGNORED_GROUPS:
        return None
    match = _ADDON_NAME.fullmatch(name) or _GLTF_NAME.fullmatch(name)
    if match is None:
        return None
    joint_hash = int(match.group(match.lastindex), 16)
    if match.re is _ADDON_NAME:
        index = int(match.group(1))
        if index < len(hashes) and hashes[index] == joint_hash:
            return index
    try:
        return hashes.index(joint_hash)
    except ValueError:
        raise ValueError(f'vertex group {name!r} is not a joint of the target skeleton') from None


def quantize(influences):
    """Top-4 influences as (joint, uint8 weight) with weights summing to 255."""
    merged = {}
    for joint, weight in influences:
        if weight > 0.0:
            merged[joint] = merged.get(joint, 0.0) + float(weight)
    top = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:MAX_INFLUENCES]
    total = sum(weight for _joint, weight in top)
    if total <= 0.0:
        return ()
    scaled = [(joint, weight * WEIGHT_TOTAL / total) for joint, weight in top]
    rounded = [(joint, int(round(value))) for joint, value in scaled]
    # Put the rounding remainder on the strongest influence so the sum is exact.
    remainder = WEIGHT_TOTAL - sum(value for _joint, value in rounded)
    rounded[0] = (rounded[0][0], rounded[0][1] + remainder)
    return tuple((joint, value) for joint, value in rounded if value > 0)


def encode_skinning(vertex_influences, fallback_joint: int | None = None) -> SkinningStream:
    """Build the palette and per-vertex records.

    ``vertex_influences`` holds one iterable of (skeleton joint, weight) per
    exported vertex. Vertices without weight use ``fallback_joint`` or fail.
    """
    quantized = []
    unweighted = 0
    for influences in vertex_influences:
        record = quantize(influences)
        if not record:
            if fallback_joint is None:
                unweighted += 1
            else:
                record = ((fallback_joint, WEIGHT_TOTAL),)
        quantized.append(record)
    if unweighted:
        raise ValueError(f'{unweighted} exported vertices have no bone weight; weight them '
                         'or set "Weight unweighted vertices to bone"')
    palette = tuple(sorted({joint for record in quantized for joint, _w in record}))
    if not palette:
        raise ValueError('skinned export has no weighted joints')
    if len(palette) > 256:
        raise ValueError('skinned export uses more than 256 joints')
    slot = {joint: index for index, joint in enumerate(palette)}
    data = bytearray()
    for record in quantized:
        bones = [slot[joint] for joint, _w in record] + [0] * (MAX_INFLUENCES - len(record))
        weights = [value for _j, value in record] + [0] * (MAX_INFLUENCES - len(record))
        data += bytes(bones) + bytes(weights)
    return SkinningStream(palette, bytes(data))
