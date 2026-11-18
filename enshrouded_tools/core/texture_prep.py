"""Prepare source images for texconv.

texconv (DirectXTex/WIC) reads a PNG's sRGB/gAMA/iCCP/cHRM chunks and then treats
the pixels as sRGB. Compressing such a file to a linear format (BC1_UNORM for
material maps, BC5_UNORM for normal maps) silently converts sRGB -> linear, which
darkens the data: a metal/roughness/AO map loses metalness and normals bend.
Blender writes those chunks even for Non-Color images, so linear maps must be
handed to texconv without them.
"""
from __future__ import annotations

import struct

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLOR_SPACE_CHUNKS = frozenset({b"sRGB", b"gAMA", b"iCCP", b"cHRM"})


def strip_png_color_space(data: bytes) -> bytes:
    """Return the PNG without colour-space chunks; pixel data is untouched."""
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG file")
    out = bytearray(PNG_SIGNATURE)
    pos = 8
    while pos < len(data):
        if pos + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        end = pos + 12 + length
        if end > len(data):
            raise ValueError("truncated PNG chunk")
        if kind not in COLOR_SPACE_CHUNKS:
            out += data[pos:end]
        pos = end
        if kind == b"IEND":
            break
    return bytes(out)
