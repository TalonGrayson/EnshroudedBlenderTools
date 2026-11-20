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


# "No Tint": a tint layer mask that is black everywhere, so no tint layer applies and the albedo shows as painted.
# Masks are sampled by UV, so a tiny texture works; BC4 stores one channel, and an all-zero 8-byte block decodes to 0.
BLACK_MASK_SIZE = 4


def black_bc4_chain(size: int = BLACK_MASK_SIZE) -> bytes:
    """Full BC4 mip chain (size x size down to 1x1) of value 0."""
    if size < 1 or size & (size - 1):
        raise ValueError("mask size must be a power of two")
    data = b""
    for level in range(size.bit_length()):
        width = max(1, size >> level)
        data += bytes(8 * ((width + 3) // 4) ** 2)
    return data


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
