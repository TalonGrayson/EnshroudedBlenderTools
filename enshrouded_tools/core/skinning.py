"""Four-influence skinning, verified on armor_m_ear_garments_01_chest (KFC3) and
armor_w_mid_adventurer_01_head / armor_v_t7_rosewarrior_01_head (KFC4)."""
import struct
import uuid
from collections import namedtuple
from .collision import _relative_array

HIERARCHY_TYPE_HASH = 897872094

# RenderModel descriptor field offsets per archive format.
#   skeleton: 16-byte hierarchy GUID
#   range:    skinningData {start, count} in renderData
#   palette:  relative array of u16 skeleton joint indices
#   counts:   u16 palette count, u16 skeleton joint count
SkinningLayout = namedtuple('SkinningLayout', 'min_size skeleton range palette counts')
LAYOUTS = {
    'KFC3': SkinningLayout(316, 148, 164, 260, 312),
    'KFC4': SkinningLayout(352, 128, 328, 212, 264),
}

RECORD_SIZE = 8  # 4 x uint8 palette index + 4 x uint8 weight (sum 255)


def joint_group_name(skeleton_index, joint_hash):
    return f'joint_{skeleton_index:03d}_{joint_hash:08x}'


def read_skinning(reader, descriptor, model, render_data, archive_format='KFC3'):
    try:
        layout = LAYOUTS[archive_format]
    except KeyError:
        raise ValueError(f'Skinning not supported for {archive_format}') from None
    if len(descriptor) < layout.min_size:
        return None
    offset, size = struct.unpack_from('<II', descriptor, layout.range)
    if not size:
        return None
    count = sum(v.data_size // v.stride for v in model.vertex_buffers)
    if size != count * RECORD_SIZE or offset + size > len(render_data):
        raise ValueError('Unsupported skinning stream size')
    joint_count, hierarchy_count = struct.unpack_from('<HH', descriptor, layout.counts)
    start, map_count = _relative_array(descriptor, layout.palette, 2)
    if map_count != joint_count:
        raise ValueError('Skinning joint count does not match palette')
    palette = struct.unpack_from(f'<{map_count}H', descriptor, start)
    guid = str(uuid.UUID(bytes_le=bytes(descriptor[layout.skeleton:layout.skeleton + 16])))
    hierarchy = reader.read_resource(reader.find_resource_index(guid, HIERARCHY_TYPE_HASH))
    start, name_count = _relative_array(hierarchy, 16, 4)
    if name_count != hierarchy_count or any(j >= name_count for j in palette):
        raise ValueError('Skinning palette exceeds hierarchy')
    hashes = struct.unpack_from(f'<{name_count}I', hierarchy, start)
    names = tuple(joint_group_name(j, hashes[j]) for j in palette)
    weights = []
    for vertex in range(count):
        record = render_data[offset + vertex * RECORD_SIZE:offset + (vertex + 1) * RECORD_SIZE]
        if sum(record[4:]) != 255:
            raise ValueError('Unsupported skinning weight normalization')
        influences = {}
        for bone, weight in zip(record[:4], record[4:]):
            if not weight:
                continue
            if bone >= map_count:
                raise ValueError('Skinning bone index exceeds palette')
            influences[bone] = influences.get(bone, 0) + weight / 255.0
        weights.append(tuple(influences.items()))
    return guid, names, tuple(weights)
