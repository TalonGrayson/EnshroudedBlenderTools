"""Cloth simulation data of a KFC4 keen::RenderModel (read side).

Layout (build 1269579, from ShroudForge's types.lua and vanilla models such as banner_dlc_01):
  RenderModel @220  cloth: keen::RenderModelClothData?  (u32 relative pointer, 0 = no cloth)
  RenderModel @224  chain: keen::RenderModelChainData?  (u32 relative pointer, 0 = no chain)
  RenderModelClothData (20 bytes):
     +0  hasSkinning bool (padded to 4)
     +4  meshes Array<RenderClothMesh>   (u32 relative offset, u32 count)
     +12 totalSimulationTriangleCount u32
     +16 totalRenderVertexCount u32
  RenderClothMesh (232 bytes): RenderClothMeshResource (4 + 20 arrays * 8 = 164 bytes) + 17 f32 parameters.
  RenderClothMeshResource:
     fabric: numParticles u32, phaseIndices, sets, restValues, stiffnessValues, indices, anchors, tetherLengths, triangles
     simulationPositions, simulationInvMasses, simulationStiffnesses, simulationVertexWeights, simulationTriangles,
     simulationBlendshapeDeltas, simulationSkinningData, skinnedVertexIndices, renderPositions, renderUvs,
     renderTriangles, renderSimulationVertexIndices
The fabric arrays follow NvCloth's ClothFabric layout (phases -> sets -> distance constraints, plus tethers).
A mesh's RenderModelMesh.clothMeshIndex byte (mesh record +54) is 1-based: 0 = not cloth.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

CLOTH_POINTER_OFFSET = 220
CHAIN_POINTER_OFFSET = 224
CLOTH_MESH_SIZE = 232

PARAMETER_NAMES = (
    "gravity", "windStrength", "dragCoefficient", "liftCoefficient", "damping", "linearInertia",
    "angularInertia", "centrifugalInertia", "tetherStiffness", "phasesStiffness", "phasesCompressionLimit",
    "phasesStretchLimit", "motionConstraintsStiffness", "collisionFriction", "solverFrequency",
    "motionConstraintsUnlimitedRadius", "motionConstraintsMaxRadius",
)

# (field name, struct format of one element)
FABRIC_ARRAYS = (
    ("phaseIndices", "I"), ("sets", "I"), ("restValues", "f"), ("stiffnessValues", "f"), ("indices", "I"),
    ("anchors", "I"), ("tetherLengths", "f"), ("triangles", "I"),
)
MESH_ARRAYS = (
    ("simulationPositions", "3f"), ("simulationInvMasses", "f"), ("simulationStiffnesses", "f"),
    ("simulationVertexWeights", "f"), ("simulationTriangles", "3I"), ("simulationBlendshapeDeltas", "H"),
    ("simulationSkinningData", "8B"), ("skinnedVertexIndices", "H"), ("renderPositions", "3f"),
    ("renderUvs", "2f"), ("renderTriangles", "3I"), ("renderSimulationVertexIndices", "H"),
)


@dataclass
class ClothFabric:
    numParticles: int
    phaseIndices: list = field(default_factory=list)
    sets: list = field(default_factory=list)
    restValues: list = field(default_factory=list)
    stiffnessValues: list = field(default_factory=list)
    indices: list = field(default_factory=list)
    anchors: list = field(default_factory=list)
    tetherLengths: list = field(default_factory=list)
    triangles: list = field(default_factory=list)


@dataclass
class ClothMesh:
    fabric: ClothFabric
    arrays: dict
    parameters: dict


@dataclass
class ClothData:
    has_skinning: bool
    meshes: list
    total_simulation_triangles: int
    total_render_vertices: int


def _read_array(payload: bytes, field_offset: int, fmt: str) -> list:
    relative, count = struct.unpack_from("<II", payload, field_offset)
    if not count:
        return []
    start = field_offset + relative
    size = struct.calcsize("<" + fmt)
    if start + count * size > len(payload):
        raise ValueError(f"cloth array at 0x{field_offset:x} exceeds descriptor ({count} x {size} bytes)")
    if len(fmt) == 1:
        return list(struct.unpack_from(f"<{count}{fmt}", payload, start))
    return [struct.unpack_from("<" + fmt, payload, start + i * size) for i in range(count)]


def cloth_offset(payload: bytes) -> int:
    """Absolute offset of RenderModelClothData, or 0 when the model has no cloth."""
    relative = struct.unpack_from("<I", payload, CLOTH_POINTER_OFFSET)[0]
    return CLOTH_POINTER_OFFSET + relative if relative else 0


def has_chain(payload: bytes) -> bool:
    return struct.unpack_from("<I", payload, CHAIN_POINTER_OFFSET)[0] != 0


def parse_cloth_mesh(payload: bytes, offset: int) -> ClothMesh:
    fabric = ClothFabric(struct.unpack_from("<I", payload, offset)[0])
    cursor = offset + 4
    for name, fmt in FABRIC_ARRAYS:
        setattr(fabric, name, _read_array(payload, cursor, fmt))
        cursor += 8
    arrays = {}
    for name, fmt in MESH_ARRAYS:
        arrays[name] = _read_array(payload, cursor, fmt)
        cursor += 8
    parameters = dict(zip(PARAMETER_NAMES, struct.unpack_from("<17f", payload, cursor)))
    return ClothMesh(fabric, arrays, parameters)


def parse_cloth_data(payload: bytes):
    """Return ClothData for a KFC4 RenderModel descriptor, or None when it has no cloth."""
    base = cloth_offset(payload)
    if not base:
        return None
    has_skinning = payload[base] != 0
    relative, count = struct.unpack_from("<II", payload, base + 4)
    start = base + 4 + relative
    meshes = [parse_cloth_mesh(payload, start + i * CLOTH_MESH_SIZE) for i in range(count)]
    total_triangles, total_vertices = struct.unpack_from("<II", payload, base + 12)
    return ClothData(has_skinning, meshes, total_triangles, total_vertices)
