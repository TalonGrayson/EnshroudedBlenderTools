"""Keep a model's cloth working when its vertices are moved (topology-preserving Replacement export).

Vanilla layout (collab/dev_tools/cloth_layout.py): every cloth part is its own RenderModel mesh with its own vertex
range; mesh record +54 (clothMeshIndex) is 1-based. That range's vertices are the cloth's renderPositions in the same
order, and renderSimulationVertexIndices maps each of them to a simulation particle.

When the Blender mesh only moves vertices, the vanilla fabric (which particles are linked, pins, masses, skinning,
solver settings) stays valid; only positions and lengths change. So this keeps every constraint and tether anchor
and recomputes:
  renderPositions       - the moved cloth vertices
  simulationPositions   - per particle, the mean of its render vertices
  fabric.restValues     - new distance of each constrained pair (vanilla rest values equal the distance at rest)
  fabric.tetherLengths  - straight-line tethers get the new distance; longer (surface) tethers keep their ratio to
                          the straight line
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .cloth import parse_cloth_data
from .render_model import _relative_array


@dataclass
class ReshapedCloth:
    cloth_index: int               # 0-based index into RenderModelClothData.meshes
    render_positions: list
    simulation_positions: list
    rest_values: list
    tether_lengths: list
    max_render_move: float         # largest distance a cloth vertex moved (for reports)


def cloth_vertex_ranges(descriptor: bytes) -> dict:
    """{cloth index (0-based): (first vertex, vertex count)} in the decoded (importer) vertex order.

    Mirrors render_model._parse_render_model_kfc4: vertex buffers are the distinct (byte offset, count) ranges in mesh
    order, and decoded vertices are those buffers back to back."""
    start, count = _relative_array(descriptor, 108, 56)
    buffers, ranges = {}, {}
    base = 0
    for i in range(count):
        o = start + i * 56
        vertex_count = struct.unpack_from("<I", descriptor, o + 20)[0]
        byte_offset = struct.unpack_from("<I", descriptor, o + 40)[0]
        stride = descriptor[o + 53]
        cloth_index = descriptor[o + 54]
        key = (byte_offset, vertex_count, stride)
        if key not in buffers:
            buffers[key] = base
            base += vertex_count
        if cloth_index:
            first = buffers[key]
            previous = ranges.get(cloth_index - 1)
            if previous is not None and previous != (first, vertex_count):
                raise ValueError(f"cloth mesh {cloth_index - 1} uses two different vertex ranges")
            ranges[cloth_index - 1] = (first, vertex_count)
    return ranges


def reshape_cloth(descriptor: bytes, positions) -> list:
    """Return ReshapedCloth per cloth mesh for the moved model `positions` (all decoded vertices, source space).
    Returns [] for a model without cloth."""
    cloth = parse_cloth_data(descriptor)
    if cloth is None:
        return []
    ranges = cloth_vertex_ranges(descriptor)
    result = []
    for k, mesh in enumerate(cloth.meshes):
        if k not in ranges:
            raise ValueError(f"cloth mesh {k} has no RenderModel mesh pointing at it")
        first, count = ranges[k]
        old_render = mesh.arrays["renderPositions"]
        if count != len(old_render):
            raise ValueError(f"cloth mesh {k}: {len(old_render)} cloth vertices but the mesh range has {count}")
        if first + count > len(positions):
            raise ValueError("cloth vertex range is outside the exported vertices")
        render = [tuple(float(c) for c in positions[first + i]) for i in range(count)]
        mapping = mesh.arrays["renderSimulationVertexIndices"]
        old_sim = mesh.arrays["simulationPositions"]
        sums = [[0.0, 0.0, 0.0, 0] for _ in old_sim]
        for i, particle in enumerate(mapping):
            acc = sums[particle]
            for axis in range(3):
                acc[axis] += render[i][axis]
            acc[3] += 1
        simulation = [tuple(acc[a] / acc[3] for a in range(3)) if acc[3] else tuple(old_sim[p])
                      for p, acc in enumerate(sums)]

        fabric = mesh.fabric
        rest = [math.dist(simulation[fabric.indices[2 * c]], simulation[fabric.indices[2 * c + 1]])
                for c in range(len(fabric.restValues))]
        tethers = []
        particles = fabric.numParticles
        for t, (anchor, old_length) in enumerate(zip(fabric.anchors, fabric.tetherLengths)):
            particle = t % particles
            old_straight = math.dist(old_sim[particle], old_sim[anchor])
            new_straight = math.dist(simulation[particle], simulation[anchor])
            if old_straight < 1e-6:          # particle is its own anchor (a pin) or sat on it
                tethers.append(new_straight if old_length < 1e-6 else old_length)
            elif abs(old_length - old_straight) <= 1e-4 * max(1.0, old_straight):
                tethers.append(new_straight)
            else:
                tethers.append(old_length * new_straight / old_straight)
        moved = max((math.dist(a, b) for a, b in zip(render, old_render)), default=0.0)
        result.append(ReshapedCloth(k, render, simulation, rest, tethers, moved))
    return result
