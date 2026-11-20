"""Cloth for Full Topology exports: rebuild a model's cloth parts from edited geometry.

Each exported cloth part becomes its own RenderModel mesh with its own vertex range (as in vanilla), and its cloth
mesh resource is rebuilt from that range:
  renderPositions / renderUvs / renderTriangles  = the part's exported vertices, game UVs and local triangles
  particles, fabric, tethers                     = cloth_cook.cook() on those (UV seams weld into one particle)
  inverse masses / tether style / solver settings = the vanilla cloth part's (settings are left untouched)
  simulationVertexWeights                        = per particle, mean of the exported vertex weights (imported from
                                                   vanilla); without any: 0 for pins, 1 otherwise (as vanilla capes)
  simulationSkinningData                         = per particle, the skin record of its first render vertex, with
                                                   palette slots turned into skeleton joints (vanilla does the same:
                                                   vertex records index model.skinningMap, particle records index the
                                                   skeleton directly; zero-weight slots are 0)
  simulationBlendshapeDeltas                     = empty (Full Topology drops the model's blend shapes)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cloth import free_inverse_mass, tether_mode
from .cloth_cook import cook


@dataclass
class ClothPartInput:
    cloth_index: int                  # vanilla cloth mesh index (0-based) this part replaces
    material: int                     # RenderModel material slot
    positions: list                   # per exported vertex, source space (x, y, z)
    uvs: list                         # per exported vertex, game UV (u, v)
    triangles: list                   # local vertex index triples
    pinned: list                      # per exported vertex: True = attached (pin)
    weights: list = field(default_factory=list)   # per exported vertex simulation weight; empty = 1.0 (pins 0)


@dataclass
class ClothPartData:
    cloth_index: int
    material: int
    cooked: object                    # cloth_cook.CookedCloth
    render_positions: list
    render_uvs: list
    render_triangles: list
    simulation_weights: list
    simulation_skinning: list         # per particle (j0, j1, j2, j3, w0, w1, w2, w3), or [] for static models


def particle_skinning(cooked, skin_records, skinning_map):
    """Per particle skin record from the per-render-vertex 8-byte records (palette slots + weights)."""
    first = {}
    for render_index, particle in enumerate(cooked.render_to_particle):
        first.setdefault(particle, render_index)
    out = []
    for particle in range(len(cooked.positions)):
        record = skin_records[first[particle]]
        slots, weights = record[:4], record[4:8]
        joints = [skinning_map[s] if w else 0 for s, w in zip(slots, weights)]
        out.append(tuple(joints) + tuple(weights))
    return out


def build_cloth_part(part: ClothPartInput, vanilla_mesh, skin_records=None, skinning_map=()) -> ClothPartData:
    """Cook one exported cloth part with the vanilla part's mass and tether style.
    skin_records: per exported vertex 8-byte skin record (skin_export layout) for skinned models, else None."""
    if not part.triangles:
        raise ValueError(f"cloth part {part.cloth_index} has no triangles")
    pinned = [i for i, flag in enumerate(part.pinned) if flag]
    if not pinned:
        raise ValueError(f"cloth part {part.cloth_index} has no pinned vertices: assign the vertices that stay "
                         f"attached to the 'enshrouded_cloth_{part.cloth_index}_pin' vertex group")
    cooked = cook(part.positions, part.triangles, pinned, free_inverse_mass=free_inverse_mass(vanilla_mesh),
                  tethers=tether_mode(vanilla_mesh))
    if len(cooked.positions) > 0xFFFF or len(part.positions) > 0xFFFF:
        raise ValueError(f"cloth part {part.cloth_index} is too large (more than 65535 vertices)")

    sums = [0.0] * len(cooked.positions)
    counts = [0] * len(cooked.positions)
    for render_index, particle in enumerate(cooked.render_to_particle):
        sums[particle] += part.weights[render_index] if part.weights else 1.0
        counts[particle] += 1
    if part.weights:   # imported per-vertex values (vanilla pins can carry a small non-zero weight)
        weights = [max(0.0, sums[p] / counts[p]) for p in range(len(sums))]
    else:
        weights = [0.0 if cooked.inv_masses[p] == 0 else 1.0 for p in range(len(sums))]

    skinning = particle_skinning(cooked, skin_records, skinning_map) if skin_records is not None else []
    return ClothPartData(
        cloth_index=part.cloth_index, material=part.material, cooked=cooked,
        render_positions=[tuple(p) for p in part.positions], render_uvs=[tuple(uv) for uv in part.uvs],
        render_triangles=[tuple(t) for t in part.triangles], simulation_weights=weights,
        simulation_skinning=skinning,
    )
