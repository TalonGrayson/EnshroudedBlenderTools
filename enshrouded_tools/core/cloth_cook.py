"""Build ("cook") Enshrouded cloth simulation data for a triangle mesh.

The game stores NvCloth-style fabrics. What vanilla models show (banner_dlc_01, see collab/dev_tools/cloth_analyse.py):
  * particles = render vertices welded by position (UV seams split render vertices, not particles);
    renderSimulationVertexIndices maps each render vertex to its particle.
  * pinned particles have inverse mass 0 and are listed in skinnedVertexIndices; free particles share one inverse
    mass (banners 1.0, capes 0.5).
  * distance constraints, rest length = straight-line distance at rest, never between two pinned particles:
      stretch  - mesh edges that aren't quad diagonals
      bending  - particle pairs two mesh edges apart along a (nearly) straight line
      shearing - both diagonals of each quad (two triangles joined across the longest edge of at least one of them)
    These defaults reproduce 95% of banner_dlc_01's constraint pairs (98% of the cooked ones are vanilla pairs) and
    the same number of phases (24); collab/dev_tools/test_cloth_cook.py checks it.
  * each constraint group is split into sets that share no particle; every set is one phase
    (phaseIndices = 0..n-1, sets = cumulative constraint ends).
  * one tether per particle: anchor = nearest pinned particle, measured along mesh edges on capes (matches every
    vanilla anchor tested) or in a straight line on the banner_dlc_* placeables; length = that distance. Vanilla capes have ~15% of
    tethers longer than any path over the mesh, so cooked tethers can be a little tighter there.
  * fabric.triangles = simulationTriangles flattened.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field

# Two mesh edges meeting at a particle count as one straight line for bending when the angle between them is at
# least this far from a fold: cos(angle a-v-b) <= BEND_STRAIGHTNESS.
BEND_STRAIGHTNESS = -0.9


@dataclass
class CookedCloth:
    positions: list                  # particle positions (x, y, z)
    inv_masses: list                 # 0.0 pinned, 1.0 free
    triangles: list                  # particle triangles (a, b, c)
    render_to_particle: list         # per render vertex -> particle index
    phase_indices: list
    sets: list
    rest_values: list
    indices: list                    # flattened constraint particle pairs
    anchors: list
    tether_lengths: list
    skinned_vertex_indices: list     # pinned particles
    groups: dict = field(default_factory=dict)   # group name -> (first set, set count), for reports


def weld(render_positions, tolerance=1e-5):
    """Merge render vertices that share a position. Returns (particle positions, render -> particle)."""
    scale = 1.0 / tolerance
    lookup, positions, mapping = {}, [], []
    for p in render_positions:
        key = tuple(round(c * scale) for c in p)
        index = lookup.get(key)
        if index is None:
            index = lookup[key] = len(positions)
            positions.append(tuple(p))
        mapping.append(index)
    return positions, mapping


def _edge(u, v):
    return (u, v) if u < v else (v, u)


def _quad_diagonals(positions, triangles, rule="both"):
    """Edges shared by two triangles that are the longest edge of both ("both") or of either ("either"): the quad
    diagonal. With "either" each triangle joins at most one quad (its longest edge).
    Returns {diagonal edge: (opposite vertex of triangle 1, opposite vertex of triangle 2)}."""
    edge_triangles = defaultdict(list)
    for t, tri in enumerate(triangles):
        for i in range(3):
            edge_triangles[_edge(tri[i], tri[(i + 1) % 3])].append(t)

    def longest(tri):
        edges = [_edge(tri[i], tri[(i + 1) % 3]) for i in range(3)]
        return max(edges, key=lambda e: math.dist(positions[e[0]], positions[e[1]]))

    diagonals = {}
    for edge, tris in edge_triangles.items():
        if len(tris) != 2:
            continue
        a, b = (triangles[t] for t in tris)
        hits = (longest(a) == edge) + (longest(b) == edge)
        if hits == 2 or (rule == "either" and hits == 1):
            far_a = next(v for v in a if v not in edge)
            far_b = next(v for v in b if v not in edge)
            diagonals[edge] = (far_a, far_b)
    return diagonals, edge_triangles


def _colour(pairs):
    """Greedy split of constraint pairs into sets in which no particle appears twice."""
    sets = []
    for pair in pairs:
        for members, used in sets:
            if pair[0] not in used and pair[1] not in used:
                members.append(pair)
                used.update(pair)
                break
        else:
            sets.append(([pair], set(pair)))
    return [members for members, _ in sets]


def _geodesic_tethers(positions, pairs, pinned):
    """Multi-source Dijkstra from every pinned particle over `pairs` (mesh edges by default: that picks the same
    anchor as vanilla capes for every particle). Length is never shorter than the straight line.
    Particles the cloth doesn't connect to a pin fall back to the straight-line nearest pin."""
    adjacency = defaultdict(list)
    for u, v in pairs:
        length = math.dist(positions[u], positions[v])
        adjacency[u].append((v, length))
        adjacency[v].append((u, length))
    best = {p: (0.0, p) for p in pinned}
    heap = [(0.0, p, p) for p in sorted(pinned)]
    heapq.heapify(heap)
    while heap:
        distance, node, anchor = heapq.heappop(heap)
        if best[node][0] < distance:
            continue
        for other, length in adjacency[node]:
            candidate = distance + length
            if other not in best or candidate < best[other][0] - 1e-12:
                best[other] = (candidate, anchor)
                heapq.heappush(heap, (candidate, other, anchor))
    anchors, lengths = [], []
    pinned_list = sorted(pinned)
    for i, p in enumerate(positions):
        if i in best:
            anchor = best[i][1]
            lengths.append(max(best[i][0], math.dist(p, positions[anchor])))
        else:
            anchor = min(pinned_list, key=lambda q: math.dist(p, positions[q]))
            lengths.append(math.dist(p, positions[anchor]))
        anchors.append(anchor)
    return anchors, lengths


def cook(render_positions, render_triangles, pinned_render_vertices, *, bend_straightness=BEND_STRAIGHTNESS,
         diagonal_rule="either", bend_over="all", free_inverse_mass=1.0, tethers="geodesic", tether_graph="edges"):
    """Cook cloth data.

    render_positions: per render vertex (x, y, z), in model space.
    render_triangles: (a, b, c) render vertex indices.
    pinned_render_vertices: render vertex indices that stay attached (any render copy pins the particle).
    free_inverse_mass: 1 / mass of every free particle (vanilla banners 1.0, vanilla capes 0.5).
    tethers: "geodesic" (anchor = nearest pinned particle measured over the cloth, as vanilla capes do)
             or "euclidean" (straight-line nearest, as all four banner_dlc_* placeables do).
    tether_graph: "edges" (paths along mesh edges) or "constraints" (also across bending/shear pairs: shorter paths).
    """
    positions, mapping = weld(render_positions)
    triangles = [tuple(mapping[i] for i in tri) for tri in render_triangles]
    triangles = [t for t in triangles if len(set(t)) == 3]
    pinned = {mapping[i] for i in pinned_render_vertices}
    inv_masses = [0.0 if i in pinned else free_inverse_mass for i in range(len(positions))]

    diagonals, edge_triangles = _quad_diagonals(positions, triangles, diagonal_rule)
    stretch = [e for e in edge_triangles if e not in diagonals]
    shear = []
    for edge, (far_a, far_b) in diagonals.items():
        shear.append(edge)
        shear.append(_edge(far_a, far_b))

    neighbours = defaultdict(set)
    for u, v in (stretch if bend_over == "stretch" else list(edge_triangles)):
        neighbours[u].add(v)
        neighbours[v].add(u)
    stretch_set = set(stretch)
    bending, seen = [], set()
    for v, around in neighbours.items():
        around = sorted(around)
        for i, a in enumerate(around):
            for b in around[i + 1:]:
                pair = _edge(a, b)
                if pair in seen or pair in stretch_set:
                    continue
                va = [positions[a][k] - positions[v][k] for k in range(3)]
                vb = [positions[b][k] - positions[v][k] for k in range(3)]
                la, lb = math.hypot(*va), math.hypot(*vb)
                if la < 1e-9 or lb < 1e-9:
                    continue
                if sum(x * y for x, y in zip(va, vb)) / (la * lb) <= bend_straightness:
                    seen.add(pair)
                    bending.append(pair)

    shear_seen, shear_unique = set(), []
    for pair in shear:
        if pair not in shear_seen and pair not in stretch_set:
            shear_seen.add(pair)
            shear_unique.append(pair)

    def free(pairs):
        return [p for p in pairs if not (p[0] in pinned and p[1] in pinned)]

    phase_sets, groups = [], {}
    for name, pairs in (("stretch", free(stretch)), ("bending", free(bending)), ("shearing", free(shear_unique))):
        coloured = _colour(pairs)
        groups[name] = (len(phase_sets), len(coloured))
        phase_sets.extend(coloured)

    sets, rest_values, indices, end = [], [], [], 0
    for members in phase_sets:
        for u, v in members:
            indices.extend((u, v))
            rest_values.append(math.dist(positions[u], positions[v]))
        end += len(members)
        sets.append(end)

    anchors, tether_lengths = [], []
    if pinned and tethers == "euclidean":
        pinned_list = sorted(pinned)
        for p in positions:
            anchor = min(pinned_list, key=lambda q: math.dist(p, positions[q]))
            anchors.append(anchor)
            tether_lengths.append(math.dist(p, positions[anchor]))
    elif pinned:
        graph_pairs = set(edge_triangles)
        if tether_graph == "constraints":
            graph_pairs |= set(bending) | set(shear_unique)
        anchors, tether_lengths = _geodesic_tethers(positions, graph_pairs, pinned)

    return CookedCloth(
        positions=positions, inv_masses=inv_masses, triangles=triangles, render_to_particle=mapping,
        phase_indices=list(range(len(sets))), sets=sets, rest_values=rest_values, indices=indices,
        anchors=anchors, tether_lengths=tether_lengths, skinned_vertex_indices=sorted(pinned), groups=groups,
    )
