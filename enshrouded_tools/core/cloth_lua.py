"""Lua (ShroudForge) statements that write cooked cloth data into a keen::RenderModel's cloth block.

Field names follow ShroudForge's types.lua (keen::RenderModelClothData / RenderClothMesh / RenderClothMeshResource).
Index values stay 0-based (they are data); only Lua's table positions are 1-based.
"""
from __future__ import annotations

from .cloth import PARAMETER_NAMES


def _number(value):
    if isinstance(value, int):
        return str(value)
    text = format(value, ".9g")
    return text if text not in ("-0", "nan", "inf", "-inf") else "0"


def _flat(values, per_line=24):
    items = [_number(v) for v in values]
    lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return "{" + (",\n  ".join(lines)) + "}"


def fabric_lua(cooked, target):
    """Statements replacing the fabric (constraints + tethers) and pins of cloth mesh table `target`
    (e.g. `model.cloth.meshes[1]`). Particle order, positions and triangles must already match."""
    return f"""do
 local resource = {target}.data
 local fabric = resource.fabric
 fabric.numParticles = {len(cooked.positions)}
 fabric.phaseIndices = {_flat(cooked.phase_indices)}
 fabric.sets = {_flat(cooked.sets)}
 fabric.restValues = {_flat(cooked.rest_values)}
 fabric.stiffnessValues = {{}}
 fabric.indices = {_flat(cooked.indices)}
 fabric.anchors = {_flat(cooked.anchors)}
 fabric.tetherLengths = {_flat(cooked.tether_lengths)}
 fabric.triangles = {_flat([i for t in cooked.triangles for i in t])}
 resource.simulationInvMasses = {_flat(cooked.inv_masses)}
 resource.skinnedVertexIndices = {_flat(cooked.skinned_vertex_indices)}
end"""


def parameters_lua(parameters, target):
    """Statements setting the solver parameters (dict, keys from cloth.PARAMETER_NAMES) of cloth mesh `target`."""
    lines = [f" {target}.{name} = {_number(float(parameters[name]))}" for name in PARAMETER_NAMES if name in parameters]
    return "do\n" + "\n".join(lines) + "\nend"
