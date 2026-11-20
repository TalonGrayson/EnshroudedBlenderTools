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


def _float3s(values):
    return "{" + ",\n  ".join(
        f"{{x = {_number(v[0])}, y = {_number(v[1])}, z = {_number(v[2])}}}" for v in values) + "}"


def reshape_lua(reshaped, mod_id, model_var="model"):
    """Statements writing cloth_reshape.ReshapedCloth results (moved positions, new lengths) into `model_var`."""
    blocks = [f'''if {model_var}.cloth == nil or #{model_var}.cloth.meshes < {max(r.cloth_index for r in reshaped) + 1} then
    error("[{mod_id}] target RenderModel has no matching cloth block")
end''']
    for r in reshaped:
        target = f"{model_var}.cloth.meshes[{r.cloth_index + 1}].data"
        blocks.append(f"""do
 local resource = {target}
 if #resource.renderPositions ~= {len(r.render_positions)} or resource.fabric.numParticles ~= {len(r.simulation_positions)} then
  error("[{mod_id}] cloth mesh {r.cloth_index} layout differs from the exported model")
 end
 resource.renderPositions = {_float3s(r.render_positions)}
 resource.simulationPositions = {_float3s(r.simulation_positions)}
 resource.fabric.restValues = {_flat(r.rest_values)}
 resource.fabric.tetherLengths = {_flat(r.tether_lengths)}
end""")
    return "\n".join(blocks)


def _uint3s(values):
    return "{" + ",\n  ".join(f"{{x = {a}, y = {b}, z = {c}}}" for a, b, c in values) + "}"


def _float2s(values):
    return "{" + ",\n  ".join(f"{{x = {_number(u)}, y = {_number(v)}}}" for u, v in values) + "}"


def _skin_records(values):
    return "{" + ",\n  ".join(
        f"{{jointIndices = {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}, jointWeights = {{{r[4]}, {r[5]}, {r[6]}, {r[7]}}}}}"
        for r in values) + "}"


def full_cloth_lua(parts, mod_id, model_var="model"):
    """Statements rebuilding `model_var`.cloth from cloth_export.ClothPartData parts, in export order.
    Exported part i becomes cloth mesh i + 1 (the RenderModel mesh writing it uses clothMeshIndex i + 1).
    Solver settings, hasSkinning and colliders of the reused vanilla cloth meshes stay as they are."""
    needed = max(p.cloth_index for p in parts) + 1
    lines = [f'''if {model_var}.cloth == nil or #{model_var}.cloth.meshes < {needed} then
    error("[{mod_id}] target RenderModel has no matching cloth block")
end
local vanilla_cloth_meshes = {{}}
for i, cloth_mesh in ipairs({model_var}.cloth.meshes) do vanilla_cloth_meshes[i] = cloth_mesh end
local exported_cloth_meshes = {{}}''']
    for p in parts:
        c = p.cooked
        lines.append(f"""do
 local cloth_mesh = vanilla_cloth_meshes[{p.cloth_index + 1}]
 local resource = cloth_mesh.data
 local fabric = resource.fabric
 fabric.numParticles = {len(c.positions)}
 fabric.phaseIndices = {_flat(c.phase_indices)}
 fabric.sets = {_flat(c.sets)}
 fabric.restValues = {_flat(c.rest_values)}
 fabric.stiffnessValues = {{}}
 fabric.indices = {_flat(c.indices)}
 fabric.anchors = {_flat(c.anchors)}
 fabric.tetherLengths = {_flat(c.tether_lengths)}
 fabric.triangles = {_flat([i for t in c.triangles for i in t])}
 resource.simulationPositions = {_float3s(c.positions)}
 resource.simulationInvMasses = {_flat(c.inv_masses)}
 resource.simulationStiffnesses = {{}}
 resource.simulationVertexWeights = {_flat(p.simulation_weights)}
 resource.simulationTriangles = {_uint3s(c.triangles)}
 resource.simulationBlendshapeDeltas = {{}}
 resource.simulationSkinningData = {_skin_records(p.simulation_skinning)}
 resource.skinnedVertexIndices = {_flat(c.skinned_vertex_indices)}
 resource.renderPositions = {_float3s(p.render_positions)}
 resource.renderUvs = {_float2s(p.render_uvs)}
 resource.renderTriangles = {_uint3s(p.render_triangles)}
 resource.renderSimulationVertexIndices = {_flat(c.render_to_particle)}
 table.insert(exported_cloth_meshes, cloth_mesh)
end""")
    in_place = [p.cloth_index for p in parts] == list(range(needed))
    lines.append(f"""{'-- parts are the vanilla cloth meshes in order: keep the array' if in_place else ''}
{'if #' + model_var + '.cloth.meshes ~= ' + str(len(parts)) + ' then ' + model_var + '.cloth.meshes = exported_cloth_meshes end' if in_place else model_var + '.cloth.meshes = exported_cloth_meshes'}
{model_var}.cloth.totalSimulationTriangleCount = {sum(len(p.cooked.triangles) for p in parts)}
{model_var}.cloth.totalRenderVertexCount = {sum(len(p.render_positions) for p in parts)}""")
    return "\n".join(lines)


def parameters_lua(parameters, target):
    """Statements setting the solver parameters (dict, keys from cloth.PARAMETER_NAMES) of cloth mesh `target`."""
    lines = [f" {target}.{name} = {_number(float(parameters[name]))}" for name in PARAMETER_NAMES if name in parameters]
    return "do\n" + "\n".join(lines) + "\nend"
