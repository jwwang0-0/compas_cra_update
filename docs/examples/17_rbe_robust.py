"""Compute inner and outer approximations of a two-dimensional safe load range."""

from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Translation
from compas_assembly.datastructures import Block

from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import rbe_robust_sample
from compas_cra.equilibrium import rbe_robust_support

support = Box(1, 1, 1)
free = Box(1, 1, 1, frame=Frame.worldXY().transformed(Translation.from_vector([0, 0, 1])))

assembly = CRA_Assembly()
assembly.add_block(Block.from_shape(support), node=0)
assembly.add_block(Block.from_shape(free), node=1)
assembly.set_boundary_conditions([0])

interface = Mesh()
corners = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]]
for index, xyz in enumerate(corners):
    interface.add_vertex(key=index, x=xyz[0], y=xyz[1], z=xyz[2])
interface.add_face([0, 1, 2, 3])
assembly.add_interfaces_from_meshes([interface], 0, 1)

load_dofs = [(1, "fx"), (1, "fy")]
inner = rbe_robust_sample(assembly, load_dofs)
outer = rbe_robust_support(assembly, load_dofs)

print(inner.polygon)
print(outer.polygon)
