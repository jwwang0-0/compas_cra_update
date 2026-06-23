import numpy as np
import pytest
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Translation
from compas_assembly.datastructures import Block
from scipy.sparse import csr_matrix

from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import rbe_robust_sample
from compas_cra.equilibrium import rbe_robust_support_dual
from compas_cra.equilibrium import rbe_robust_support_primal
from compas_cra.equilibrium import rbe_uncertainty_disturb
from compas_cra.equilibrium import rbe_uncertainty_disturb_sample
from compas_cra.equilibrium import rbe_uncertainty_disturb_support
from compas_cra.equilibrium import rbe_uncertainty_disturb_support_dual
from compas_cra.equilibrium import rbe_uncertainty_disturb_support_primal
from compas_cra.equilibrium.rbe_robust import _prepare_problem
from compas_cra.equilibrium.rbe_uncertainty_disturb import _uncertainty_shifts


def two_block_assembly():
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
    return assembly


def xyz(point):
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return np.asarray([point.x, point.y, point.z], dtype=float)
    return np.asarray(point, dtype=float)


def point_from_offset(assembly, node, offset):
    block = assembly.graph.node_attribute(node, "block")
    return (xyz(block.center()) + np.asarray(offset, dtype=float)).tolist()


def four_grasp_points(assembly, node):
    return [
        point_from_offset(assembly, node, [0.5, -0.5, 0.5]),
        point_from_offset(assembly, node, [0.5, 0.5, 0.5]),
        point_from_offset(assembly, node, [0.5, -0.5, -0.5]),
        point_from_offset(assembly, node, [0.5, 0.5, -0.5]),
    ]


def test_zero_uncertainty_matches_existing_robust_solvers():
    assembly = two_block_assembly()
    load_dofs = [(1, "fx"), (1, "fy")]
    robust_sample = rbe_robust_sample(assembly, load_dofs, density=1, num_directions=12)
    robust_primal = rbe_robust_support_primal(assembly, load_dofs, density=1, num_directions=12)
    robust_dual = rbe_robust_support_dual(assembly, load_dofs, density=1, num_directions=12)

    uncertain_sample = rbe_uncertainty_disturb_sample(assembly, load_dofs, density=1, num_directions=12)
    uncertain_primal = rbe_uncertainty_disturb_support_primal(assembly, load_dofs, density=1, num_directions=12)
    uncertain_dual = rbe_uncertainty_disturb_support_dual(assembly, load_dofs, density=1, num_directions=12)

    assert uncertain_sample.radial_limits == pytest.approx(robust_sample.radial_limits)
    assert uncertain_primal.support_values == pytest.approx(robust_primal.support_values)
    assert uncertain_dual.support_values == pytest.approx(robust_dual.support_values)
    assert np.asarray(uncertain_dual.outer_polygon) == pytest.approx(np.asarray(robust_dual.outer_polygon))


def test_uncertainty_shifts_from_dofs_match_direct_basis():
    assembly = two_block_assembly()
    problem = _prepare_problem(assembly, [(1, "fx"), (1, "fy")], 0.84, 1.0, None)
    vertices = np.asarray([[-0.2, 0.3], [0.4, -0.5]])
    basis = csr_matrix((np.ones(2), ([0, 2], [0, 1])), shape=(problem.equilibrium.shape[0], 2))

    dof_shifts = _uncertainty_shifts(
        problem,
        assembly=assembly,
        uncertainty_load_dofs=[(1, "fx"), (1, "fz")],
        uncertainty_vertices=vertices,
    )
    basis_shifts = _uncertainty_shifts(
        problem,
        uncertainty_vertices=vertices,
        uncertainty_basis=basis,
    )

    assert np.asarray(dof_shifts) == pytest.approx(np.asarray(basis_shifts))


def test_uncertain_support_matches_axis_intersection_of_vertex_sets():
    assembly = two_block_assembly()
    vertices = np.asarray([[-0.1], [0.2]])
    uncertain = rbe_uncertainty_disturb_support_dual(
        assembly,
        [(1, "fx"), (1, "fy")],
        density=1,
        uncertainty_vertices=vertices,
        uncertainty_load_dofs=[(1, "fx")],
        num_directions=4,
    )
    vertex_results = [
        rbe_robust_support_dual(
            assembly,
            [(1, "fx"), (1, "fy")],
            density=1,
            external_forces={1: [float(vertex[0]), 0, 0, 0, 0, 0]},
            num_directions=4,
        )
        for vertex in vertices
    ]
    expected = np.min(np.asarray([result.support_values for result in vertex_results]), axis=0)

    assert uncertain.support_values == pytest.approx(expected)


def test_uncertain_primal_and_dual_support_formulations_match():
    assembly = two_block_assembly()
    kwargs = {
        "density": 1,
        "uncertainty_vertices": [[-0.1], [0.2]],
        "uncertainty_load_dofs": [(1, "fx")],
        "num_directions": 12,
    }
    primal = rbe_uncertainty_disturb_support_primal(assembly, [(1, "fx"), (1, "fy")], **kwargs)
    dual = rbe_uncertainty_disturb_support_dual(assembly, [(1, "fx"), (1, "fy")], **kwargs)

    assert primal.support_values == pytest.approx(dual.support_values)
    assert np.asarray(primal.halfspaces) == pytest.approx(np.asarray(dual.halfspaces))
    assert rbe_uncertainty_disturb(assembly, [(1, "fx"), (1, "fy")], **kwargs).support_formulation == "dual"
    assert rbe_uncertainty_disturb_support(assembly, [(1, "fx"), (1, "fy")], **kwargs).support_formulation == "dual"


def test_uncertain_radial_points_satisfy_dual_halfspaces():
    assembly = two_block_assembly()
    kwargs = {
        "density": 1,
        "uncertainty_vertices": [[-0.1], [0.2]],
        "uncertainty_load_dofs": [(1, "fx")],
        "num_directions": 12,
    }
    sampled = rbe_uncertainty_disturb_sample(assembly, [(1, "fx"), (1, "fy")], **kwargs)
    dual = rbe_uncertainty_disturb_support_dual(assembly, [(1, "fx"), (1, "fy")], **kwargs)

    for point in sampled.boundary_points:
        for direction_0, direction_1, support in dual.halfspaces:
            assert direction_0 * point[0] + direction_1 * point[1] <= support + 1e-7


@pytest.mark.parametrize(
    "uncertainty_vertices, uncertainty_load_dofs, uncertainty_basis, message",
    [
        ([1.0, 2.0], [(1, "fx")], None, "2D array"),
        ([[1.0, 2.0]], [(1, "fx")], None, "dimension"),
        ([[1.0]], None, None, "Provide uncertainty_basis or uncertainty_load_dofs"),
        ([[1.0]], [(0, "fx")], None, "is a support"),
        ([[1.0]], [(1, "bad")], None, "Unknown uncertainty load component"),
        ([[1.0]], None, csr_matrix((5, 1)), "row count"),
    ],
)
def test_invalid_uncertainty_inputs_raise(
    uncertainty_vertices,
    uncertainty_load_dofs,
    uncertainty_basis,
    message,
):
    with pytest.raises(ValueError, match=message):
        rbe_uncertainty_disturb_support_dual(
            two_block_assembly(),
            [(1, "fx"), (1, "fy")],
            density=1,
            uncertainty_vertices=uncertainty_vertices,
            uncertainty_load_dofs=uncertainty_load_dofs,
            uncertainty_basis=uncertainty_basis,
            num_directions=4,
        )


def test_uncertain_empty_safe_load_set_raises():
    with pytest.raises(ValueError, match="safe load set is empty"):
        rbe_uncertainty_disturb_support_primal(
            two_block_assembly(),
            [(1, "fx"), (1, "fy")],
            density=1,
            uncertainty_vertices=[[2.0]],
            uncertainty_load_dofs=[(1, "fz")],
            num_directions=4,
        )


def test_four_point_hidden_application_works_with_uncertainty():
    assembly = two_block_assembly()
    kwargs = {
        "density": 1,
        "load_application_points": {1: four_grasp_points(assembly, 1)},
        "application_force_bound": 10.0,
        "uncertainty_vertices": [[-0.1], [0.1]],
        "uncertainty_load_dofs": [(1, "fx")],
        "num_directions": 8,
    }
    primal = rbe_uncertainty_disturb_support_primal(assembly, [(1, "fx"), (1, "fz")], **kwargs)
    dual = rbe_uncertainty_disturb_support_dual(assembly, [(1, "fx"), (1, "fz")], **kwargs)

    assert primal.is_bounded
    assert dual.is_bounded
    assert primal.support_values == pytest.approx(dual.support_values)
