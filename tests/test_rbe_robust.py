import numpy as np
import pytest
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Translation
from compas_assembly.datastructures import Block
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack

from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import RobustForceResult
from compas_cra.equilibrium import rbe_robust_sample
from compas_cra.equilibrium import rbe_robust_support
from compas_cra.equilibrium.cra_helper import equilibrium_setup
from compas_cra.equilibrium.cra_helper import external_force_setup
from compas_cra.equilibrium.cra_helper import friction_setup
from compas_cra.equilibrium.cra_helper import num_vertices


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


def polygon_area(polygon):
    return 0.5 * sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )


def primal_support(assembly, direction, mu=0.84, density=1.0):
    equilibrium = equilibrium_setup(assembly, penalty=False)
    friction = friction_setup(assembly, mu, penalty=False)
    baseline = external_force_setup(assembly, density).flatten()
    vertex_count = num_vertices(assembly)
    force_count = vertex_count * 3

    normal_nonnegative = csr_matrix(
        (
            -np.ones(vertex_count),
            (np.arange(vertex_count), np.arange(vertex_count) * 3),
        ),
        shape=(vertex_count, force_count),
    )
    inequalities = vstack([friction, normal_nonnegative], format="csr")
    load_basis = csr_matrix((np.ones(2), ([0, 1], [0, 1])), shape=(equilibrium.shape[0], 2))

    objective = np.zeros(force_count + 2)
    objective[-2:] = -np.asarray(direction)
    result = linprog(
        objective,
        A_ub=hstack([inequalities, csr_matrix((inequalities.shape[0], 2))]),
        b_ub=np.zeros(inequalities.shape[0]),
        A_eq=hstack([equilibrium, load_basis]),
        b_eq=-baseline,
        bounds=[(None, None)] * (force_count + 2),
        method="highs",
    )
    assert result.status == 0
    return -result.fun


def test_robust_sample_and_support_bound_horizontal_loads():
    assembly = two_block_assembly()
    interface = assembly.graph.edge_attribute((0, 1), "interfaces")[0]

    sampled = rbe_robust_sample(assembly, [(1, "fx"), (1, "fy")], density=1)
    supported = rbe_robust_support(assembly, [(1, "fx"), (1, "fy")], density=1)

    assert isinstance(sampled, RobustForceResult)
    assert sampled.origin_feasible
    assert supported.origin_feasible
    assert sampled.is_bounded
    assert supported.is_bounded
    assert len(sampled.directions) == 36
    assert len(sampled.radial_limits) == 36
    assert len(supported.support_values) == 36
    assert len(sampled.polygon) >= 3
    assert len(supported.polygon) >= 3
    assert polygon_area(sampled.polygon) > 0
    assert polygon_area(supported.polygon) > 0
    assert interface.forces is None

    for point in sampled.boundary_points:
        for direction_0, direction_1, support in supported.halfspaces:
            assert direction_0 * point[0] + direction_1 * point[1] <= support + 1e-7


def test_dual_support_matches_independent_primal_lp():
    assembly = two_block_assembly()
    result = rbe_robust_support(assembly, [(1, "fx"), (1, "fy")], density=1)

    assert result.support_values[0] == pytest.approx(primal_support(assembly, [1.0, 0.0]))
    assert result.support_values[9] == pytest.approx(primal_support(assembly, [0.0, 1.0]))


@pytest.mark.parametrize(
    "load_dofs",
    [
        [(1, "fx")],
        [(1, "fx"), (1, "fx")],
        [(0, "fx"), (1, "fy")],
        [(1, "invalid"), (1, "fy")],
        [(99, "fx"), (1, "fy")],
    ],
)
def test_invalid_load_dofs_raise(load_dofs):
    with pytest.raises(ValueError):
        rbe_robust_sample(two_block_assembly(), load_dofs, density=1, num_directions=4)


@pytest.mark.parametrize("solver", [rbe_robust_sample, rbe_robust_support])
def test_infeasible_baseline_raises(solver):
    upward_force = {1: [0, 0, 2, 0, 0, 0]}
    with pytest.raises(ValueError, match="baseline load is infeasible"):
        solver(
            two_block_assembly(),
            [(1, "fx"), (1, "fy")],
            density=1,
            external_forces=upward_force,
            num_directions=4,
        )


def test_known_baseline_force_translates_increment_range():
    assembly = two_block_assembly()
    reference = rbe_robust_support(assembly, [(1, "fx"), (1, "fy")], density=1)
    shifted = rbe_robust_support(
        assembly,
        [(1, "fx"), (1, "fy")],
        density=1,
        external_forces={1: [0.1, 0, 0, 0, 0, 0]},
    )

    assert shifted.support_values[0] == pytest.approx(reference.support_values[0] - 0.1)
    assert shifted.support_values[18] == pytest.approx(reference.support_values[18] + 0.1)


@pytest.mark.parametrize("solver", [rbe_robust_sample, rbe_robust_support])
def test_downward_vertical_load_is_unbounded(solver):
    result = solver(two_block_assembly(), [(1, "fz"), (1, "fx")], density=1, num_directions=12)

    assert not result.is_bounded
    assert "unbounded" in result.statuses
    assert result.polygon == []
