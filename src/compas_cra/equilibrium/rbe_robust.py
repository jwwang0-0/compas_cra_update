"""Robust rigid-block equilibrium force-range analysis."""

import math
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from compas_assembly.datastructures import Assembly
from scipy.optimize import linprog
from scipy.sparse import bmat
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError

from .cra_helper import equilibrium_setup
from .cra_helper import external_force_setup
from .cra_helper import free_nodes
from .cra_helper import friction_setup
from .cra_helper import num_vertices

_COMPONENT_INDEX = {
    "fx": 0,
    "fy": 1,
    "fz": 2,
    "mx": 3,
    "my": 4,
    "mz": 5,
}


@dataclass
class RobustForceResult:
    """Result of a two-dimensional robust RBE force-range analysis.

    Attributes
    ----------
    method : str
        ``"sample"`` for radial sampling or ``"support"`` for the dual
        support-function method.
    load_dofs : tuple
        The two normalized ``(node, component)`` load degrees of freedom.
    directions : list[list[float]]
        Unit directions used by the analysis.
    statuses : list[str]
        Solver status for each direction. Values are ``"optimal"`` or
        ``"unbounded"``.
    origin_feasible : bool
        Whether the baseline load, corresponding to zero load increment, is
        feasible.
    is_bounded : bool
        Whether the complete two-dimensional feasible load set is bounded.
    radial_limits : list[float | None]
        Maximum radial load factors. Populated by radial sampling.
    boundary_points : list[list[float]]
        Finite radial boundary points. Populated by radial sampling.
    support_values : list[float | None]
        Support-function values. Populated by the dual method.
    halfspaces : list[list[float]]
        Finite supporting halfspaces stored as ``[d0, d1, h]``, representing
        ``d0 * b0 + d1 * b1 <= h``.
    polygon : list[list[float]]
        Counterclockwise inner polygon for sampling or outer polygon for the
        support method. Empty when the approximation is not bounded and
        full-dimensional.
    """

    method: str
    load_dofs: Tuple[Tuple[Any, str], Tuple[Any, str]]
    directions: List[List[float]]
    statuses: List[str]
    origin_feasible: bool
    is_bounded: bool
    radial_limits: List[Optional[float]] = field(default_factory=list)
    boundary_points: List[List[float]] = field(default_factory=list)
    support_values: List[Optional[float]] = field(default_factory=list)
    halfspaces: List[List[float]] = field(default_factory=list)
    polygon: List[List[float]] = field(default_factory=list)


@dataclass
class _RobustProblem:
    equilibrium: csr_matrix
    inequalities: csr_matrix
    baseline_load: np.ndarray
    load_basis: csr_matrix
    load_dofs: Tuple[Tuple[Any, str], Tuple[Any, str]]


def rbe_robust_sample(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
) -> RobustForceResult:
    """Approximate a two-dimensional safe load range by radial sampling.

    The two selected load components are increments about the baseline load
    defined by gravity and ``external_forces``. Contact forces use the
    compression-only RBE formulation ``[fn, fu, fv]``.

    Parameters
    ----------
    assembly : :class:`~compas_assembly.datastructures.Assembly`
        Rigid-block assembly.
    load_dofs : sequence[tuple[node, str]]
        Exactly two unique load degrees of freedom. Components are ``"fx"``,
        ``"fy"``, ``"fz"``, ``"mx"``, ``"my"``, or ``"mz"``.
    mu : float, optional
        Friction coefficient.
    density : float, optional
        Default block density.
    external_forces : dict, optional
        Known baseline wrenches keyed by assembly node.
    num_directions : int, optional
        Number of uniformly spaced radial directions.
    tolerance : float, optional
        Geometric tolerance used for polygon construction.
    solver_options : dict, optional
        Options passed to SciPy HiGHS.
    verbose : bool, optional
        Print a result summary.
    timer : bool, optional
        Print total analysis time.

    Returns
    -------
    :class:`RobustForceResult`
        Radial limits, finite boundary points, and an inner polygon.

    Raises
    ------
    ValueError
        If the input is invalid or the baseline load is infeasible.
    RuntimeError
        If HiGHS reports an unexpected numerical or solver failure.
    """
    start_time = time.time()
    problem = _prepare_problem(assembly, load_dofs, mu, density, external_forces)
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    _require_feasible_origin(problem, options)

    statuses = []
    radial_limits = []
    boundary_points = []
    for direction in directions:
        status, limit = _solve_radial(problem, direction, options)
        statuses.append(status)
        radial_limits.append(limit)
        if limit is not None:
            boundary_points.append((limit * direction).tolist())

    is_bounded = _check_global_boundedness(problem, options)
    polygon = _inner_polygon(boundary_points, tolerance) if is_bounded else []
    result = RobustForceResult(
        method="sample",
        load_dofs=problem.load_dofs,
        directions=[direction.tolist() for direction in directions],
        statuses=statuses,
        origin_feasible=True,
        is_bounded=is_bounded,
        radial_limits=radial_limits,
        boundary_points=boundary_points,
        polygon=polygon,
    )
    _report(result, start_time, verbose, timer)
    return result


def rbe_robust_support(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
) -> RobustForceResult:
    """Approximate a two-dimensional safe load range with dual supports.

    Each solve produces a supporting halfspace of the feasible load set. The
    intersection of the finite halfspaces is an outer approximation.

    Parameters
    ----------
    assembly : :class:`~compas_assembly.datastructures.Assembly`
        Rigid-block assembly.
    load_dofs : sequence[tuple[node, str]]
        Exactly two unique load degrees of freedom. Components are ``"fx"``,
        ``"fy"``, ``"fz"``, ``"mx"``, ``"my"``, or ``"mz"``.
    mu : float, optional
        Friction coefficient.
    density : float, optional
        Default block density.
    external_forces : dict, optional
        Known baseline wrenches keyed by assembly node.
    num_directions : int, optional
        Number of uniformly spaced support directions.
    tolerance : float, optional
        Geometric tolerance used for polygon construction.
    solver_options : dict, optional
        Options passed to SciPy HiGHS.
    verbose : bool, optional
        Print a result summary.
    timer : bool, optional
        Print total analysis time.

    Returns
    -------
    :class:`RobustForceResult`
        Support values, supporting halfspaces, and an outer polygon.

    Raises
    ------
    ValueError
        If the input is invalid or the baseline load is infeasible.
    RuntimeError
        If HiGHS reports an unexpected numerical or solver failure.
    """
    start_time = time.time()
    problem = _prepare_problem(assembly, load_dofs, mu, density, external_forces)
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    _require_feasible_origin(problem, options)

    statuses = []
    support_values = []
    halfspaces = []
    for direction in directions:
        status, support = _solve_support(problem, direction, options)
        statuses.append(status)
        support_values.append(support)
        if support is not None:
            halfspaces.append([float(direction[0]), float(direction[1]), support])

    is_bounded = _check_global_boundedness(problem, options)
    polygon = _outer_polygon(halfspaces, tolerance) if is_bounded else []
    result = RobustForceResult(
        method="support",
        load_dofs=problem.load_dofs,
        directions=[direction.tolist() for direction in directions],
        statuses=statuses,
        origin_feasible=True,
        is_bounded=is_bounded,
        support_values=support_values,
        halfspaces=halfspaces,
        polygon=polygon,
    )
    _report(result, start_time, verbose, timer)
    return result


def _prepare_problem(assembly, load_dofs, mu, density, external_forces):
    normalized_dofs = _validate_load_dofs(assembly, load_dofs)
    equilibrium = equilibrium_setup(assembly, penalty=False).tocsr()
    friction = friction_setup(assembly, mu, penalty=False).tocsr()
    baseline_load = external_force_setup(assembly, density, external_forces).flatten()

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
    load_basis = _load_basis(assembly, normalized_dofs, equilibrium.shape[0])

    return _RobustProblem(
        equilibrium=equilibrium,
        inequalities=inequalities,
        baseline_load=baseline_load,
        load_basis=load_basis,
        load_dofs=normalized_dofs,
    )


def _validate_load_dofs(assembly, load_dofs):
    if len(load_dofs) != 2:
        raise ValueError("load_dofs must contain exactly two (node, component) pairs.")

    node_keys = list(assembly.graph.nodes())
    normalized = []
    for load_dof in load_dofs:
        if not isinstance(load_dof, (list, tuple)) or len(load_dof) != 2:
            raise ValueError("Each load degree of freedom must be a (node, component) pair.")
        node, component = load_dof
        if node not in node_keys:
            raise ValueError("Unknown assembly node: {!r}.".format(node))
        if assembly.graph.node_attribute(node, "is_support"):
            raise ValueError("Load degree of freedom node {!r} is a support.".format(node))
        if not isinstance(component, str) or component.lower() not in _COMPONENT_INDEX:
            raise ValueError("Unknown load component: {!r}.".format(component))
        normalized.append((node, component.lower()))

    if normalized[0] == normalized[1]:
        raise ValueError("load_dofs must contain two unique degrees of freedom.")
    return normalized[0], normalized[1]


def _load_basis(assembly, load_dofs, row_count):
    node_keys = list(assembly.graph.nodes())
    node_index = {node: index for index, node in enumerate(node_keys)}
    free = free_nodes(assembly)
    rows = []
    columns = []
    for column, (node, component) in enumerate(load_dofs):
        free_position = free.index(node_index[node])
        rows.append(free_position * 6 + _COMPONENT_INDEX[component])
        columns.append(column)
    return csr_matrix((np.ones(2), (rows, columns)), shape=(row_count, 2))


def _directions(num_directions):
    if isinstance(num_directions, bool) or not isinstance(num_directions, int) or num_directions < 4:
        raise ValueError("num_directions must be an integer greater than or equal to 4.")
    angles = np.arange(num_directions, dtype=float) * (2.0 * math.pi / num_directions)
    return [np.array([math.cos(angle), math.sin(angle)]) for angle in angles]


def _validate_tolerance(tolerance):
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")


def _require_feasible_origin(problem, options):
    force_count = problem.equilibrium.shape[1]
    result = linprog(
        np.zeros(force_count),
        A_ub=problem.inequalities,
        b_ub=np.zeros(problem.inequalities.shape[0]),
        A_eq=problem.equilibrium,
        b_eq=-problem.baseline_load,
        bounds=[(None, None)] * force_count,
        method="highs",
        options=options,
    )
    if result.status == 0:
        return
    if result.status == 2:
        raise ValueError("The baseline load is infeasible for compression-only RBE.")
    raise RuntimeError("Baseline feasibility solve failed: {}.".format(result.message))


def _solve_radial(problem, direction, options):
    force_count = problem.equilibrium.shape[1]
    load_column = problem.load_basis.dot(direction).reshape((-1, 1))
    equality = hstack([problem.equilibrium, load_column], format="csr")
    inequalities = hstack(
        [problem.inequalities, csr_matrix((problem.inequalities.shape[0], 1))],
        format="csr",
    )
    objective = np.zeros(force_count + 1)
    objective[-1] = -1.0
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(inequalities.shape[0]),
        A_eq=equality,
        b_eq=-problem.baseline_load,
        bounds=[(None, None)] * force_count + [(0, None)],
        method="highs",
        options=options,
    )
    if result.status == 0:
        return "optimal", max(0.0, float(result.x[-1]))
    if result.status == 3:
        return "unbounded", None
    raise RuntimeError("Radial load solve failed: {}.".format(result.message))


def _solve_support(problem, direction, options):
    equilibrium_rows = problem.equilibrium.shape[0]
    inequality_rows = problem.inequalities.shape[0]
    zero_load_inequality = csr_matrix((2, inequality_rows))
    dual_equalities = bmat(
        [
            [problem.equilibrium.transpose(), problem.inequalities.transpose()],
            [problem.load_basis.transpose(), zero_load_inequality],
        ],
        format="csr",
    )
    right_hand_side = np.concatenate([np.zeros(problem.equilibrium.shape[1]), direction])
    objective = np.concatenate([-problem.baseline_load, np.zeros(inequality_rows)])
    result = linprog(
        objective,
        A_eq=dual_equalities,
        b_eq=right_hand_side,
        bounds=[(None, None)] * equilibrium_rows + [(0, None)] * inequality_rows,
        method="highs",
        options=options,
    )
    if result.status == 0:
        return "optimal", float(result.fun)
    if result.status == 2:
        return "unbounded", None
    raise RuntimeError("Dual support solve failed: {}.".format(result.message))


def _check_global_boundedness(problem, options):
    cardinal_directions = (
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    )
    return all(_solve_support(problem, direction, options)[0] == "optimal" for direction in cardinal_directions)


def _inner_polygon(points, tolerance):
    unique_points = _deduplicate_points(points, tolerance)
    if len(unique_points) < 3:
        return []
    try:
        hull = ConvexHull(np.asarray(unique_points))
    except QhullError:
        return []
    polygon = [unique_points[index] for index in hull.vertices]
    return _counterclockwise_polygon(polygon, tolerance)


def _outer_polygon(halfspaces, tolerance):
    if len(halfspaces) < 3:
        return []
    coefficients = np.asarray([halfspace[:2] for halfspace in halfspaces])
    limits = np.asarray([halfspace[2] for halfspace in halfspaces])
    intersections = []
    for first in range(len(halfspaces)):
        for second in range(first + 1, len(halfspaces)):
            matrix = coefficients[[first, second], :]
            determinant = np.linalg.det(matrix)
            if abs(determinant) <= tolerance:
                continue
            point = np.linalg.solve(matrix, limits[[first, second]])
            if np.all(coefficients.dot(point) <= limits + tolerance):
                intersections.append(point.tolist())

    unique_points = _deduplicate_points(intersections, tolerance)
    if len(unique_points) < 3:
        return []
    center = np.mean(np.asarray(unique_points), axis=0)
    polygon = sorted(unique_points, key=lambda point: math.atan2(point[1] - center[1], point[0] - center[0]))
    return _counterclockwise_polygon(polygon, tolerance)


def _deduplicate_points(points, tolerance):
    unique_points = []
    for point in points:
        candidate = np.asarray(point, dtype=float)
        if not any(np.linalg.norm(candidate - np.asarray(existing)) <= tolerance for existing in unique_points):
            unique_points.append(candidate.tolist())
    return unique_points


def _counterclockwise_polygon(polygon, tolerance):
    area_twice = sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )
    if abs(area_twice) <= tolerance:
        return []
    if area_twice < 0:
        polygon.reverse()
    return polygon


def _report(result, start_time, verbose, timer):
    if verbose:
        optimal_count = result.statuses.count("optimal")
        print(
            "robust RBE {}: {} of {} directions bounded; load set bounded={}".format(
                result.method,
                optimal_count,
                len(result.statuses),
                result.is_bounded,
            )
        )
    if timer:
        print("--- robust RBE time: {} seconds ---".format(time.time() - start_time))
