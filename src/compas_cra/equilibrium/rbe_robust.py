r"""Robust rigid-block equilibrium force-range analysis.

For ``B`` free blocks and ``V`` interface vertices, the safe load set is

.. math::

    U = \{u \in \mathbb{R}^2 \mid \exists f:
    A f + T u = -p_0,\ H f \leq 0\}.

Here, ``A`` is the ``(6B, 3V)`` equilibrium matrix, ``H`` is the
``(9V, 3V)`` friction and compression matrix, ``p0`` is the ``(6B,)``
baseline wrench vector, and ``T`` is the ``(6B, 2)`` basis for the selected
load components. The variables ``f`` and ``u`` contain ``3V`` contact-force
components and two unknown load increments, respectively.
"""

import math
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

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
    feasible_center : list[float]
        Feasible load point used as the center of radial sampling. This is the
        origin when the origin is feasible.
    support_formulation : str, optional
        ``"primal"`` or ``"dual"`` for support analyses. ``None`` for radial
        sampling.
    radial_limits : list[float | None]
        Maximum distances from ``feasible_center``. Populated by radial
        sampling.
    boundary_points : list[list[float]]
        Finite radial boundary points. Populated by radial sampling.
    support_values : list[float | None]
        Support-function values. Populated by primal and dual support methods.
    support_points : list[list[float]]
        Feasible load points returned by the primal support method.
    halfspaces : list[list[float]]
        Finite supporting halfspaces stored as ``[d0, d1, h]``, representing
        ``d0 * b0 + d1 * b1 <= h``.
    inner_polygon : list[list[float]]
        Counterclockwise inner approximation from radial or primal support
        points.
    outer_polygon : list[list[float]]
        Counterclockwise outer approximation from supporting halfspaces.
    polygon : list[list[float]]
        Backward-compatible polygon alias. This is ``inner_polygon`` for radial
        sampling and ``outer_polygon`` for both support formulations.
    """

    method: str
    load_dofs: Tuple[Tuple[Any, str], Tuple[Any, str]]
    directions: List[List[float]]
    statuses: List[str]
    origin_feasible: bool
    is_bounded: bool
    feasible_center: List[float] = field(default_factory=list)
    support_formulation: Optional[str] = None
    radial_limits: List[Optional[float]] = field(default_factory=list)
    boundary_points: List[List[float]] = field(default_factory=list)
    support_values: List[Optional[float]] = field(default_factory=list)
    support_points: List[List[float]] = field(default_factory=list)
    halfspaces: List[List[float]] = field(default_factory=list)
    inner_polygon: List[List[float]] = field(default_factory=list)
    outer_polygon: List[List[float]] = field(default_factory=list)
    polygon: List[List[float]] = field(default_factory=list)


@dataclass
class _RobustProblem:
    """Sparse matrices defining ``A f + T u = -p0`` and ``H f <= 0``."""

    equilibrium: csr_matrix
    inequalities: csr_matrix
    baseline_load: np.ndarray
    load_basis: csr_matrix
    load_dofs: Tuple[Tuple[Any, str], Tuple[Any, str]]


@dataclass
class _LoadSetAnalysis:
    """Feasibility, boundedness, and center shared by all robust methods."""

    origin_feasible: bool
    is_bounded: bool
    feasible_center: np.ndarray


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
    defined by gravity and ``external_forces``. Rays start at a feasible load
    point, which is not necessarily the origin. Contact forces use the
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
        If the input is invalid or the complete safe load set is empty.
    RuntimeError
        If HiGHS reports an unexpected numerical or solver failure.
    """
    start_time = time.time()
    problem = _prepare_problem(assembly, load_dofs, mu, density, external_forces)
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set(problem, options)

    statuses = []
    radial_limits = []
    boundary_points = []
    for direction in directions:
        status, limit = _solve_radial(problem, direction, analysis.feasible_center, options)
        statuses.append(status)
        radial_limits.append(limit)
        if limit is not None:
            boundary_points.append((analysis.feasible_center + limit * direction).tolist())

    inner_polygon = _inner_polygon(boundary_points, tolerance) if analysis.is_bounded else []
    result = RobustForceResult(
        method="sample",
        load_dofs=problem.load_dofs,
        directions=[direction.tolist() for direction in directions],
        statuses=statuses,
        origin_feasible=analysis.origin_feasible,
        is_bounded=analysis.is_bounded,
        feasible_center=analysis.feasible_center.tolist(),
        radial_limits=radial_limits,
        boundary_points=boundary_points,
        inner_polygon=inner_polygon,
        polygon=inner_polygon,
    )
    _report(result, start_time, verbose, timer)
    return result


def rbe_robust_support_primal(
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
    r"""Approximate a two-dimensional safe load range with primal supports.

    For each unit direction ``d``, this function solves

    .. math::

        \max_{f,u}\ d^T u
        \quad\mathrm{s.t.}\quad
        A f + T u = -p_0,\ H f \leq 0.

    Each solve returns a feasible support point for an inner convex hull and a
    support value for an outer halfspace intersection.

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
        Support points and values, supporting halfspaces, and inner and outer
        polygons.

    Raises
    ------
    ValueError
        If the input is invalid or the complete safe load set is empty.
    RuntimeError
        If HiGHS reports an unexpected numerical or solver failure.
    """
    start_time = time.time()
    problem = _prepare_problem(assembly, load_dofs, mu, density, external_forces)
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set(problem, options)

    statuses = []
    support_values = []
    support_points = []
    halfspaces = []
    for direction in directions:
        status, support, point = _solve_primal_support(problem, direction, options)
        statuses.append(status)
        support_values.append(support)
        if support is not None:
            halfspaces.append([float(direction[0]), float(direction[1]), support])
        if point is not None:
            support_points.append(point)

    inner_polygon = _inner_polygon(support_points, tolerance) if analysis.is_bounded else []
    outer_polygon = _outer_polygon(halfspaces, tolerance) if analysis.is_bounded else []
    result = RobustForceResult(
        method="support",
        load_dofs=problem.load_dofs,
        directions=[direction.tolist() for direction in directions],
        statuses=statuses,
        origin_feasible=analysis.origin_feasible,
        is_bounded=analysis.is_bounded,
        feasible_center=analysis.feasible_center.tolist(),
        support_formulation="primal",
        support_values=support_values,
        support_points=support_points,
        halfspaces=halfspaces,
        inner_polygon=inner_polygon,
        outer_polygon=outer_polygon,
        polygon=outer_polygon,
    )
    _report(result, start_time, verbose, timer)
    return result


def rbe_robust_support_dual(
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

    Each solve is the linear-programming dual of the primal support problem
    and produces a supporting halfspace ``d^T u <= h``. The intersection of
    finite halfspaces is an outer approximation of the safe load set.

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
        If the input is invalid or the complete safe load set is empty.
    RuntimeError
        If HiGHS reports an unexpected numerical or solver failure.
    """
    start_time = time.time()
    problem = _prepare_problem(assembly, load_dofs, mu, density, external_forces)
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set(problem, options)

    statuses = []
    support_values = []
    halfspaces = []
    for direction in directions:
        status, support = _solve_dual_support(problem, direction, options)
        statuses.append(status)
        support_values.append(support)
        if support is not None:
            halfspaces.append([float(direction[0]), float(direction[1]), support])

    outer_polygon = _outer_polygon(halfspaces, tolerance) if analysis.is_bounded else []
    result = RobustForceResult(
        method="support",
        load_dofs=problem.load_dofs,
        directions=[direction.tolist() for direction in directions],
        statuses=statuses,
        origin_feasible=analysis.origin_feasible,
        is_bounded=analysis.is_bounded,
        feasible_center=analysis.feasible_center.tolist(),
        support_formulation="dual",
        support_values=support_values,
        halfspaces=halfspaces,
        outer_polygon=outer_polygon,
        polygon=outer_polygon,
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
    """Backward-compatible alias for :func:`rbe_robust_support_dual`."""
    return rbe_robust_support_dual(
        assembly=assembly,
        load_dofs=load_dofs,
        mu=mu,
        density=density,
        external_forces=external_forces,
        num_directions=num_directions,
        tolerance=tolerance,
        solver_options=solver_options,
        verbose=verbose,
        timer=timer,
    )


def plot_rbe_robust_results(
    results: Union[RobustForceResult, Sequence[RobustForceResult]],
    labels: Optional[Sequence[str]] = None,
    ax=None,
    show_points: bool = True,
    show_center: bool = True,
    show_origin: bool = True,
):
    """Plot one or more two-dimensional robust RBE result regions.

    Matplotlib is imported only when this function is called. Inner polygons
    are filled, outer polygons are drawn with dashed boundaries, and finite
    sample or support points remain visible for unbounded or degenerate sets.

    Parameters
    ----------
    results : :class:`RobustForceResult` or sequence
        One result or multiple results with identical ``load_dofs``.
    labels : sequence[str], optional
        Legend labels. Defaults describe each result method.
    ax : :class:`matplotlib.axes.Axes`, optional
        Existing axes. A new figure and axes are created when omitted.
    show_points : bool, optional
        Draw radial boundary points or primal support points.
    show_center : bool, optional
        Draw each feasible sampling center.
    show_origin : bool, optional
        Draw the load-increment origin.

    Returns
    -------
    tuple
        Matplotlib ``(figure, axes)``.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    ValueError
        If no results are provided, labels do not match, or load DOFs differ.
    TypeError
        If an item is not a :class:`RobustForceResult`.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("plot_rbe_robust_results requires the optional Matplotlib package.") from error

    if isinstance(results, RobustForceResult):
        result_list = [results]
    else:
        result_list = list(results)
    if not result_list:
        raise ValueError("results must contain at least one RobustForceResult.")
    if not all(isinstance(result, RobustForceResult) for result in result_list):
        raise TypeError("results must contain only RobustForceResult instances.")

    load_dofs = result_list[0].load_dofs
    if any(result.load_dofs != load_dofs for result in result_list[1:]):
        raise ValueError("All robust results must use the same load_dofs.")

    if labels is None:
        labels = [_result_label(result) for result in result_list]
    else:
        labels = list(labels)
        if len(labels) != len(result_list):
            raise ValueError("labels must have the same length as results.")

    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = axes.figure

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    for index, (result, label) in enumerate(zip(result_list, labels)):
        color = colors[index % len(colors)]
        if result.inner_polygon:
            _plot_polygon(
                axes,
                result.inner_polygon,
                color=color,
                linestyle="-",
                alpha=0.18,
                label="{} inner".format(label),
            )
        if result.outer_polygon:
            _plot_polygon(
                axes,
                result.outer_polygon,
                color=color,
                linestyle="--",
                alpha=0.0,
                label="{} outer".format(label),
            )
        if show_points:
            points = result.support_points if result.support_points else result.boundary_points
            if points:
                coordinates = np.asarray(points)
                axes.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    color=color,
                    marker=".",
                    label="{} points".format(label),
                )
        if show_center and result.feasible_center:
            axes.scatter(
                [result.feasible_center[0]],
                [result.feasible_center[1]],
                color=color,
                marker="x",
                s=55,
                label="{} center".format(label),
            )

    if show_origin:
        axes.scatter([0.0], [0.0], color="black", marker="+", s=65, label="origin")

    axes.set_xlabel(_load_dof_label(load_dofs[0]))
    axes.set_ylabel(_load_dof_label(load_dofs[1]))
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(True, alpha=0.3)
    axes.legend()
    return figure, axes


def _prepare_problem(assembly, load_dofs, mu, density, external_forces):
    """Construct ``A``, ``H``, ``p0``, and ``T`` for robust analysis.

    ``A`` has one six-component equilibrium row block per free body and three
    contact-force columns per interface vertex. ``H`` contains eight
    linearized friction inequalities and one compression inequality per
    interface vertex. ``T`` maps the two selected load increments into the
    equilibrium rows.
    """
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


def _analyze_load_set(problem, options):
    """Find a feasible load center and determine global boundedness."""
    origin_feasible = _is_origin_feasible(problem, options)
    feasibility_witness = _solve_load_set_feasibility(problem, options)
    cardinal_directions = (
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    )
    cardinal_results = [_solve_primal_support(problem, direction, options) for direction in cardinal_directions]
    is_bounded = all(status == "optimal" for status, _, _ in cardinal_results)

    if origin_feasible:
        feasible_center = np.zeros(2)
    elif is_bounded:
        feasible_center = np.mean(
            np.asarray([point for _, _, point in cardinal_results if point is not None]),
            axis=0,
        )
    else:
        feasible_center = feasibility_witness

    return _LoadSetAnalysis(
        origin_feasible=origin_feasible,
        is_bounded=is_bounded,
        feasible_center=np.asarray(feasible_center, dtype=float),
    )


def _is_origin_feasible(problem, options):
    """Return whether the zero unknown-load increment belongs to the load set."""
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
        return True
    if result.status == 2:
        return False
    raise RuntimeError("Origin feasibility solve failed: {}.".format(result.message))


def _solve_load_set_feasibility(problem, options):
    """Return one feasible ``u`` or raise when the complete safe set is empty."""
    force_count = problem.equilibrium.shape[1]
    inequality_rows = problem.inequalities.shape[0]
    equality = hstack([problem.equilibrium, problem.load_basis], format="csr")
    inequalities = hstack(
        [problem.inequalities, csr_matrix((inequality_rows, 2))],
        format="csr",
    )
    result = linprog(
        np.zeros(force_count + 2),
        A_ub=inequalities,
        b_ub=np.zeros(inequality_rows),
        A_eq=equality,
        b_eq=-problem.baseline_load,
        bounds=[(None, None)] * (force_count + 2),
        method="highs",
        options=options,
    )
    if result.status == 0:
        return np.asarray(result.x[force_count : force_count + 2], dtype=float)
    if result.status == 2:
        raise ValueError("The safe load set is empty for compression-only RBE.")
    raise RuntimeError("Safe load-set feasibility solve failed: {}.".format(result.message))


def _solve_radial(problem, direction, center, options):
    """Maximize distance ``alpha`` from a verified feasible load center."""
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
        b_eq=-problem.baseline_load - problem.load_basis.dot(center),
        bounds=[(None, None)] * force_count + [(0, None)],
        method="highs",
        options=options,
    )
    if result.status == 0:
        return "optimal", max(0.0, float(result.x[-1]))
    if result.status == 3:
        return "unbounded", None
    raise RuntimeError("Radial load solve failed: {}.".format(result.message))


def _solve_primal_support(problem, direction, options):
    """Solve ``max d^T u`` and return its value and feasible support point."""
    force_count = problem.equilibrium.shape[1]
    inequality_rows = problem.inequalities.shape[0]
    equality = hstack([problem.equilibrium, problem.load_basis], format="csr")
    inequalities = hstack(
        [problem.inequalities, csr_matrix((inequality_rows, 2))],
        format="csr",
    )
    objective = np.zeros(force_count + 2)
    objective[force_count:] = -np.asarray(direction, dtype=float)
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(inequality_rows),
        A_eq=equality,
        b_eq=-problem.baseline_load,
        bounds=[(None, None)] * (force_count + 2),
        method="highs",
        options=options,
    )
    if result.status == 0:
        point = np.asarray(result.x[force_count : force_count + 2], dtype=float)
        support = float(np.asarray(direction, dtype=float).dot(point))
        return "optimal", support, point.tolist()
    if result.status == 3:
        return "unbounded", None, None
    raise RuntimeError("Primal support solve failed: {}.".format(result.message))


def _solve_dual_support(problem, direction, options):
    """Solve the dual support LP for one direction ``d``."""
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


def _plot_polygon(axes, polygon, color, linestyle, alpha, label):
    """Draw a closed polygon boundary and optional translucent fill."""
    coordinates = np.asarray(polygon + [polygon[0]])
    axes.plot(
        coordinates[:, 0],
        coordinates[:, 1],
        color=color,
        linestyle=linestyle,
        label=label,
    )
    if alpha:
        axes.fill(coordinates[:, 0], coordinates[:, 1], color=color, alpha=alpha)


def _result_label(result):
    if result.method == "sample":
        return "radial sample"
    return "{} support".format(result.support_formulation or "dual")


def _load_dof_label(load_dof):
    node, component = load_dof
    return "node {!r} {}".format(node, component)


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
