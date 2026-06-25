"""Verify the L10 boundary line from example 18.

This script checks the full three-block, block-0-fixed robust model from
``18_rbe_robust_3block.py``. It solves the 45-degree support LP, then checks
that points just inside, on, and outside the resulting ``Fx + Fz`` boundary
are feasible/infeasible in the same four-point hidden-load model.
"""

import importlib.util
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack

from compas_cra.equilibrium import plot_rbe_robust_results
from compas_cra.equilibrium.cra_helper import equilibrium_setup
from compas_cra.equilibrium.cra_helper import external_force_setup
from compas_cra.equilibrium.cra_helper import friction_setup
from compas_cra.equilibrium.cra_helper import num_vertices
from compas_cra.equilibrium.rbe_robust import _application_bound_rows
from compas_cra.equilibrium.rbe_robust import _application_load_basis
from compas_cra.equilibrium.rbe_robust import _prepare_problem

MU = 0.8
DENSITY = 1.0
APPLICATION_FORCE_BOUND = 1e3
LOAD_NODE = 2
LOAD_DOFS = [(LOAD_NODE, "fx"), (LOAD_NODE, "fz")]
L10_DIRECTION = np.asarray([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
DELTA = 1e-3
XLIM = (-1.0, 0.1)
YLIM = (-0.5, 1.0)
TENSION_TOLERANCE = 1e-7


def load_example18_module():
    """Load the numeric example-18 module by file path."""
    path = Path(__file__).with_name("18_rbe_robust_3block.py")
    spec = importlib.util.spec_from_file_location("example18", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_full_example18_assembly(example18):
    """Build the full three-block example-18 assembly with block 0 fixed."""
    geometry = example18.Arch(
        b0_width=0.5,
        b1_height=0.5,
        b1_base=0.5,
        b2_base=0.6,
        b2_top=0.9,
        alpha=math.pi / 2,
        beta=2 * math.pi / 3,
        gamma=2 * math.pi / 3,
        thickness=1,
    )
    return example18.build_assembly(geometry, block_nodes=[0, 1, 2], support_nodes=[0])


def prepare_l10_problem(example18):
    """Return the assembly, robust LP problem, and block-2 application points."""
    assembly = build_full_example18_assembly(example18)
    load_application_points = {
        LOAD_NODE: example18.rightmost_side_vertices(assembly.graph.node_attribute(LOAD_NODE, "block")),
    }
    problem = _prepare_problem(
        assembly,
        LOAD_DOFS,
        MU,
        DENSITY,
        None,
        load_application_points,
        APPLICATION_FORCE_BOUND,
    )
    return assembly, problem, load_application_points


def solve_support_direction(problem, direction):
    """Solve ``max direction dot u`` and return the HiGHS result."""
    objective = -np.asarray(problem.load_projection.transpose().dot(direction)).ravel()
    return linprog(
        objective,
        A_ub=problem.inequalities,
        b_ub=problem.inequality_rhs,
        A_eq=problem.equilibrium,
        b_eq=-problem.baseline_load,
        bounds=[(None, None)] * problem.equilibrium.shape[1],
        method="highs",
    )


def solve_fixed_visible_load(problem, load):
    """Check feasibility with the visible load fixed to ``load``."""
    equality = vstack([problem.equilibrium, problem.load_projection], format="csr")
    rhs = np.concatenate([-problem.baseline_load, np.asarray(load, dtype=float)])
    return linprog(
        np.zeros(problem.equilibrium.shape[1]),
        A_ub=problem.inequalities,
        b_ub=problem.inequality_rhs,
        A_eq=equality,
        b_eq=rhs,
        bounds=[(None, None)] * problem.equilibrium.shape[1],
        method="highs",
    )


def solve_penalty_tension_check(assembly, load_application_points, load):
    """Minimize tensile normal force needed to equilibrate one visible load."""
    base_equilibrium = equilibrium_setup(assembly, penalty=True)
    friction = friction_setup(assembly, MU, penalty=True)
    baseline_load = np.asarray(external_force_setup(assembly, DENSITY, None), dtype=float).ravel()
    base_force_count = base_equilibrium.shape[1]
    points = [np.asarray(point, dtype=float) for point in load_application_points[LOAD_NODE]]

    application_basis, load_projection = _application_load_basis(
        assembly,
        LOAD_DOFS,
        LOAD_NODE,
        points,
        base_equilibrium.shape[0],
        base_force_count,
    )
    application_force_count = application_basis.shape[1]
    equilibrium = hstack([base_equilibrium, application_basis], format="csr")
    friction = hstack(
        [friction, csr_matrix((friction.shape[0], application_force_count))],
        format="csr",
    )
    bound_rows, bound_rhs = _application_bound_rows(
        base_force_count,
        application_force_count,
        APPLICATION_FORCE_BOUND,
    )
    inequalities = vstack([friction, bound_rows], format="csr")
    inequality_rhs = np.concatenate([np.zeros(friction.shape[0]), bound_rhs])
    equality = vstack([equilibrium, load_projection], format="csr")
    equality_rhs = np.concatenate([-baseline_load, np.asarray(load, dtype=float)])

    contact_count = num_vertices(assembly)
    objective = np.zeros(equilibrium.shape[1])
    objective[1 : 4 * contact_count : 4] = 1.0
    bounds = []
    for _ in range(contact_count):
        bounds.extend([(0.0, None), (0.0, None), (None, None), (None, None)])
    bounds.extend([(None, None)] * application_force_count)

    return linprog(
        objective,
        A_ub=inequalities,
        b_ub=inequality_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
    )


def feasibility_label(result):
    """Return a compact text label for one fixed-load LP result."""
    if result.status == 0:
        return "feasible"
    if result.status == 2:
        return "infeasible"
    return "solver status {}".format(result.status)


def contact_table(assembly):
    """Return contact metadata in the same order as RBE force variables."""
    contacts = []
    contact_index = 0
    for edge in assembly.graph.edges(False):
        interfaces = assembly.graph.edge_attribute(edge, "interfaces") or []
        for interface_index, interface in enumerate(interfaces):
            for point_index, point in enumerate(interface.points):
                contacts.append(
                    {
                        "index": contact_index,
                        "edge": edge,
                        "interface_index": interface_index,
                        "point_index": point_index,
                        "point": point,
                        "frame": interface.frame,
                    }
                )
                contact_index += 1
    return contacts


def penalty_contact_forces(result, assembly):
    """Return penalty contact variables as ``[fn_plus, fn_minus, fu, fv]`` rows."""
    contact_count = num_vertices(assembly)
    return np.asarray(result.x[: 4 * contact_count], dtype=float).reshape((-1, 4))


def penalty_tension(result, assembly):
    """Return the total tensile normal force for one penalty LP result."""
    return float(np.sum(penalty_contact_forces(result, assembly)[:, 1]))


def report_penalty_tension(label, result, assembly):
    """Print the tensile contacts needed by one penalty LP result."""
    print("\npenalty tension diagnostic for {}:".format(label))
    if result.status != 0:
        print("  penalty LP failed: status {}, {}".format(result.status, result.message))
        return

    contact_forces = penalty_contact_forces(result, assembly)
    total_tension = penalty_tension(result, assembly)
    print("  minimum total fn_minus={:.9g}".format(total_tension))

    tensile_contacts = []
    for contact in contact_table(assembly):
        contact_index = contact["index"]
        fn_plus, fn_minus, fu, fv = contact_forces[contact_index]
        if fn_minus <= TENSION_TOLERANCE:
            continue
        effective_normal = fn_plus - fn_minus
        tangent = math.sqrt(fu * fu + fv * fv)
        if fn_plus > TENSION_TOLERANCE:
            utilization = tangent / (MU * fn_plus)
            utilization_text = "{:.9g}".format(utilization)
        else:
            utilization_text = "nan"
        tensile_contacts.append(
            "  edge {edge}, contact {index}, point {point_index}: "
            "fn+={fn_plus:.9g}, fn-={fn_minus:.9g}, fn_eff={fn_eff:.9g}, "
            "fu={fu:.9g}, fv={fv:.9g}, |ft|/(mu*fn+)={utilization}".format(
                edge=contact["edge"],
                index=contact_index,
                point_index=contact["point_index"],
                fn_plus=fn_plus,
                fn_minus=fn_minus,
                fn_eff=effective_normal,
                fu=fu,
                fv=fv,
                utilization=utilization_text,
            )
        )

    if not tensile_contacts:
        print("  no tensile contacts above {:.1e}".format(TENSION_TOLERANCE))
        return
    print("  contacts requiring tensile normal force:")
    for line in tensile_contacts:
        print(line)


def report_b1_b2_contact_state(problem, solution, assembly):
    """Print b1-b2 normal and friction usage at the L10 support solution."""
    force_count = 3 * num_vertices(assembly)
    contact_forces = np.asarray(solution[:force_count], dtype=float).reshape((-1, 3))
    slacks = problem.inequality_rhs - problem.inequalities.dot(solution)
    friction_rows = friction_setup(assembly, MU, penalty=False).shape[0]
    normal_row_start = friction_rows

    print("\nb1-b2 contact state at L10 support point:")
    for contact in contact_table(assembly):
        if contact["edge"] != (1, 2):
            continue
        contact_index = contact["index"]
        fn, fu, fv = contact_forces[contact_index]
        normal_force = max(0.0, fn)
        friction_slacks = slacks[8 * contact_index : 8 * contact_index + 8]
        normal_slack = slacks[normal_row_start + contact_index]
        if normal_force <= 1e-8:
            fv_ratio = math.nan
            state = "open"
        else:
            fv_ratio = fv / (MU * normal_force)
            state = "carrying"
        print(
            "  contact {index} point {point_index}: state={state}, "
            "fn={fn:.9g}, fu={fu:.9g}, fv={fv:.9g}, "
            "fv/(mu*fn)={ratio}, min_friction_slack={min_slack:.3g}, normal_slack={normal_slack:.3g}".format(
                index=contact_index,
                point_index=contact["point_index"],
                state=state,
                fn=fn,
                fu=fu,
                fv=fv,
                ratio="nan" if math.isnan(fv_ratio) else "{:.9g}".format(fv_ratio),
                min_slack=float(np.min(friction_slacks)),
                normal_slack=float(normal_slack),
            )
        )


def report_hidden_forces(problem, solution, assembly, application_points):
    """Print the four hidden point forces that generate visible force and moment."""
    force_count = 3 * num_vertices(assembly)
    hidden = np.asarray(solution[force_count:], dtype=float)
    block = assembly.graph.node_attribute(LOAD_NODE, "block")
    center = np.asarray([block.center().x, block.center().y, block.center().z], dtype=float)

    print("\nfour-point hidden load variables at L10 support point:")
    resultant = np.zeros(3)
    resultant_moment = np.zeros(3)
    for point_index, point in enumerate(application_points[LOAD_NODE]):
        point = np.asarray(point, dtype=float)
        force = np.asarray([hidden[2 * point_index], 0.0, hidden[2 * point_index + 1]])
        moment = np.cross(point - center, force)
        resultant += force
        resultant_moment += moment
        print(
            "  q{} at {}: qx={:.9g}, qz={:.9g}, My={:.9g}".format(
                point_index,
                [round(value, 6) for value in point.tolist()],
                force[0],
                force[2],
                moment[1],
            )
        )
    print("  resultant Fx={:.9g}, Fz={:.9g}, My={:.9g}".format(resultant[0], resultant[2], resultant_moment[1]))


def plot_l10_check(example18, assembly, support_load, inside_load, outside_load, constant):
    """Save a small SVG showing the L10 line and three checked points."""
    options = {
        "mu": MU,
        "density": DENSITY,
        "application_force_bound": APPLICATION_FORCE_BOUND,
        "num_directions": 72,
    }
    results = example18.robust_case_results(assembly, LOAD_NODE, options)
    figure, axes = plot_rbe_robust_results(
        results,
        labels=["radial", "primal", "dual"],
        xlim=XLIM,
        ylim=YLIM,
    )

    xs = np.asarray(XLIM)
    axes.plot(xs, -xs + constant, color="black", linewidth=1.5, label="L10: Fx + Fz = {:.6g}".format(constant))
    points = np.asarray([inside_load, support_load, outside_load])
    axes.scatter(points[:, 0], points[:, 1], color=["green", "black", "red"], marker="o", zorder=5)
    for label, point in zip(("inside", "on L10", "outside"), points):
        axes.annotate(label, xy=point, xytext=(8, 8), textcoords="offset points")
    axes.set_title("L10 feasibility check")
    legend = axes.legend(loc="upper right", fontsize=7)
    legend.set_draggable(True)
    figure.tight_layout()
    figure.savefig("docs/examples/18_special_check_l10.svg", bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    example18 = load_example18_module()
    assembly, problem, application_points = prepare_l10_problem(example18)
    support = solve_support_direction(problem, L10_DIRECTION)
    if support.status != 0:
        raise RuntimeError("L10 support solve failed: {}".format(support.message))

    support_load = np.asarray(problem.load_projection.dot(support.x), dtype=float).ravel()
    support_value = float(L10_DIRECTION.dot(support_load))
    l10_constant = support_value / L10_DIRECTION[1]
    inside_load = support_load - DELTA * L10_DIRECTION
    outside_load = support_load + DELTA * L10_DIRECTION

    print("L10 support direction: {}".format(L10_DIRECTION.tolist()))
    print("support point: Fx={:.9g}, Fz={:.9g}".format(support_load[0], support_load[1]))
    print("support value d.u={:.9g}".format(support_value))
    print("L10 line: Fx + Fz = {:.9g}".format(l10_constant))
    print("L10 line: Fz = -Fx + {:.9g}".format(l10_constant))

    checks = [
        ("inside", inside_load),
        ("on L10", support_load),
        ("outside", outside_load),
    ]
    print("\nfixed visible-load feasibility checks:")
    for label, load in checks:
        result = solve_fixed_visible_load(problem, load)
        penalty_result = solve_penalty_tension_check(assembly, application_points, load)
        if penalty_result.status == 0:
            penalty_state = "penalty min fn-={:.9g}".format(penalty_tension(penalty_result, assembly))
        else:
            penalty_state = "penalty status {}".format(penalty_result.status)
        print(
            "  {label}: Fx={fx:.9g}, Fz={fz:.9g}, Fx+Fz={total:.9g} -> {state}; {penalty_state}".format(
                label=label,
                fx=load[0],
                fz=load[1],
                total=float(load[0] + load[1]),
                state=feasibility_label(result),
                penalty_state=penalty_state,
            )
        )
        report_penalty_tension(label, penalty_result, assembly)

    report_b1_b2_contact_state(problem, support.x, assembly)
    report_hidden_forces(problem, support.x, assembly, application_points)
    plot_l10_check(example18, assembly, support_load, inside_load, outside_load, l10_constant)
