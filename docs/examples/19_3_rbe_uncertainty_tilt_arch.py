"""Compare four-block arch safe-load regions under foundation tilt uncertainty."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from compas_assembly.datastructures import Block

from compas_cra.algorithms import assembly_interfaces_numpy
from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import plot_rbe_robust_results
from compas_cra.equilibrium import rbe_uncertainty_disturb_support_dual
from compas_cra.geometry import Arch

HEIGHT = 2
SPAN = 4
THICKNESS = 0.5
DEPTH = 0.5
NUM_BLOCKS = 4
MU = 0.7
DENSITY = 1.0
NUM_DIRECTIONS = 72
APPLICATION_FORCE_BOUND = 1e6
TILT_DEGREES = (0.0, 5.0, 10.0, 15.0, 20.0)
TILT_ANGLE_COUNT = 2
VIEW_XLIM = (-2.0, 0.0)
VIEW_YLIM = (0.0, 1.0)
OUTPUT_SVG = Path(__file__).with_suffix(".svg")


def point_coordinates(point):
    """Return point coordinates as a plain list."""
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return [point.x, point.y, point.z]
    return list(point)


def rightmost_side_vertices(block):
    """Return the four vertices of the block face with maximum x-coordinate."""
    face_coordinates = []
    for face in block.faces():
        coordinates = [point_coordinates(block.vertex_coordinates(vertex)) for vertex in block.face_vertices(face)]
        face_coordinates.append(coordinates)
    return max(face_coordinates, key=lambda coordinates: sum(point[0] for point in coordinates) / len(coordinates))


def left_to_right_blocks():
    """Create four arch blocks ordered left to right."""
    arch = Arch(
        height=HEIGHT,
        span=SPAN,
        thickness=THICKNESS,
        depth=DEPTH,
        num_blocks=NUM_BLOCKS,
        extra_support=False,
    )
    return list(reversed(arch.blocks()))


def build_assembly():
    """Build the fixed four-block arch assembly."""
    assembly = CRA_Assembly()
    for node, mesh in enumerate(left_to_right_blocks()):
        assembly.add_block(mesh.copy(cls=Block), node=node)
    assembly.set_boundary_conditions([0])
    assembly_interfaces_numpy(assembly, nmax=10, amin=1e-2, tmax=1e-2)
    return assembly


def tilt_label(degrees):
    """Return a compact label for one symmetric tilt interval."""
    if degrees == 0.0:
        return "0°"
    value = "{:.1f}".format(degrees).rstrip("0").rstrip(".")
    return "±{}°".format(value)


def report_result(degrees, scenario_count, result):
    """Print a compact tilt-analysis summary."""
    bounded_directions = result.statuses.count("optimal")
    print(
        "{:<7} | scenarios={} | bounded={} | center={} | bounded_dirs={}/{} | outer_polygon_empty={}".format(
            tilt_label(degrees),
            scenario_count,
            result.is_bounded,
            result.feasible_center,
            bounded_directions,
            len(result.directions),
            not bool(result.outer_polygon),
        )
    )


def solve_tilt_cases(assembly):
    """Solve baseline and tilt-robust block-3 safe-load regions."""
    block_3 = assembly.graph.node_attribute(3, "block")
    load_dofs = [(3, "fx"), (3, "fz")]
    common_options = {
        "mu": MU,
        "density": DENSITY,
        "load_application_points": {3: rightmost_side_vertices(block_3)},
        "application_force_bound": APPLICATION_FORCE_BOUND,
        "num_directions": NUM_DIRECTIONS,
    }

    print("hidden hand-force component bound: {:.6g} (placeholder)".format(APPLICATION_FORCE_BOUND))
    print("foundation tilt changes gravity direction only; geometry/contact/friction are fixed")
    print("case    | tilt scenario summary")

    results = []
    labels = []
    for degrees in TILT_DEGREES:
        if degrees == 0.0:
            result = rbe_uncertainty_disturb_support_dual(assembly, load_dofs, **common_options)
            scenario_count = 1
        else:
            angle = math.radians(degrees)
            result = rbe_uncertainty_disturb_support_dual(
                assembly,
                load_dofs,
                tilt_angle_bounds=(-angle, angle),
                tilt_angle_count=TILT_ANGLE_COUNT,
                **common_options,
            )
            scenario_count = TILT_ANGLE_COUNT
        report_result(degrees, scenario_count, result)
        results.append(result)
        labels.append(tilt_label(degrees))
    return results, labels


def render_results(results, labels):
    """Render the four-block tilt comparison plot."""
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plot_rbe_robust_results(
        results,
        labels=labels,
        show_points=False,
        show_center=True,
        xlim=VIEW_XLIM,
        ylim=VIEW_YLIM,
    )
    figure.set_size_inches(10, 6)
    axes.set_xlabel("Block 3 load Fx")
    axes.set_ylabel("Block 3 load Fz")
    axes.set_title("Four-block arch safe-load regions with foundation tilt uncertainty")
    axes.text(
        0.02,
        0.02,
        "Tilt rotates gravity in the XZ plane only\n"
        "Block 3 load uses four rightmost-side points; force bound = 1e6 placeholder",
        transform=axes.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    figure.tight_layout()
    return figure


def save_svg(figure, path):
    """Save one Matplotlib figure as SVG and report the path."""
    figure.savefig(path, format="svg", bbox_inches="tight")
    print("saved SVG: {}".format(path))


if __name__ == "__main__":
    assembly = build_assembly()
    results, labels = solve_tilt_cases(assembly)
    figure = render_results(results, labels)
    save_svg(figure, OUTPUT_SVG)

    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
