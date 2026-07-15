"""Compare four-block arch safe-load regions under disturbance uncertainty."""

from itertools import product
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
DISTURBANCE_RATIOS = (0.0, 0.05, 0.10, 0.15, 0.20)
VIEW_XLIM = (-2.0, 0.0)
VIEW_YLIM = (0.0, 1.0)
OUTPUT_SVG = Path(__file__).with_suffix(".svg")


def point_coordinates(point):
    """Return point coordinates as a plain list."""
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return [point.x, point.y, point.z]
    return list(point)


def arch_center():
    """Return the center of the arch circular axis."""
    radius = HEIGHT / 2.0 + SPAN**2 / (8.0 * HEIGHT)
    return [0.0, 0.0, HEIGHT - radius]


def unique_xz_points(coordinates, tolerance=1e-9):
    """Return unique face points by x-z coordinates."""
    unique = []
    for point in coordinates:
        if not any(
            abs(point[0] - other[0]) <= tolerance and abs(point[2] - other[2]) <= tolerance for other in unique
        ):
            unique.append(point)
    return unique


def radial_alignment_score(points):
    """Return how far two x-z points deviate from one arch radial line."""
    center = arch_center()
    vectors = [[point[0] - center[0], point[2] - center[2]] for point in points]
    lengths = [(vector[0] ** 2 + vector[1] ** 2) ** 0.5 for vector in vectors]
    if min(lengths) <= 1e-12:
        return None
    cross = vectors[0][0] * vectors[1][1] - vectors[0][1] * vectors[1][0]
    return abs(cross) / (lengths[0] * lengths[1])


def exposed_radial_side_vertices(block):
    """Return vertices of the loaded block's right exposed radial face."""
    candidates = []
    for face in block.faces():
        coordinates = [point_coordinates(block.vertex_coordinates(vertex)) for vertex in block.face_vertices(face)]
        y_values = [point[1] for point in coordinates]
        if max(y_values) - min(y_values) <= 1e-9:
            continue
        unique_points = unique_xz_points(coordinates)
        if len(unique_points) != 2:
            continue
        score = radial_alignment_score(unique_points)
        if score is None or score > 1e-8:
            continue
        centroid_x = sum(point[0] for point in coordinates) / len(coordinates)
        candidates.append((centroid_x, coordinates))
    if not candidates:
        raise ValueError("Could not find an exposed radial load face for the arch block.")
    return max(candidates, key=lambda item: item[0])[1]


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


def disturbance_vertices(block_1_bound, block_2_bound):
    """Return the 16 vertices of independent block-1 and block-2 disturbance boxes."""
    return [
        list(vertex)
        for vertex in product(
            (-block_1_bound, block_1_bound),
            (-block_1_bound, block_1_bound),
            (-block_2_bound, block_2_bound),
            (-block_2_bound, block_2_bound),
        )
    ]


def disturbance_label(ratio):
    """Return a compact label for one disturbance ratio."""
    return "{:.0f}%".format(100.0 * ratio)


def report_result(ratio, block_1_bound, block_2_bound, result):
    """Print a compact disturbance-analysis summary."""
    bounded_directions = result.statuses.count("optimal")
    print(
        "{:<4} | b1_bound={:.6g} | b2_bound={:.6g} | bounded={} | center={} | "
        "bounded_dirs={}/{} | outer_polygon_empty={}".format(
            disturbance_label(ratio),
            block_1_bound,
            block_2_bound,
            result.is_bounded,
            result.feasible_center,
            bounded_directions,
            len(result.directions),
            not bool(result.outer_polygon),
        )
    )


def solve_disturbance_cases(assembly):
    """Solve baseline and disturbance-robust block-3 safe-load regions."""
    block_1 = assembly.graph.node_attribute(1, "block")
    block_2 = assembly.graph.node_attribute(2, "block")
    block_3 = assembly.graph.node_attribute(3, "block")
    block_1_weight = block_1.volume() * DENSITY
    block_2_weight = block_2.volume() * DENSITY
    load_dofs = [(3, "fx"), (3, "fz")]
    common_options = {
        "mu": MU,
        "density": DENSITY,
        "load_application_points": {3: exposed_radial_side_vertices(block_3)},
        "application_force_bound": APPLICATION_FORCE_BOUND,
        "num_directions": NUM_DIRECTIONS,
    }

    print("block 1 weight: {:.6g}".format(block_1_weight))
    print("block 2 weight: {:.6g}".format(block_2_weight))
    print("hidden hand-force component bound: {:.6g} (placeholder)".format(APPLICATION_FORCE_BOUND))
    print("disturbance boxes: Fx,Fz independently in [-r Wi, r Wi] on blocks 1 and 2")
    print("case | block disturbance bounds | safe-load summary")

    results = []
    labels = []
    for ratio in DISTURBANCE_RATIOS:
        block_1_bound = ratio * block_1_weight
        block_2_bound = ratio * block_2_weight
        if ratio == 0.0:
            result = rbe_uncertainty_disturb_support_dual(assembly, load_dofs, **common_options)
        else:
            result = rbe_uncertainty_disturb_support_dual(
                assembly,
                load_dofs,
                uncertainty_vertices=disturbance_vertices(block_1_bound, block_2_bound),
                uncertainty_load_dofs=[(1, "fx"), (1, "fz"), (2, "fx"), (2, "fz")],
                **common_options,
            )
        report_result(ratio, block_1_bound, block_2_bound, result)
        results.append(result)
        labels.append(disturbance_label(ratio))
    return results, labels


def render_results(results, labels):
    """Render the four-block disturbance comparison plot."""
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
    axes.set_title("Four-block arch safe-load regions with block-1/2 disturbance uncertainty")
    axes.text(
        0.02,
        0.02,
        "Blocks 1 and 2: independent Fx,Fz disturbances in +/- r Wi\n"
        "Block 3 load uses four right exposed radial-face points; force bound = 1e6 placeholder",
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
    results, labels = solve_disturbance_cases(assembly)
    figure = render_results(results, labels)
    save_svg(figure, OUTPUT_SVG)

    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
