"""Compare robust RBE safe-load approximations for a three-block assembly."""

import math

import matplotlib.pyplot as plt
from compas.datastructures import Mesh
from compas.geometry import Point
from compas.geometry import Translation
from compas_assembly.datastructures import Block

from compas_cra.algorithms import assembly_interfaces_numpy
from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import plot_rbe_robust_results
from compas_cra.equilibrium import rbe_robust_sample
from compas_cra.equilibrium import rbe_robust_support_dual
from compas_cra.equilibrium import rbe_robust_support_primal

SHOW_BOUNDARY_LINES = True
PRINT_BOUNDARY_EQUATIONS = True
XLIM = (-1.0, 0.1)
YLIM = (-0.5, 1.0)


class Arch(object):
    """Create the three-block geometry used for the robust RBE comparison.

    Parameters
    ----------
    b0_width : float
        Width of block 0.
    b1_height : float
        Height of block 1.
    b1_base : float
        Base length of block 1.
    b2_base : float
        Base length of block 2.
    b2_top : float
        Top length of block 2.
    alpha : float
        Left bottom angle of block 1 in radians.
    beta : float
        Right bottom angle of block 1 in radians.
    gamma : float
        Right bottom angle of block 2 in radians.
    thickness : float
        Assembly thickness.
    """

    def __init__(self, b0_width, b1_height, b1_base, b2_base, b2_top, alpha, beta, gamma, thickness):
        super().__init__()
        self.b0_width = b0_width
        self.b1_height = b1_height
        self.b1_base = b1_base
        self.b2_base = b2_base
        self.b2_top = b2_top
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.thickness = thickness

    def assembly(self):
        """Create the assembly with block 0 fixed."""
        assembly = CRA_Assembly()
        for mesh in self.blocks():
            assembly.add_block(mesh.copy(cls=Block))
        assembly.graph.node_attribute(0, "is_support", True)
        return assembly

    def blocks(self):
        """Create the three block meshes."""
        blocks = []
        faces = [
            [0, 1, 2, 3],
            [7, 6, 5, 4],
            [3, 7, 4, 0],
            [6, 2, 1, 5],
            [7, 3, 2, 6],
            [5, 1, 0, 4],
        ]
        translation = Translation.from_vector([0, self.thickness, 0])

        b0_1 = Point(0, 0, 0)
        b0_2 = Point(-self.b1_height * math.tan(self.alpha - math.pi / 2), 0, self.b1_height)
        b0_0 = Point(
            self.b0_width * math.cos(self.alpha + math.pi / 2),
            0,
            self.b0_width * math.sin(self.alpha + math.pi / 2),
        )
        b0_3 = Point((b0_0 + b0_2 - b0_1).x, (b0_0 + b0_2 - b0_1).y, (b0_0 + b0_2 - b0_1).z)
        b0_front = [b0_0, b0_1, b0_2, b0_3]
        b0_back = [vertex.transformed(translation) for vertex in b0_front]
        blocks.append(Mesh.from_vertices_and_faces(b0_front + b0_back, faces))

        b1_0 = b0_1
        b1_3 = b0_2
        b1_1 = Point(self.b1_base, 0, 0)
        b1_2 = Point(
            self.b1_base + self.b1_height * math.tan(self.beta - math.pi / 2),
            0,
            self.b1_height,
        )
        b1_front = [b1_0, b1_1, b1_2, b1_3]
        b1_back = [vertex.transformed(translation) for vertex in b1_front]
        blocks.append(Mesh.from_vertices_and_faces(b1_front + b1_back, faces))

        b2_0 = b1_1
        b2_3 = b1_2
        b2_1 = Point(
            self.b1_base + self.b2_base * math.cos(math.pi - self.beta - self.gamma),
            0,
            self.b2_base * math.sin(math.pi - self.beta - self.gamma),
        )
        b2_direction = (b2_1 - b2_0).unitized()
        b2_2 = Point(
            (b2_direction * self.b2_top + b2_3).x,
            (b2_direction * self.b2_top + b2_3).y,
            (b2_direction * self.b2_top + b2_3).z,
        )
        b2_front = [b2_0, b2_1, b2_2, b2_3]
        b2_back = [vertex.transformed(translation) for vertex in b2_front]
        blocks.append(Mesh.from_vertices_and_faces(b2_front + b2_back, faces))

        return blocks


def report_result(label, result):
    """Print a compact robust-analysis summary."""
    bounded_directions = result.statuses.count("optimal")
    print(
        "{}: bounded={}, feasible_center={}, bounded_directions={}/{}".format(
            label,
            result.is_bounded,
            result.feasible_center,
            bounded_directions,
            len(result.directions),
        )
    )


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


def build_assembly(geometry, block_nodes, support_nodes):
    """Build an assembly from selected original block indices."""
    assembly = CRA_Assembly()
    blocks = geometry.blocks()
    for node in block_nodes:
        assembly.add_block(blocks[node].copy(cls=Block), node=node)
    assembly.set_boundary_conditions(support_nodes)
    assembly_interfaces_numpy(assembly)
    return assembly


def robust_case_results(assembly, load_node, options):
    """Run radial, primal support, and dual support analyses for one case."""
    load_dofs = [(load_node, "fx"), (load_node, "fz")]
    load_application_points = {
        load_node: rightmost_side_vertices(assembly.graph.node_attribute(load_node, "block")),
    }
    solver_options = dict(options)
    solver_options["load_application_points"] = load_application_points
    radial = rbe_robust_sample(assembly, load_dofs, **solver_options)
    primal = rbe_robust_support_primal(assembly, load_dofs, **solver_options)
    dual = rbe_robust_support_dual(assembly, load_dofs, **solver_options)
    return radial, primal, dual


def print_case_report(label, results, boundaries):
    """Print boundedness and optional governing boundary equations for one case."""
    radial, primal, dual = results
    print(label)
    report_result("  radial sampling", radial)
    report_result("  primal support", primal)
    report_result("  dual support", dual)
    if PRINT_BOUNDARY_EQUATIONS:
        print_boundary_equations(label, boundaries)


def line_value(line, point):
    """Evaluate one support line at a visible load point."""
    _, a, b, c = line
    return a * point[0] + b * point[1] - c


def edge_line_tolerance(line, first, second, relative_tolerance=1e-6):
    """Return a scale-aware tolerance for matching a polygon edge to a support line."""
    _, a, b, c = line
    scale = max(
        1.0,
        abs(c),
        abs(a * first[0] + b * first[1]),
        abs(a * second[0] + b * second[1]),
    )
    return relative_tolerance * scale


def governing_boundary_lines(result):
    """Return support halfspaces that form edges of the outer polygon."""
    polygon = result.outer_polygon
    if len(polygon) < 2:
        return []

    governing = []
    used_halfspace_indices = set()
    halfspaces = [
        (index, halfspace[0], halfspace[1], halfspace[2]) for index, halfspace in enumerate(result.halfspaces)
    ]
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        for line in halfspaces:
            tolerance = edge_line_tolerance(line, first, second)
            if abs(line_value(line, first)) <= tolerance and abs(line_value(line, second)) <= tolerance:
                if line[0] not in used_halfspace_indices:
                    label = "L{}".format(len(governing) + 1)
                    governing.append(
                        {
                            "label": label,
                            "halfspace_index": line[0],
                            "a": line[1],
                            "b": line[2],
                            "c": line[3],
                        }
                    )
                    used_halfspace_indices.add(line[0])
                break
    return governing


def format_float(value):
    """Format one coefficient for compact equation reporting."""
    if abs(value) < 5e-12:
        value = 0.0
    return "{:.6g}".format(value)


def format_boundary_equation(boundary, tolerance=1e-9):
    """Format one visible load-space boundary line as ``Fz = a * Fx + b``."""
    a = boundary["a"]
    b = boundary["b"]
    c = boundary["c"]
    if abs(b) <= tolerance:
        if abs(a) <= tolerance:
            return "{}: degenerate line".format(boundary["label"])
        return "{}: Fx = {}".format(boundary["label"], format_float(c / a))
    return "{}: Fz = {} Fx + {}".format(
        boundary["label"],
        format_float(-a / b),
        format_float(c / b),
    )


def line_box_segment(boundary, xlim, ylim, tolerance=1e-9):
    """Return the visible segment of one line clipped to a rectangular viewport."""
    a = boundary["a"]
    b = boundary["b"]
    c = boundary["c"]
    points = []

    if abs(b) > tolerance:
        for x in xlim:
            z = (c - a * x) / b
            if ylim[0] - tolerance <= z <= ylim[1] + tolerance:
                points.append((x, z))
    if abs(a) > tolerance:
        for z in ylim:
            x = (c - b * z) / a
            if xlim[0] - tolerance <= x <= xlim[1] + tolerance:
                points.append((x, z))

    unique = []
    for point in points:
        if not any(
            abs(point[0] - existing[0]) <= tolerance and abs(point[1] - existing[1]) <= tolerance
            for existing in unique
        ):
            unique.append(point)
    if len(unique) < 2:
        return None
    return unique[0], unique[1]


def boundary_label_candidates(axes, segment, index):
    """Yield candidate anchor points and display offsets for a boundary label."""
    (x0, z0), (x1, z1) = segment
    display_start = axes.transData.transform((x0, z0))
    display_end = axes.transData.transform((x1, z1))
    display_vector = [display_end[0] - display_start[0], display_end[1] - display_start[1]]
    display_length = math.hypot(display_vector[0], display_vector[1])

    if display_length <= 1e-12:
        tangent = [1.0, 0.0]
    else:
        tangent = [display_vector[0] / display_length, display_vector[1] / display_length]
    normal = [-tangent[1], tangent[0]]

    fractions = [0.2, 0.35, 0.5, 0.65, 0.8]
    fractions = fractions[index % len(fractions) :] + fractions[: index % len(fractions)]
    normal_offsets = [14, -14, 24, -24, 34, -34, 44, -44]
    tangent_offsets = [0, 12, -12, 24, -24]

    for fraction in fractions:
        anchor = [x0 + fraction * (x1 - x0), z0 + fraction * (z1 - z0)]
        for normal_offset in normal_offsets:
            for tangent_offset in tangent_offsets:
                offset = (
                    normal_offset * normal[0] + tangent_offset * tangent[0],
                    normal_offset * normal[1] + tangent_offset * tangent[1],
                )
                yield anchor, offset


def boundary_label_overlaps(annotation, placed_bboxes, renderer):
    """Return True if a rendered annotation overlaps an existing label bbox."""
    bbox = annotation.get_window_extent(renderer).expanded(1.08, 1.18)
    return any(bbox.overlaps(placed_bbox) for placed_bbox in placed_bboxes)


def annotate_boundary_lines(axes, boundaries, xlim, ylim):
    """Draw labels for boundary lines visible inside the plot viewport."""
    visible_boundaries = []
    for boundary in boundaries:
        segment = line_box_segment(boundary, xlim, ylim)
        if segment is None:
            continue
        visible_boundaries.append((boundary, segment))

    placed_bboxes = []
    for index, (boundary, segment) in enumerate(visible_boundaries):
        (x0, z0), (x1, z1) = segment
        axes.plot([x0, x1], [z0, z1], color="black", linewidth=1.2, alpha=0.75)
        placed_annotation = None
        for anchor, offset in boundary_label_candidates(axes, segment, index):
            annotation = axes.annotate(
                boundary["label"],
                xy=anchor,
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color="black",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "black", "alpha": 0.75},
                arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.7, "alpha": 0.75},
            )
            axes.figure.canvas.draw()
            renderer = axes.figure.canvas.get_renderer()
            if not boundary_label_overlaps(annotation, placed_bboxes, renderer):
                placed_bboxes.append(annotation.get_window_extent(renderer).expanded(1.08, 1.18))
                placed_annotation = annotation
                break
            annotation.remove()
        if placed_annotation is None:
            anchor, offset = next(boundary_label_candidates(axes, segment, index))
            placed_annotation = axes.annotate(
                boundary["label"],
                xy=anchor,
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color="black",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "black", "alpha": 0.75},
                arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.7, "alpha": 0.75},
            )
            axes.figure.canvas.draw()
            renderer = axes.figure.canvas.get_renderer()
            placed_bboxes.append(placed_annotation.get_window_extent(renderer).expanded(1.08, 1.18))
    return len(visible_boundaries)


def visible_boundary_equation_text(boundaries, xlim, ylim):
    """Return boundary equations for lines visible inside a viewport."""
    return "\n".join(
        format_boundary_equation(boundary) for boundary in boundaries if line_box_segment(boundary, xlim, ylim)
    )


def print_boundary_equations(label, boundaries):
    """Print all governing visible load-space boundary equations."""
    print("{} boundary lines:".format(label))
    for boundary in boundaries:
        print("  {}".format(format_boundary_equation(boundary)))


def plot_case(axes, results, boundaries, title, xlim, ylim):
    """Plot one boundary-condition case on an existing axes."""
    plot_rbe_robust_results(
        results,
        labels=["radial", "primal", "dual"],
        ax=axes,
        xlim=xlim,
        ylim=ylim,
    )
    if SHOW_BOUNDARY_LINES:
        visible_boundary_count = annotate_boundary_lines(axes, boundaries, xlim, ylim)
        if visible_boundary_count < len(boundaries):
            axes.text(
                0.02,
                0.02,
                "Some governing lines are outside this viewport; see console table.",
                transform=axes.transAxes,
                fontsize=7,
                va="bottom",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
            )
        equation_text = visible_boundary_equation_text(boundaries, xlim, ylim)
        if equation_text:
            axes.text(
                0.02,
                0.98,
                equation_text,
                transform=axes.transAxes,
                fontsize=7,
                va="top",
                family="monospace",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
            )
    axes.set_xlabel("Block 2 load Fx")
    axes.set_ylabel("Block 2 load Fz")
    axes.set_title(title)
    legend = axes.legend(loc="upper right", fontsize=7)
    legend.set_draggable(True)


if __name__ == "__main__":
    geometry = Arch(
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
    options = {
        "mu": 0.8,
        "density": 1.0,
        "application_force_bound": 1e3,
        "num_directions": 72,
    }
    print("hidden hand-force component bound: 1000000.0 (placeholder)")

    b1_fixed_assembly = build_assembly(geometry, block_nodes=[1, 2], support_nodes=[1])
    b1_fixed_results = robust_case_results(b1_fixed_assembly, load_node=2, options=options)
    b1_fixed_boundaries = governing_boundary_lines(b1_fixed_results[2])
    print_case_report("b1 fixed / block 0 removed", b1_fixed_results, b1_fixed_boundaries)

    b0_fixed_assembly = build_assembly(geometry, block_nodes=[0, 1, 2], support_nodes=[0])
    b0_fixed_results = robust_case_results(b0_fixed_assembly, load_node=2, options=options)
    b0_fixed_boundaries = governing_boundary_lines(b0_fixed_results[2])
    print_case_report("b0 fixed / full assembly", b0_fixed_results, b0_fixed_boundaries)

    figure, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    figure.set_size_inches(14, 6)
    plot_case(
        axes[0],
        b1_fixed_results,
        b1_fixed_boundaries,
        "Block 1 fixed, block 0 removed",
        XLIM,
        YLIM,
    )
    plot_case(
        axes[1],
        b0_fixed_results,
        b0_fixed_boundaries,
        "Block 0 fixed, full three-block assembly",
        XLIM,
        YLIM,
    )
    figure.suptitle("Boundary-condition comparison for block 2 safe-load regions")
    figure.tight_layout()
    if plt.get_backend().lower() != "agg":
        plt.show()
