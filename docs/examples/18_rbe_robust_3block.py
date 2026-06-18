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


if __name__ == "__main__":
    geometry = Arch(
        b0_width=0.5,
        b1_height=0.5,
        b1_base=0.5,
        b2_base=0.4,
        b2_top=0.7,
        alpha=math.pi / 2,
        beta=2 * math.pi / 3,
        gamma=2 * math.pi / 3,
        thickness=1,
    )
    assembly = geometry.assembly()
    assembly_interfaces_numpy(assembly)

    load_dofs = [(2, "fx"), (2, "fz")]
    load_application_points = {
        2: rightmost_side_vertices(assembly.graph.node_attribute(2, "block")),
    }
    options = {
        "mu": 0.8,
        "density": 1.0,
        "load_application_points": load_application_points,
        "application_force_bound": 1e6,
        "num_directions": 72,
    }
    radial = rbe_robust_sample(assembly, load_dofs, **options)
    primal = rbe_robust_support_primal(assembly, load_dofs, **options)
    dual = rbe_robust_support_dual(assembly, load_dofs, **options)

    report_result("radial sampling", radial)
    report_result("primal support", primal)
    report_result("dual support", dual)
    print("hidden hand-force component bound: 1000000.0 (placeholder)")

    figure, axes = plot_rbe_robust_results(
        [radial, primal, dual],
        labels=["radial", "primal", "dual"],
        xlim=(-1.0, 0.1),
        ylim=(-0.5, 1.0),
    )
    figure.set_size_inches(10, 6)
    axes.set_xlabel("Block 2 load Fx")
    axes.set_ylabel("Block 2 load Fz")
    axes.set_title("Three-block robust RBE safe-load regions with four-point grasp")
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    figure.tight_layout()
    if plt.get_backend().lower() != "agg":
        plt.show()
