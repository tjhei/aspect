import numpy as np
from landlab import RasterModelGrid


grid = None
cumulative_deposition = 0.0


def initialize(comm_handle):
    pass


def finalize():
    pass


def set_mesh_information(information):
    global grid
    if grid is None:
        # A fixed Landlab mesh covering the unit square, independent of the
        # ASPECT boundary mesh. Boundary nodes have zero dual-cell area.
        grid = RasterModelGrid((7, 9), xy_spacing=(1.0 / 8.0, 1.0 / 6.0))


def get_grid_x(dim):
    return grid.x_of_node


def get_grid_y(dim):
    return grid.y_of_node


def get_grid_weights(dim):
    # Do not normalize: only core nodes own cells, so their areas sum to
    # less than the full unit square. Preserve the actual deposited volume.
    return grid.cell_area_at_node


def get_initial_topography(dim):
    return np.zeros(grid.number_of_nodes)


def update_until(solution, auxiliary):
    global cumulative_deposition
    dt = auxiliary["ASPECT timestep size"]
    step = int(auxiliary["ASPECT timestep number"])
    x = grid.x_of_node[grid.core_nodes]
    y = grid.y_of_node[grid.core_nodes]
    # Prescribe spatially and temporally varying deposition on core nodes.
    depo_rate = (1.0 + 0.1 * step) * (
        0.2 + 0.1 * x + 0.05 * y
        + 0.1 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    )
    deposition = np.zeros(grid.number_of_nodes)
    deposition[grid.core_nodes] = depo_rate * dt
    total_deposition = np.sum(
        depo_rate * dt * grid.cell_area_at_node[grid.core_nodes]
    )
    cumulative_deposition += total_deposition

    # Supply an independent Python-side total to the C++ postprocessor. The
    # postprocessor checks both this step and the cumulative volume change.
    filename = auxiliary["ASPECT output directory"] + "landlab_deposition.txt"
    with open(filename, "w") as output:
        output.write(
            f"{step} {total_deposition:.17e} {cumulative_deposition:.17e}\n"
        )
    return deposition
