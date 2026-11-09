# A fixed Cartesian surface supplied through the Landlab coupling API.
# Manufactured nodal weights integrate the Q2 trace on the reference surface.
import numpy as np


def initialize(comm_handle):
    pass


def finalize():
    pass


def set_mesh_information(information):
    pass


def get_grid_x(dim):
    return np.repeat(np.linspace(0.0, 1.0, 5), 5)


def get_grid_y(dim):
    return np.tile(np.linspace(0.0, 1.0, 5), 5)


def get_grid_weights(dim):
    weights = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0
    return np.outer(weights, weights).ravel()


def get_initial_topography(dim):
    return np.full(25, 0.01)


def update_until(solution, auxiliary):
    return np.full(25, 0.1 * auxiliary["ASPECT timestep size"])
