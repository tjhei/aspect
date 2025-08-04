"""
This is an experiment to use pymetis and mpi4py for landlab parallel

Test to run on
SimpleSubmarineDiffuser
https://landlab.csdms.io/tutorials/marine_sediment_transport/simple_submarine_diffuser_tutorial.html

ver3: identify boundary nodes for each subdomain, run model with multiple times

To run the program:
mpiexec -np 5 python mpi_landlab3.py

"""
import os
import numpy as np
import pymetis

# import matplotlib
# matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

from landlab import HexModelGrid, VoronoiDelaunayGrid
from landlab.components import SimpleSubmarineDiffuser


## step 0: set up parallel
from mpi4py import MPI
from collections import defaultdict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure number of partitions matches the MPI processes
num_partitions = size
assert size == num_partitions, "Number of MPI processes must match the number of partitions!"

if rank == 0:
    output_dir = os.path.join(os.getcwd(),'output')
    os.makedirs(output_dir, exist_ok=True)

    ## step 1: define hex model grid and assign z values
    mg = HexModelGrid((17, 17), spacing=1, node_layout='rect')
    z = mg.add_zeros("topographic__elevation", at="node")
    cum_depo = mg.add_zeros("total_deposit__thickness", at="node")


    midpoint = 8
    dx = np.abs(mg.x_of_node - midpoint)
    dy = np.abs(mg.y_of_node - midpoint)
    ds = np.sqrt(dx * dx + dy * dy)
    z[:] = (midpoint - ds) - 3.0
    z[z < -3.0] = -3.0
    z0 = z.copy()

    # identify boundary nodes
    boundary_nodes = mg.boundary_nodes

    # plot z 2D and 1D
    mg.imshow(z, cmap="coolwarm", vmin=-3)
    plt.title("Elevation on Global Grid")
    plt.savefig(os.path.join(output_dir, "dem_hex.png"))

    plt.clf()
    plt.plot(mg.x_of_node, z, ".")
    plt.plot([0, 17], [0, 0], "b:")
    plt.grid(True)
    plt.xlabel("Distance (m)")
    plt.ylabel("Elevation (m)")
    plt.savefig(os.path.join(output_dir,"dem_hex_slice.png"))

    # plot total_deposit__thickness
    plt.clf()
    mg.imshow("total_deposit__thickness", cmap="coolwarm", vmin=-1,vmax=1)
    plt.title("Total deposit thickness initiation (m)")
    plt.savefig(os.path.join(output_dir,"total_deposit_init.png"))


    ## step2: grid partition
    adjacency_list = []

    # create adjacency list for corners
    for node_id in mg.nodes.flat:
        adjacent_nodes = [n for n in mg.adjacent_nodes_at_node[node_id] if n != -1]
        adjacency_list.append(np.array(adjacent_nodes))
        # print(node_id, mg.adjacent_nodes_at_node[node_id], adjacent_nodes)

    # Partition the grid using pymetis
    n_cuts, part_labels = pymetis.part_graph(num_partitions, adjacency=adjacency_list)

    # Convert partition labels to a NumPy array
    partition_array = np.array(part_labels)
    print(partition_array)

    # visualization
    fig, ax = plt.subplots(figsize=[16, 14])
    ax.scatter(mg.node_x, mg.node_y, c=partition_array, cmap='viridis')
    ax.set_title('grid partition based on nodes')
    for node_id in mg.nodes.flat:
        ax.annotate(f"{node_id}/par{partition_array[node_id]}",
                    (mg.node_x[node_id], mg.node_y[node_id]),
                    color='black', fontsize=8, ha='center', va='top')
    fig.savefig(os.path.join(output_dir,'global_grid_partition.png'))

    print(f"grid partition at rank {rank}")
else:
    part_labels = None
    mg = None
    adjacency_list = None
    boundary_nodes = None
    output_dir = None

# broadcast results to other ranks
part_labels = comm.bcast(part_labels, root=0)

adjacency_list = comm.bcast(adjacency_list, root=0)
boundary_nodes = comm.bcast(boundary_nodes, root=0)
output_dir = comm.bcast(output_dir)

## step3: identify ghost nodes
# local grid nodes
local_nodes = [node for node, part in enumerate(part_labels) if part == rank]


# identify ghost nodes for sending and receiving data
send_to = defaultdict(set)
recv_from = defaultdict(set)

for node in local_nodes:
    for neighbor in adjacency_list[node]:
        neighbor_part = part_labels[neighbor]
        if neighbor_part != rank:
            print(neighbor_part,node)
            send_to[neighbor_part].add(node)
            recv_from[neighbor_part].add(neighbor)

# loop for multiple time steps
for time_step in range(0,20):
    mg = comm.bcast(mg, root=0)
    ## step4: send and receive data for ghost nodes
    for pid, nodes_to_send in send_to.items():
        # Convert to sorted list
        nodes_to_send = sorted(nodes_to_send)
        elev_to_send = mg.at_node["topographic__elevation"][list(nodes_to_send)]
        comm.send((nodes_to_send, elev_to_send), dest=pid, tag=rank)
        #print(f"Rank {rank} sent data to {pid} for nodes: {nodes_to_send}")

    ghost_nodes_values= {}
    for pid, ghost_nodes in recv_from.items():
        nodes, elev_values = comm.recv(source=pid, tag=pid)
        ghost_nodes_values.update(dict(zip(nodes, elev_values)))
        #if rank==1:
            #print(f"Rank {rank} received ghost data from {pid} for nodes {nodes}")
            #print(ghost_nodes_values)


    ## step5: define local grid for simulation
    # define voronoi grid
    ghost_nodes = list(ghost_nodes_values.keys())

    vmg_global_ind = sorted(local_nodes + ghost_nodes)
    x = mg.node_x[vmg_global_ind]
    y = mg.node_y[vmg_global_ind]
    elev_local = mg.at_node["topographic__elevation"][vmg_global_ind].copy()
    cum_depo_local = mg.at_node["total_deposit__thickness"][vmg_global_ind].copy()

    # if rank==4:
    #     print(f"loop {time_step}")
    #     print(elev_local)
    local_vmg = VoronoiDelaunayGrid(x, y)
    local_vmg.add_field("topographic__elevation", elev_local, at="node")
    local_cum_depo = local_vmg.add_field("total_deposit__thickness", cum_depo_local, at="node")
    local_z0 = elev_local.copy()

    local_nodes_as_boundary = [node for node in local_nodes if node in boundary_nodes]
    local_boundary_nodes_ind = [vmg_global_ind.index(val) for val in ghost_nodes + local_nodes_as_boundary]
    local_vmg.status_at_node[local_boundary_nodes_ind] = local_vmg.BC_NODE_IS_FIXED_VALUE
    # print(local_nodes_as_boundary)
    # print(local_boundary_nodes_ind)
    # print(vmg_global_ind)
    # print(local_vmg.status_at_node[local_boundary_nodes_ind])

    # plot subgrid for each rank
    fig, ax = plt.subplots(figsize=[18, 14])
    sc = ax.scatter(local_vmg.node_x, local_vmg.node_y,
               c=local_vmg.at_node["topographic__elevation"], cmap="coolwarm", vmin=-3)
    ax.set_title(f'subgrid nodes rank={rank}')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Elevation (m)')
    fig.savefig(os.path.join(output_dir,f'subgrid_for_rank{rank}.png'))


    ## step 6: run simulation
    # define model
    ssd = SimpleSubmarineDiffuser(
        local_vmg, sea_level=0.0, wave_base=1.0, shallow_water_diffusivity=1.0
    )

    # run one step
    ssd.run_one_step(0.2)
    local_cum_depo += local_vmg.at_node["sediment_deposit__thickness"]

    # plot results
    elev_diff = (local_z0 - elev_local) * 100
    fig, ax = plt.subplots(figsize=[16, 14])
    sc = ax.scatter(local_vmg.node_x, local_vmg.node_y,c=elev_diff, cmap="coolwarm", vmin=-6,vmax=6)
    ax.set_title(f'Changes of elevation rank={rank}')
    for node_id in local_boundary_nodes_ind:
        ax.annotate(f"B",
                    (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
                    color='blue', fontsize=12, ha='center', va='top')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Elevation Change (cm)')
    fig.savefig(os.path.join(output_dir,f'elev_diff_for_rank{rank}.png'))

    # close all plots
    plt.close('all')

    ## step 7 gather all updates to rank 0
    # Create local update data (only for owned nodes, not ghost)
    local_updates = []
    for node in local_nodes:
        vmg_local_ind = vmg_global_ind.index(node)
        local_updates.append( (node,
                                elev_local[vmg_local_ind],
                                local_cum_depo[vmg_local_ind]) )


    all_updates = comm.gather(local_updates, root=0)

    if rank == 0:
        # Flatten list of updates from all ranks
        flat_updates = [item for sublist in all_updates for item in sublist]
        for node_id, elev, cum_depo in flat_updates:
            mg.at_node["topographic__elevation"][node_id] = elev
            mg.at_node["total_deposit__thickness"][node_id] = cum_depo

        #print(z0 == mg.at_node["topographic__elevation"])

        # plot results
        # diff_z and total_deposit_result should be the same
        plt.clf()
        mg.imshow("total_deposit__thickness", cmap="coolwarm")
        plt.title("Total deposit thickness results (m)")
        plt.savefig(os.path.join(output_dir,"total_deposit_result.png"))

        plt.clf()
        diff_z = mg.at_node["topographic__elevation"] - z0
        mg.imshow(diff_z, cmap="coolwarm")
        plt.title("Changes of elevation (m)")
        plt.savefig(os.path.join(output_dir,"dem_diff_hex_model_result.png"))

        plt.clf()
        mg.imshow("topographic__elevation", cmap="coolwarm")
        plt.title("Topographic elevation result (m)")
        plt.savefig(os.path.join(output_dir,"dem_hex_model_result.png"))
        plt.close('all')

        print(f"loop {time_step}")
        print(diff_z.max(), diff_z.min())
        print(mg.at_node["total_deposit__thickness"].max(),
              mg.at_node["total_deposit__thickness"].min())
