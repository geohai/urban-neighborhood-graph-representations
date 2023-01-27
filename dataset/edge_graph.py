
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import itertools
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('safegraph/utils/')
from mobility_processor import *

#--------------- SETUP --------------#
graph_obj_path = 'safegraph/compute_graph_checkpoints/checkpoint_0.pkl'
with open(graph_obj_path, 'rb') as f:
    g = pickle.load(f) # CensusTractMobility object

node_list = g.get_node_idx() # dictionary = (regionid: idx)

# graph properties
num_nodes = len(node_list)
node_embedding_length = 200
num_sample = 50

# Read census tract shapefiles: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2021&layergroup=Census+Tracts
df = g.get_geopandas_tracts()
df = df.to_crs("EPSG:32643") # https://www.spatialreference.org/ref/epsg/wgs-84-utm-zone-43n/ -- units in meters
shapes = df['geometry']

# dummy data for now
# df = df.iloc[0:num_nodes]
# df.plot()
# df.boundary.plot()
# plt.show()

# ---- COMPUTE EDGE WEIGHT MATRIX  ---- #
# weights
# spatial distance with geopandas
dist_edge_matrix = np.zeros((num_nodes, num_nodes))

# directed, so we only need to compute one side of the diagonal
for i in shapes.index:
    for j in range(0, shapes.shape[0]):
        dist_edge_matrix[i][j] = shapes.iloc[i].distance(shapes.iloc[j])

# now mirror it along the diagonal
cp = dist_edge_matrix.T
for i in shapes.index:
    for j in range(i, shapes.shape[0]):
        if dist_edge_matrix[i][j] == 0:
            dist_edge_matrix[i][j] == cp[i][j]
        else:
            print('not empty')

# TODO: save distance and mobility values 
np.save('data_files/final_edge_data/distance.npy', dist_edge_matrix)

mob_edge_matrix = g.get_edge_mat() # 2D array of mobility weights of nxn
np.save('data_files/final_edge_data/mobility.npy', mob_edge_matrix)



# ------ Played around with Pytorch Geometric below, ended up not using it ------- #

# edge_idx_matrix = itertools.product(nodes, nodes) # create edges between all nodes (directed)
# edge_idx_matrix = list(edge_idx_matrix)

# # remove self loops
# for i in range(0, num_nodes):
#     edge_idx_matrix.remove((i, i))

# # create a 2d matrix of [num_nodes, num_nodes-1] indexed by node idx which contains the idx of outgoing edges from that node in the edge_idx_matrix
# node_idx_in_edge = [[] for i in nodes]
# for node_idx in nodes:
#     for i, edge in enumerate(edge_idx_matrix):
#         if node_idx == edge[0]:
#             node_idx_in_edge[node_idx].append(i)


# node_idx_in_edge = torch.tensor(np.array(node_idx_in_edge))


# # create graph object
# data = Data(neighborhood=node_list.reshape(-1, 1), num_nodes=num_nodes, edge_index=torch.tensor(edge_idx_matrix), edge_attr=torch.tensor(edge_idx_weights), node_idx_in_edge=node_idx_in_edge) 
# print(data)


# # now save graph object
# with open('data_files/edge_graph_obj', 'wb') as f:
#     pickle.dump(data, f)
