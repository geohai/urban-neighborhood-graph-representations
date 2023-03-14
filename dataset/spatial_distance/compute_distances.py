
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('../safegraph/')
from mobility_processor import *
#--------------- SETUP --------------#
graph_obj_path = '../safegraph/graph_checkpoints/nyc_metro/checkpoint_norm.pkl'
with open(graph_obj_path, 'rb') as f:
    g = pickle.load(f) # CensusTractMobility object

# graph properties
num_nodes = g.num_nodes
print(f'Number of regions: {num_nodes}')
node_embedding_length = 200

# Read census tract shapefiles: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2021&layergroup=Census+Tracts
df = g.get_geopandas_tracts()
df = df.to_crs("EPSG:32643") # https://www.spatialreference.org/ref/epsg/wgs-84-utm-zone-43n/ -- units in meters
shapes = df[['GEOID', 'geometry']]

shapes['centroid'] = shapes['geometry'].apply(lambda x: x.centroid)

print(shapes.head())

# ---- COMPUTE EDGE WEIGHT MATRIX  ---- #
# ------- array load -------
dist_edge_matrix = np.zeros((num_nodes, num_nodes))

# --------dictionary load------
# distance_dict = {}
# with open('distance.pkl', 'rb') as f:
#     distance_dict = pickle.load(f)


geoid_list = shapes.GEOID.unique()
node_idx_map = {}
for row in range(len(geoid_list)):
    i = geoid_list[row]
    node_idx_map[i] = row
#     if (i in distance_dict.keys()) and (len(distance_dict[i].keys()) == 8231):
#         continue
#     print(i)
#     distance_dict[i] = {}
    for col in range(row, len(geoid_list)):
        j = geoid_list[col]
#         dist_edge_matrix[i][j] = shapes['centroid'].iloc[i].distance(shapes['centroid'].iloc[j])
        dist_edge_matrix[row][col] = shapes.loc[shapes.GEOID == i, 'centroid'].iloc[0].distance(shapes.loc[shapes.GEOID == j, 'centroid'].iloc[0])
    with open('distance_array.pkl', 'wb') as f:
        pickle.dump(dist_edge_matrix, f)
    print(f'saving checkpoint...{row} out of {len(geoid_list)}')
    if row % 200 == 0:
        with open(f'distance_array_{row}.pkl', 'wb') as f:
            pickle.dump(dist_edge_matrix, f)


with open('distance_array.pkl', 'wb') as f:
    pickle.dump(dist_edge_matrix, f)
with open('node_idx_map.pkl', 'wb') as f:
    pickle.dump(node_idx_map, f)

idx_node_map = dict(zip(d.values(), d.keys()))
with open('idx_node_map.pkl', 'wb') as f:
    pickle.dump(idx_node_map, f)
    
# mob_edge_matrix = g.get_edge_mat() # 2D array of mobility weights of nxn
# np.save('data_files/final_edge_data/mobility.npy', mob_edge_matrix)
# print('saved')


# ------ Played around with Pytorch Geometric below, ended up not using it ------- #

# from torch_geometric.data import Data
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
