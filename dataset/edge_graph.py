
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import itertools
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle

#--------------- SETUP --------------#
# read node list
nodes = pd.read_csv('data_files/node_list.csv')
node_list = np.array(nodes['RegionID'])

# graph properties
num_nodes = len(node_list)
node_embedding_length = 200
num_sample = 50

# Read census tract shapefiles: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2021&layergroup=Census+Tracts
path = "data_files/tl_2021_08_tract/tl_2021_08_tract.shp"
df = gpd.read_file(path)
df = df.to_crs("EPSG:32643") # https://www.spatialreference.org/ref/epsg/wgs-84-utm-zone-43n/ -- units in meters
shapes = df['geometry']

# dummy data for now
df = df.iloc[0:num_nodes]
# df.plot()
# df.boundary.plot()
# plt.show()

# embedding_table = torch.nn.Embedding(num_nodes, node_embedding_length,  max_norm=True)
nodes = np.array([i for i in range(0, num_nodes)])

# generate tuples of every combo between nodes
edge_idx_matrix = itertools.product(nodes, nodes) # create edges between all nodes (directed)
edge_idx_matrix = list(edge_idx_matrix)

# remove self loops
for i in range(0, num_nodes):
    edge_idx_matrix.remove((i, i))

# create a 2d matrix of [num_nodes, num_nodes-1] indexed by node idx which contains the idx of outgoing edges from that node in the edge_idx_matrix
node_idx_in_edge = [[] for i in nodes]
for node_idx in nodes:
    for i, edge in enumerate(edge_idx_matrix):
        if node_idx == edge[0]:
            node_idx_in_edge[node_idx].append(i)


node_idx_in_edge = torch.tensor(np.array(node_idx_in_edge))

# weights
# spatial distance with geopandas
edge_idx_weights = []
for pair in edge_idx_matrix:
    edge_dist = shapes[pair[0]].distance(shapes[pair[1]])
    edge_idx_weights.append(edge_dist)

# TODO: get mobility values 



# create graph object
data = Data(neighborhood=node_list.reshape(-1, 1), num_nodes=num_nodes, edge_index=torch.tensor(edge_idx_matrix), edge_attr=torch.tensor(edge_idx_weights), node_idx_in_edge=node_idx_in_edge) 
print(data)


# now save graph object
with open('data_files/edge_graph_obj', 'wb') as f:
    pickle.dump(data, f)
