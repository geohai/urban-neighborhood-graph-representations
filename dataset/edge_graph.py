
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import itertools
import geopandas as gpd
import matplotlib.pyplot as plt

#--------------- SETUP --------------#
# read node list
nodes = pd.read_csv('data_files/node_list.csv')
node_list = np.array(nodes['RegionID'])

# POI and SV do not require an actual graph structure
num_nodes = len(node_list)
node_embedding_length = 200
num_sample = 50


# embedding_table = torch.nn.Embedding(num_nodes, node_embedding_length,  max_norm=True)
nodes = np.array([i for i in range(0, num_nodes)])

edge_idx_matrix = itertools.product([i for i in range(0, num_nodes)], [i for i in range(0, num_nodes)]) # create edges between all nodes
edge_idx_matrix = list(edge_idx_matrix)

# weights
# compute distance between nodes
# for now randomly generate distances
edge_idx_weights = torch.randn(len(edge_idx_matrix), 2) # 1 dim for spatial distance, 1 dim for mobility value

# census tract shapefiles: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2021&layergroup=Census+Tracts
path = "data_files/tl_2021_08_tract/tl_2021_08_tract.shp"
df = gpd.read_file(path)
df = df.to_crs("EPSG:4326")

plt.figure()
df.plot()
plt.figure()
df.boundary.plot()
plt.show()


data = Data(neighborhood=node_list.reshape(-1, 1), num_nodes=num_nodes, edge_index=edge_idx_matrix, edge_attr=edge_idx_weights) 
print(data)
print(data.num_nodes)


# data.train_idx = torch.tensor([...], dtype=torch.long)
# data.test_mask = torch.tensor([...], dtype=torch.bool)





# #-----------------------------------------
# # edges and values are determined by
# num_node_attr = 10
# num_nodes = 5
# edge_mob_feat_len = 1
# edge_dist_feat_len = 1



# # remove self loops
# for i in range(0, num_nodes):
#     edge_idx_matrix.remove((i, i))

# edge_id_mob_attr = np.random.randn(len(edge_idx_matrix), edge_mob_feat_len) # edge_idx corresponds to row idx, feature vector

# # print(edge_id_mob_attr)
# data = HeteroData()
# data['neighborhood'] = torch.randn(num_nodes, num_node_attr)
# data['neighborhood', 'mobility-to', 'neighborhood'].edge_index = edge_idx_matrix # mapping of node to node edges
# data['neighborhood', 'mobility-to', 'neighborhood'].edge_attr = edge_id_mob_attr
# # data['neighborhood', 'distance', 'neighborhood'].edge_index = edge_idx_matrix # mapping of node to node edges
# # data['neighborhood', 'distance', 'neighborhood'].edge_attr = edge_id_mob_attr

# print(data.num_edge_features)
# print(data.stores)

# print(data.to_dict())

# # print(data.num_nodes)
# # x = torch.randn(num_node_attr, num_)
# # print(x)

# # import networkx as nx
# # G = nx.MultiDiGraph()
# # import matplotlib.pyplot as plt
# # plt.figure()
# # nx.draw(G, pos=nx.spring_layout(G))
# # nx.draw_networkx_labels(G, pos=nx.spring_layout(G))
# # plt.show()