import numpy as np
import pandas as pd
import sys, os
import pickle

sys.path.append('safegraph/utils/')
from mobility_processor import *

# save_path = tract_data_dir='data_files/Tracts/grandjunction/tl_2016_08077_edges.shp'
ct = CensusTractMobility(tract_data_dir='data_files/Tracts/denver/census_tracts_2010.shp')
checkpoint_version = -1
# checkpoint_path = f'safegraph/compute_graph_checkpoints/denver/checkpoint_{checkpoint_version}.pkl'
# with open(checkpoint_path, 'rb') as f:
#     ct = pickle.load(f)


ct.add_census_tracts()

for i, fn in enumerate(os.listdir('safegraph/colorado')): # one file for each month
    if i <= checkpoint_version:
        continue

    path = 'safegraph/colorado/' + fn
    d = pd.read_csv(path)
    d = d.drop(index=d[d['visitor_home_aggregation'].isna()].index)
    print(d.shape)


    ct.add_weighted_edge_matrix(d)
    mat = ct.get_edge_mat()
    print(mat.sum())

    ct.save_pickle(f'safegraph/compute_graph_checkpoints/grandjunction_denver/checkpoint_{i}.pkl')
    print('------ SAVED FILED !!! -------')
    print()



# with open('test.pkl', 'rb') as f:
#     df = pickle.load(f)


