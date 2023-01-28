import numpy as np
import pandas as pd
import sys, os
import pickle

sys.path.append('safegraph/utils/')
from mobility_processor import *

# ct = CensusTractMobility()
checkpoint_version = 9
checkpoint_path = f'safegraph/compute_graph_checkpoints/checkpoint_{checkpoint_version}.pkl'
with open(checkpoint_path, 'rb') as f:
    ct = pickle.load(f)


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

    ct.save_pickle(f'safegraph/compute_graph_checkpoints/checkpoint_{i}.pkl')
    print('------ SAVED FILED !!! -------')
    print()



# with open('test.pkl', 'rb') as f:
#     df = pickle.load(f)


