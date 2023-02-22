import numpy as np
import pandas as pd
import sys, os
from os.path import exists
from census import Census

sys.path.append('safegraph/utils/')
from mobility_processor import *
from helper_funcs import *

with open('api_keys/census_key.txt', 'r') as f:
    key = f.readline()
c = Census(key)

dirr= 'safegraph/graph_checkpoints/nyc_metro/'
createDir(dirr)


ct = CensusTractMobility(tract_data_dir='Tracts/nyc_metro_boundaries/nyc_metro_boundaries.shp')
# checkpoint_version = -1
# checkpoint_path = f'safegraph/compute_graph_checkpoints/denver/checkpoint_{checkpoint_version}.pkl'
# with open(checkpoint_path, 'rb') as f:
#     ct = pickle.load(f)


for i, fn in enumerate(os.listdir('safegraph/data/nyc_metro')): # one file for each month
    if 'csv' not in fn:
        continue
    if i <= checkpoint_version:
        continue

    path = 'safegraph/data/nyc_metro/' + fn
    d = pd.read_csv(path, dtype={'GEOID': str})
    d = d.drop(index=d[d['visitor_home_aggregation'].isna()].index)

    ct.add_weighted_edge_matrix(d)
    mat = ct.get_edge_mat()
    print(mat.sum())

    ct.save_pickle(f'{dirr}checkpoint_{i}.pkl')
    print('------ SAVED FILED !!! -------')
    print()

