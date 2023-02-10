import numpy as np
import pandas as pd
import sys, os
from os.path import exists
from census import Census

sys.path.append('safegraph/utils/')
from mobility_processor import *
from helper_funcs import *

with open('../evaluation/census_key.txt', 'r') as f:
    key = f.readline()
c = Census(key)

dir = 'safegraph/graph_checkpoints/nyc_metro/'
createDir(dir)

def geoid_to_cty_fps(geoid):
    if len(geoid) != 11:
        print('GEOID not correct length.')
        return
    countyfps = geoid[2:5]
    return countyfps

def geoid_to_st_fps(geoid):
    if len(geoid) != 11:
        print('GEOID not correct length.')
        return
    statefps = geoid[0:2]
    return statefps

# normalize by population of area
def get_population(statefps, countyfps, var='B01001_001E', varname='total_pop'):
    
    census = c.acs5.state_county_tract(fields = var,
                                        state_fips = statefps,
                                        county_fips = countyfps,
                                        tract = '*',
                                        year = 2019)
    df = pd.DataFrame(census)
    df.rename(columns={var: varname}, inplace=True)
    return df


# save_path = tract_data_dir='data_files/Tracts/grandjunction/tl_2016_08077_edges.shp'
ct = CensusTractMobility(tract_data_dir='Tracts/nyc_metro_boundaries/nyc_metro_boundaries.shp')
checkpoint_version = -1
# checkpoint_path = f'safegraph/compute_graph_checkpoints/denver/checkpoint_{checkpoint_version}.pkl'
# with open(checkpoint_path, 'rb') as f:
#     ct = pickle.load(f)

# get population of each tract
df = pd.DataFrame()
pop_file_path = 'safegraph/data/nyc_metro/population.pkl'
if not exists(pop_file_path):
    df['state_fips'] = ct.tract_data.GEOID.apply(geoid_to_st_fps)
    df['county_fips'] = ct.tract_data.GEOID.apply(geoid_to_cty_fps)
    df.drop_duplicates(inplace=True)

    pop = pd.DataFrame()
    for i in range(0, df.shape[0]):
        x = df.iloc[i]
        pop = pd.concat([pop, get_population(x['state_fips'], x['county_fips'])])

    pop['GEOID'] = pop['state'] + pop['county'] + pop['tract']
    pop_dict = dict(zip(pop['GEOID'], pop['total_pop']))

    f = open(pop_file_path, 'wb')
    pickle.dump(pop_dict, f)
    f.close()
else:
    with open(pop_file_path, 'rb') as f:
        pop_dict = pickle.load(f)

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

    ct.save_pickle(f'{dir}checkpoint_{i}.pkl')
    print('------ SAVED FILED !!! -------')
    print()

