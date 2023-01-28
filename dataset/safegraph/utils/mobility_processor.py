import os
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.path as mplPath
import geopandas as gpd
import json
import pickle

class CensusTractMobility(object):
    def __init__(self, tract_data_dir='data_files/Tracts/tl_2021_08_tract.shp'):
        """
        Loads the tract data into geopandas.
        """
        self.tract_data = None
        self.num_nodes = None
        if tract_data_dir is not None:
            if exists(tract_data_dir):
                df = gpd.read_file(tract_data_dir)
                df = df[['GEOID', 'geometry']]

                self.node_idx_map = dict(zip(df['GEOID'], df.index))
                self.idx_node_map = dict(zip(df.index, df['GEOID']))
                self.tract_data =  df #.to_crs("EPSG:32643")
                self.num_nodes = df.shape[0] # each row is a unique combo of state/county/tract code 
                print('loaded data')

                self.count_checker = 0

        self.edge_mat = np.zeros((self.num_nodes, self.num_nodes))
        pass

    def add_row_weights_to_graph(self, row):
            if pd.isna(row['visitor_home_aggregation']):
                return
                     
            # extract dict
            d = json.loads(row['visitor_home_aggregation'])
            # check if source of visit is in our regions
            for src, weight in d.items():
                if src in self.node_idx_map.keys():
                    self.count_checker += weight

                    src_idx = self.node_idx_map[src]

                    try:
                        dest_idx = self.node_idx_map[row['GEOID']]
                        self.edge_mat[src_idx][dest_idx] += weight
                    except:
                        print('except')
                        pass

                # no normalization right now
                

    def add_weighted_edge_matrix(self, df):
        """
        """
        # get the tract that the POIs are located within
        point = gpd.points_from_xy(x=df['longitude'], y=df['latitude'])
        points = gpd.GeoSeries(point, index=df.index)
       
        # df['GEOID'] = points.apply(lambda p: self.get_tract_num(x=p.x, y=p.y))
        df['GEOID'] = points.apply(lambda p: self.get_tract_num(p))

        # now iterate thru the df and sum weights
        df.apply(self.add_row_weights_to_graph, axis=1)

    def get_tract_num(self, point):
        tract_info = self.tract_data.loc[self.tract_data['geometry'].contains(point), ['GEOID']]
        tract_info = tract_info.iat[0, 0]
            
        return tract_info
    
    def get_edge_mat(self):
        print(f'Summed weights: {self.count_checker}')
        return self.edge_mat
    
    def get_node_idx(self):
        return self.node_idx_map
    
    def get_idx_node(self):
        return self.idx_node_map
    
    def get_geopandas_tracts(self):
        return self.tract_data
    
    def save_pickle(self, path):
        with open(path, 'wb') as outf:
            pickle.dump(self, outf)

if __name__ == '__main__':
    ct = CensusTractMobility()
    # ct.save_tract_data(boundary_file_dir='./Tracts')

    d = pd.read_csv('safegraph/colorado/weekly_patterns-0.csv')[0:20]
    ct.add_weighted_edge_matrix(d)
    mat = ct.get_edge_mat()
    print(mat)
    print(mat.sum())

    ct.save_pickle('test.pkl')


    with open('test.pkl', 'rb') as f:
        df = pickle.load(f)

    print(df)
    print(df.get_edge_mat().sum())

    



