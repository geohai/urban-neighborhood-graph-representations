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
                df.rename(columns={'GEOID_TRAC': 'GEOID'}, inplace=True)  
                print(df.head())
                df = df[['GEOID', 'geometry']] 

                self.node_idx_map = dict(zip(df['GEOID'], df.index))
                self.idx_node_map = dict(zip(df.index, df['GEOID']))
                self.tract_data =  df #.to_crs("EPSG:32643")
                self.num_nodes = df.shape[0] # each row is a unique combo of state/county/tract code 
                print('loaded data')

        self.count_checker = 0
        self.edge_mat = np.zeros((self.num_nodes, self.num_nodes))


    def add_census_tracts(self, tract_data_dir='data_files/Tracts/tl_2021_08_tract.shp'):
        mesa_county = [8107000800, 8107000700, 8107000500, 8107000200, 8107000400, 8107000600, 8107000300, 8107000100]
        co_springs = [8041003801, 8041005111, 8041007101, 8041007201, 8041006901,8041006301, 8041006302, 8041005110, 8041006801, 8041006802, 8041004601, 8041004511, 8041004602, 8041007601, 8041007602, 8041007202, 8041003802, 8041003307, 8041003400, 8041006700, 8041003701, 8041004100, 8041004200, 
        8041004508, 8041004507, 8041004506, 8041004300, 8041005000, 8041006200, 8041006100,8041006000, 8041005900, 8041005800, 8041005700, 8041005602, 8041004703, 8041004009,8041005202, 8041003905, 8041003706, 8041003705, 8041003000, 8041003303, 8041004503, 8041002900, 8041002000, 8041004502, 8041001900, 8041001800, 8041001700, 8041001600, 8041001500, 8041003906,8041001101, 8041001000, 
        8041000900, 8041005109, 8041005105, 8041005107, 8041007102, 8041004705, 8041005106, 8041004706, 8041006902, 8041004603, 8041004510, 8041005108,  8041003308, 8041003305, 8041004401, 8041004403, 8041004402, 8041003306, 8041005104, 8041005502, 8041004702, 8041000101, 8041008000,8041003709, 
        8041007400, 8041003707, 8041003708, 8041006502, 8041006600,8041006501, 8041006400, 8041005601, 8041005400, 8041005300, 8041005201, 8041004800,  8041002700, 8041002501 ,8041002502, 8041002400, 8041002200, 8041002102, 8041003100, 8041002101, 8041001400,8041001302, 8041001104, 8041000800 ,
        8041000700 ,8041000600, 8041000500 ,8041000400, 8041007800 ,8041000302 ,8041007700]
        mesa_county = ['0' + str(i) for i in mesa_county]
        co_springs = ['0' + str(i) for i in co_springs]
        df = gpd.read_file(tract_data_dir)
        df.rename(columns={'GEOID_TRAC': 'GEOID'}, inplace=True)  

        
        print(f'Old num of nodes: {self.num_nodes}')
        df = df[['GEOID', 'geometry']].to_crs(self.tract_data['geometry'].crs)
        df = df[df['GEOID'].isin(mesa_county) |df['GEOID'].isin(co_springs) ]

        # modify index based on existing data
        max_index = self.tract_data.index.max()
        new_index = [i + max_index + 1 for i in range(0, df.shape[0])]
        df.set_index([pd.Index(new_index)], inplace=True)

        node_idx_map = dict(zip(df['GEOID'], df.index))
        for key, value in node_idx_map.items():
            if value in self.idx_node_map.keys():
                print('Trying to insert an index that already exists in the object. Check.')
                quit()

        idx_node_map = dict(zip(df.index, df['GEOID']))
        tract_data =  df #.to_crs("EPSG:32643")
        num_nodes = df.shape[0] # each row is a unique combo of state/county/tract code 

        # combine
        self.node_idx_map = self.node_idx_map | node_idx_map
        self.idx_node_map = self.idx_node_map | idx_node_map
        self.tract_data = pd.concat([self.tract_data, tract_data], ignore_index=False)
        self.num_nodes += num_nodes
        self.edge_mat = np.zeros((self.num_nodes, self.num_nodes))

        print(f'New num of nodes: {self.num_nodes}')
        print('IMPORTANT: Note that the edge matrix weights were reset and need to be recomputed.')

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
        df.drop(index=df[df.GEOID.isna()].index, inplace=True)

        # now iterate thru the df and sum weights
        df.apply(self.add_row_weights_to_graph, axis=1)

    def get_tract_num(self, point):
        if not self.tract_data['geometry'].contains(point).any():
            return np.nan
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

    



