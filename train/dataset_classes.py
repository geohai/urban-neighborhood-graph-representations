import torch
import torch.nn as nn
from torch.utils.data import Dataset

import random
import pandas as pd
import numpy as np
import pickle
import json
import sys
sys.path.append('../dataset/safegraph/utils/')
import os
import imageio as io
import glob

class SatelliteImageryDataset(Dataset):
    """
    csv_file: Path to the csv file containing "node" column with geoids.
    root_dir: Directory with subfolders containing images for each geometry.
    transform (callable, optional): transforms on images.
    """
    
    def __init__(self, node_list_path, root_image_dir, fn, is_train=True, transform=None):
        self.node_list = pd.read_csv(node_list_path, dtype={'GEOID': str})
        self.root_dir = root_image_dir
        self.fn = fn
        self.is_train = is_train
        self.transform = transform
        
        if self.is_train:
            self.index = self.node_list.index.values
        
    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # path to the positive sample
        pos_idx = idx
        img_dir = os.path.join(self.root_dir, str(self.node_list.iloc[pos_idx, 0]))
        img_path = os.path.join(img_dir, self.fn)
        
        pos_image = io.imread(img_path)
        pos_image = self.get_band_from_landsat(pos_image, bands=[3, 2, 1])
        
        if self.is_train:
            # choose a negative sample
            negative_list = self.index[self.index!=pos_idx]
            neg_idx = random.choice(negative_list)
            
            img_dir = os.path.join(self.root_dir, str(self.node_list.iloc[neg_idx, 0]))
            img_path = os.path.join(img_dir, self.fn)
            
            neg_image = io.imread(img_path)
            neg_image = self.get_band_from_landsat(neg_image, bands=[3, 2, 1])
            

        if self.transform:
            pos_image = self.transform(pos_image)                   
            neg_image = self.transform(neg_image)
            
        return idx, pos_image, neg_image
    
    def get_band_from_landsat(self, img, bands):
        img = np.array(img)
        B1 = img[:,:,bands[0]]
        B2 = img[:,:,bands[1]]
        B3 = img[:,:,bands[2]]

        image = np.stack([B1, B2, B3], axis=0)
#         print(image)
        return image
                                 


class StreetViewDataset(Dataset):

    """
    csv_file: Path to the csv file containing "node" column with geoids.
    root_dir: Directory with subfolders containing images for each geometry.
    transform (callable, optional): transforms on images.
    """
    
    def __init__(self, node_list_path, root_image_dir, is_train=True, transform=None):
        self.node_list = pd.read_csv(node_list_path, dtype={'GEOID': str})
        self.root_dir = root_image_dir
        self.is_train = is_train
        self.transform = transform
        
        if self.is_train:
            self.index = self.node_list.index.values
        
    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # path to the positive sample
        pos_idx = idx
        img_dir = os.path.join(self.root_dir, str(self.node_list.iloc[pos_idx, 0]))
        img_dir_search = os.path.join(img_dir, '*.jpg')
        
        files = glob.glob(img_dir_search, recursive=False)
#         print(files)
        
        is_valid = False
        while(is_valid == False):
            pos_fn = random.choice(files)
    #         img_path = os.path.join(img_dir, pos_fn)
            pos_image = np.array(io.imread(pos_fn))
            if len(pos_image.shape) == 3:
                is_valid = True
            else:
                print(pos_fn)
                     
        if self.is_train:
            # choose a negative sample
            negative_list = self.index[self.index!=pos_idx]
            neg_idx = random.choice(negative_list)
            
            img_dir = os.path.join(self.root_dir, str(self.node_list.iloc[neg_idx, 0]))
            img_dir_search = os.path.join(img_dir, '*.jpg')
            
            files = glob.glob(img_dir_search, recursive=False)
            
            is_valid = False
            while(is_valid == False):
                neg_fn = random.choice(files)
                
                neg_image = np.array(io.imread(neg_fn))
                if len(neg_image.shape) == 3:
                        is_valid = True
                else:
                    print(neg_fn)

        if self.transform:
            
            pos_image = self.transform(pos_image)                   
            neg_image = self.transform(neg_image)
            
        return idx, pos_image, neg_image

    
class EdgeDataset(Dataset):
    """
    Generates node indices for triplet sampling with probability based on edge weights between nodes.
    """
    def __init__(self, node_list_path, graph_obj_path, data_dir='../dataset/data_files/final_edge_data/', fn='distance.npy', data_type='distance', threshold=0.5):
        self.node_list = pd.read_csv(node_list_path, dtype={'GEOID': str})
        path= data_dir + fn
        self.g = None
        self.threshold = threshold # distance threshold from which to sample a positive sample
        self.data_type = data_type
        self.edge_weight_mat = None

        # read node list
        with open(graph_obj_path, 'rb') as f:
            g = pickle.load(f) # CensusTractMobility object

        self.node_idx_map = g.get_node_idx() # dictionary = (regionid: idx)
        self.idx_node_map = g.get_idx_node() # dictionary = (idx: region_id)
        print(f'Num Nodes: {self.__len__()}')
        # read graph obj
        with open(path, 'rb') as f:
            print(path)
            self.edge_weight_mat = np.load(f)
            print(self.edge_weight_mat.shape)

    def __len__(self):
         return len(self.node_list)

    def __getitem__(self, idx):
        node_idx_list_cp = list(self.idx_node_map.keys())

        anchor_sampled_idx = idx # sample anchor node

        candidate_idx_list = self.return_positive_candidates_distance(anchor_sampled_idx)
        [node_idx_list_cp.remove(i) for i in candidate_idx_list] # remove neighbors 

        pos_idx = random.choice(candidate_idx_list)
        neg_idx = random.choice(node_idx_list_cp) # sample negative node

        return anchor_sampled_idx, pos_idx, neg_idx


    def return_positive_candidates_distance(self, anchor_idx):
        # get 5 closest neighbors according to paper (just for distance)
        edges = self.edge_weight_mat[anchor_idx]
        neighbor_weights = np.reciprocal(edges, where=edges!=0)*1000 # (km)
        
        # if reciprocol distances is over threshold, it is a neighbor
        valid = neighbor_weights*1000 > self.threshold # (km)
        candidates = edges[valid] 

        # now filter to five nearest neighbors and return their indices
        idxs = candidates.argsort()[-5:][::-1]
        return idxs
    
    def return_positive_candidates_weights(self, anchor_idx):
        neighbor_weights = np.reciprocal(self.edge_weight_mat[anchor_idx], where=self.edge_weight_mat[anchor_idx]!=0) # (km)
        neighbor_weights = np.nan_to_num(neighbor_weights, 0, 0, 0)
        edges = self.edge_weight_mat[anchor_idx]
 
        # if reciprocol distances is over threshold, it is a neighbor
        valid = neighbor_weights*1000 > self.threshold  
        candidates = np.array(edges[valid])
        idxs = np.nonzero(candidates)

        return  idxs[0]# return indices of nonzero elements
        