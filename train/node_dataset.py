import torch
import torch.nn as nn
from torch.utils.data import Dataset

import random
import pandas as pd
import numpy as np
import pickle
import json

graph_obj_path = '../dataset/safegraph/compute_graph_checkpoints/checkpoint_0.pkl'
class NodeDataset(Dataset):
    """
    This can be used for both images and text data.
    """
    def __init__(self, graph_obj_path=graph_obj_path, data_dir='../dataset/data_files/', fn='poi.json', data_type='poi', threshold=None):
        path = data_dir + fn
        
        # read obj
        with open(graph_obj_path, 'rb') as f:
            g = pickle.load(f) # CensusTractMobility object
        self.idx_node_map = g.get_idx_node() # dictionary = (idx: region_id)

        # read json which contains BOW embeddings keyed by neighborhood idx
        with open(path, 'r') as fp:
            self.bow = json.load(fp)

    def __len__(self):
        return len(self.idx_node_map.keys())

    def __getitem__(self, idx):
        # generate anchor, pos, neg triplets
        anchor_sampled_idx = self.sample_anchor_node(0, self.__len__()) # sample anchor node
        neg_sampled_idx = self.sample_negative_node(0, self.__len__(), anchor_sampled_idx) # sample negative node
        
        pos_bow_embeddings = self.bow[str(anchor_sampled_idx)]
        neg_bow_embeddings = self.bow[str(neg_sampled_idx)]        
        
        return anchor_sampled_idx, torch.tensor(random.choice(pos_bow_embeddings)), torch.tensor(random.choice(neg_bow_embeddings))

    def sample_anchor_node(self, low, high):
        # set the seed and draw tensor of random indices for which to select nodes
        sample = torch.randint(low, high, (1, 1)).item()
        return sample
        
    def sample_negative_node(self, low, high, anchor):
        # draw tensor of random indices for which to select nodes
        sample = torch.randint(low, high, (1, 1)).item()
        while sample == anchor:
            sample = torch.randint(low, high, (1, 1)).item()
        return sample
    
class EdgeDataset(Dataset):
    """
    Generates node indices for triplet sampling with probability based on edge weights between nodes.
    """
    def __init__(self, graph_obj_path=graph_obj_path, data_dir='../dataset/data_files/final_edge_data/', fn='distance.npy', data_type='distance', threshold=500):
        path= data_dir + fn
        self.g = None
        self.threshold = threshold # distance threshold in meters from anchor from which to sample a positive sample
        self.data_type = data_type
        self.edge_weight_mat = None

        # read node list
        with open(graph_obj_path, 'rb') as f:
            g = pickle.load(f) # CensusTractMobility object

        self.node_idx_map = g.get_node_idx() # dictionary = (regionid: idx)
        self.idx_node_map = g.get_idx_node() # dictionary = (idx: region_id)

        # read graph obj
        with open(path, 'rb') as f:
            self.edge_weight_mat = np.load(f)

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        node_idx_list_cp = self.idx_node_map.keys()

        anchor_sampled_idx = self.sample_node(node_idx_list_cp) # sample anchor node

        if self.data_type == 'distance':
            candidate_idx_list = self.return_positive_candidates_distance(anchor_sampled_idx)   # list of positive sample candidates
        else:
            candidate_idx_list = self.return_positive_candidates_weights(anchor_sampled_idx)   # list of positive sample candidates

        [node_idx_list_cp.remove(i) for i in candidate_idx_list] # remove neighbors 

        try:
            pos_idx = random.choice(candidate_idx_list)
            neg_idx = self.sample_node(node_idx_list_cp) # sample negative node
        except: # if the node has no neighbors
            return self.__getitem__(idx)
        
        return anchor_sampled_idx, pos_idx, neg_idx


    def return_positive_candidates_distance(self, anchor_idx):
        # get 5 closest neighbors according to paper (just for distance)
        neighbor_weights = np.reciprocal(self.edge_weight_mat[anchor_idx], where=self.edge_weight_mat[anchor_idx]!=0)
        edges = self.edge_weight_mat[anchor_idx]
 
        # if reciprocol distances is over threshold, it is a neighbor
        valid = neighbor_weights < self.threshold
        candidates = edges[valid] 

        # now filter to five nearest neighbors
        idxs = candidates.argsort()[-5:][::-1]

        return candidates[idxs]
    
    def return_positive_candidates_weights(self, anchor_idx):
        neighbor_weights = np.reciprocal(self.edge_weight_mat[anchor_idx], where=self.edge_weight_mat[anchor_idx]!=0)
        edges = self.edge_weight_mat[anchor_idx]
 
        # if reciprocol distances is over threshold, it is a neighbor
        valid = neighbor_weights < self.threshold
        candidates = edges[valid] 

        return candidates
        

    def sample_node(self, node_idx_list_cp):
        #  draw tensor of random indices for which to select nodes
        sample = random.choice(node_idx_list_cp)
        return sample
        
        