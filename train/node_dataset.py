import torch
import torch.nn as nn
from torch.utils.data import Dataset

import random
import pandas as pd
import numpy as np
import pickle
import json

path_to_node_list = '../dataset/data_files/node_list.csv'
class NodeDataset(Dataset):
    """
    This can be used for both images and text data.
    """
    def __init__(self, node_list_file=path_to_node_list, data_dir='../dataset/data_files/poi.json'):
        self.data_dir = data_dir
        
        # read node list
        self.node_list = np.array(pd.read_csv(node_list_file))

        # read json which contains BOW embeddings keyed by neighborhood idx
        with open(self.data_dir, 'r') as fp:
            self.bow = json.load(fp)

    def __len__(self):
        return len(self.node_list)

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
    Generates node indices for triplet sampling with probability based on distance between nodes.
    """
    def __init__(self, node_list_file=path_to_node_list, data_dir='../dataset/data_files/edge_graph_obj', threshold=1/500):
        self.data_dir = data_dir
        self.g = None
        self.threshold = threshold # distance threshold in meters from anchor from which to sample a positive sample

        # read node list
        self.node_list = np.array(pd.read_csv(node_list_file))
        self.node_idx_list = [i for i in range(0, len(self.node_list))]

        # read graph obj
        with open(self.data_dir, 'rb') as f:
            self.g = pickle.load(f)

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        node_idx_list_cp = self.node_idx_list.copy()

        anchor_sampled_idx = self.sample_node(node_idx_list_cp) # sample anchor node
        candidate_idx_list = self.return_positive_candidates(anchor_sampled_idx)   # list of positive sample candidates
        [node_idx_list_cp.remove(i) for i in candidate_idx_list] # remove neighbors 

        
        try:
            pos_idx = random.choice(candidate_idx_list)
            neg_idx = self.sample_node(node_idx_list_cp) # sample negative node
        except: # if the node has no neighbors
            print(candidate_idx_list)
            return self.__getitem__(idx)
        
        print('return')
        return anchor_sampled_idx, pos_idx, neg_idx


    def return_positive_candidates(self, anchor_idx):
        node_edge_idx = self.g.node_idx_in_edge[anchor_idx].tolist() # list of edge idxs

        # get the distances of edges
        out_edges = self.g.edge_index[node_edge_idx]
        raw_weights = self.g.edge_attr[node_edge_idx]
        reciprocol_weights = 1/raw_weights
     
        # if reciprocol distances is over threshold, it is a neighbor
        valid = reciprocol_weights > self.threshold
        candidates = out_edges[valid][:, 1] # return node index
        return candidates

    def sample_node(self, node_idx_list_cp):
        #  draw tensor of random indices for which to select nodes
        sample = random.choice(node_idx_list_cp)
        return sample
        
        