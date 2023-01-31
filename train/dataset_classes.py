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


#graph_obj_path = '../dataset/safegraph/compute_graph_checkpoints/checkpoint_11.pkl'
class NodeDataset(Dataset):
    """
    This can be used for both images and text data. Assumes that BOW dictionary has a key for every region in the graph. If the value is null, it resamples a new region key.
    """
    def __init__(self, graph_obj_path, data_dir='../dataset/data_files/final_node_data/', fn='poi.json', data_type='poi', threshold=None):
        path = data_dir + fn
        self.data_type = data_type
        # read obj
        with open(graph_obj_path, 'rb') as f:
            g = pickle.load(f) # CensusTractMobility object

        self.node_idx_map = g.get_node_idx() # dictionary = (regionid: idx)
        self.idx_node_map = g.get_idx_node() # dictionary = (idx: region_id)

        # read json which contains BOW embeddings keyed by neighborhood idx
        with open(path, 'r') as fp:
            self.bow = json.load(fp)
            print(f'Num Nodes: {self.__len__()}')

    def __len__(self):
        return len(self.idx_node_map.keys())

    def __getitem__(self, idx):
        # generate anchor, pos, neg triplets
        # print(idx)

        # anchor idx
        valid = False
        while valid == False:
            anchor_sampled_idx = idx
            anchor_geoid = self.idx_node_map[anchor_sampled_idx] 
            try:
                pos_bow_embeddings = self.bow[anchor_geoid]
                torch.tensor(random.choices(pos_bow_embeddings, k=30))
                valid=True
            except:
                # what to do if region has no data?
                # for now, use data for closest region
                print('No data in anchor region.')
                return (0, torch.zeros(200), torch.zeros(200))
            
        # some regions don't have data. For the negative sample keep choosing until we find one with data.
        valid = False  
        while valid == False:
            neg_sampled_idx = self.sample_negative_node(0, self.__len__(), anchor_sampled_idx) # sample negative node
            try:
                neg_geoid = self.idx_node_map[neg_sampled_idx] 
                neg_bow_embeddings = self.bow[neg_geoid]
                random.choices(neg_bow_embeddings, k=30)
                valid = True
            except:
                # print('Invalid negative node, resampling.')
                pass
        
        if self.data_type == 'poi':
            positives = random.choices(pos_bow_embeddings, k=30)
            positives = np.mean(np.array(positives), axis=0)
            negatives =  random.choices(neg_bow_embeddings, k=30)
            negatives = np.mean(np.array(negatives),axis=0)
        else:
            positives = random.choice(pos_bow_embeddings)
            negatives =  random.choice(neg_bow_embeddings)
            pass
        
        return anchor_sampled_idx, torch.tensor(positives), torch.tensor(negatives)

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
    def __init__(self, graph_obj_path, data_dir='../dataset/data_files/final_edge_data/', fn='distance.npy', data_type='distance', threshold=0.5):
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
         return len(self.idx_node_map.keys())

    def __getitem__(self, idx):
        node_idx_list_cp = list(self.idx_node_map.keys())

        anchor_sampled_idx = idx # sample anchor node

        candidate_idx_list = self.return_positive_candidates_distance(anchor_sampled_idx)
        # if self.data_type == 'distance':
        #     candidate_idx_list = self.return_positive_candidates_distance(anchor_sampled_idx)   # list of positive sample candidates
        # else:
        #     candidate_idx_list = self.return_positive_candidates_weights(anchor_sampled_idx)   # list of positive sample candidates

        [node_idx_list_cp.remove(i) for i in candidate_idx_list] # remove neighbors 

        try:
            pos_idx = random.choice(candidate_idx_list)
            neg_idx = self.sample_node(node_idx_list_cp) # sample negative node
        except: # if the node has no neighbors
            return self.__getitem__(idx)
        
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
        edges = self.edge_weight_mat[anchor_idx]
 
        # if reciprocol distances is over threshold, it is a neighbor
        valid = neighbor_weights*1000 > self.threshold  # (km)
        candidates = np.array(edges[valid] )
        # print(candidates)
        idxs = np.nonzero(candidates)

        # print(candidates[idxs])
        # print('--')
        return  idxs[0]# return indices of nonzero elements
        

    def sample_node(self, node_idx_list_cp):
        #  draw tensor of random indices for which to select nodes
        sample = random.choice(node_idx_list_cp)
        return sample
        
        