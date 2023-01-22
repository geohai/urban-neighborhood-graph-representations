import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import json
from torchvision.io import read_image

import os, random
from os.path import join, exists
import copy
import pandas as pd
import numpy as np

# functions


class NodeDataset(Dataset):
    """
    This can be used for both street view images and text data.
    """
    def __init__(self, node_list_file='../dataset/data_files/node_list.csv', data_dir='../dataset/data_files/poi.json'):
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
        neg_sampled_idx = self.sample_negative_node(0, self.__len__(), anchor_sampled_idx) # sample anchor node
        
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

    

# class PlacePairDataset(Dataset):
#     def __init__(self, path_list,weight,id_list,num):
#         self.path_list = path_list
#         self.weight=weight
#         self.id_list=id_list
#         self.num=num
#     def __len__(self):
#         return len(self.path_list)
#     def __getitem__(self, idx):
#         pos_idx,place_idx = self.path_list[idx]
#         media1=copy.deepcopy(self.weight)
#         media2=np.zeros(self.num)
#         media2[self.id_list[place_idx]]=self.weight[self.id_list[place_idx]]
#         media1=media1-media2
#         media1=np.power(media1,0.5)
#         media1=media1/np.sum(media1)
#         neg_idx=np.random.choice(np.arange(self.num),size=1,p=media1)[0]
#         sample = [torch.tensor(place_idx, dtype=torch.long), torch.tensor(pos_idx, dtype=torch.long), torch.tensor(neg_idx, dtype=torch.long)]
#         return sample