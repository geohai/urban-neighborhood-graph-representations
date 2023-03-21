from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from os.path import join, exists
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms.functional as TF
import random
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms

import random
import sys
sys.path.append('../dataset/safegraph/')
import imageio.v2 as io
import glob

sys.path.append('utils/')


###### ---------------------TRANSFORM / DATALOADER CLASSES------------------- ######


class Inceptionv3Transform:
    """Feed images through ResNet-18 model to get embeddings."""

    def __init__(self, device):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, progress=False)
        num_features = model.fc.in_features 
        model.fc = nn.Linear(num_features, 200)
        self.inception_v3 = model.eval()
        self.inception_v3 = self.inception_v3.to(device)
        
        self.transforms = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
            

    def __call__(self, x):
        with torch.no_grad():
            x = torch.tensor(x).float()
            x = torch.nan_to_num(x, 0, 0, 0)
        
#             if len(x.shape) != 3:
#                 print(x.shape)
#                 print(x)
            try:
                x = torch.reshape(x, (3, x.shape[0], x.shape[1]))
            except:
                print(x.shape)
                x = torch.reshape(x, (x.shape[2], x.shape[0], x.shape[1]))
            x = self.transforms(x)
            x = torch.unsqueeze(x, 0)
            with torch.no_grad():
#                 print(x.shape)
                x = x.to(device)
                y_pred = self.inception_v3(x)
            y_pred = torch.squeeze(y_pred)
            if(not np.isfinite(y_pred.cpu().detach().numpy()).all()):
                print('Invalid num in inception transform: {y_pred}')
            return y_pred
        
class ResNetTransform:
    """Feed images through ResNet-18 model to get embeddings."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights, progress=False)
        num_features = model.fc.in_features 
        model.fc = nn.Linear(num_features, 200)
#         model = FullyConvolutionalResnet18(pretrained=True)

        self.resnet18 = model.eval()
        self.resnet18 = self.resnet18.to(self.device)
        
        self.transforms =  torchvision.transforms.Compose([
#             torchvision.transforms.RandomHorizontalFlip(p=0.5),
#             torchvision.transforms.Resize((224, 224)),
#             torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            

    def __call__(self, x):
        with torch.no_grad():
            x = torch.tensor(x)
            x = torch.nan_to_num(x, 0, 0, 0)
            x = x.to(self.device)
            x = self.transforms(x)
            x = torch.unsqueeze(x, 0)
            with torch.no_grad():
                y_pred = self.resnet18(x)
                y_pred = torch.softmax(y_pred, dim=1)
            y_pred = torch.squeeze(y_pred)
            return y_pred
        
        
########### -------------------DATA CLASSES ----------------------##########
class SatelliteImageryDataset(Dataset):
    """
    csv_file: Path to the csv file containing "node" column with geoids.
    root_dir: Directory with subfolders containing images for each geometry.
    transform (callable, optional): transforms on images.
    """
    
    def __init__(self, node_list_path, root_image_dir, fn, is_train=True, transform=None, load_embeddings=True):
        self.node_list = pd.read_csv(node_list_path, dtype={'GEOID': str})
        self.root_dir = root_image_dir
        self.fn = fn
        self.is_train = is_train
        self.transform = transform
        self.load_embeddings = load_embeddings
        
        if self.is_train:
            self.index = self.node_list.index.values
        
    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # path to the positive sample
        pos_idx = idx
        pos_img_dir = os.path.join(self.root_dir, str(self.node_list.iloc[pos_idx, 0]))
        
        img_path = os.path.join(pos_img_dir, 'img_resnet.pt')
        if self.load_embeddings == True and os.path.exists(img_path):
            pos_image = torch.load(img_path)
        else:
            img_path = os.path.join(pos_img_dir, self.fn)
            pos_image = io.imread(img_path)
            pos_image = self.get_band_from_landsat(pos_image, bands=[2, 1, 0])
            if self.transform:
                pos_image = self.transform(pos_image)
                save_path = os.path.join(pos_img_dir, 'img_resnet.pt')
                torch.save(pos_image, save_path)
        
        if self.is_train:
            # choose a negative sample
            negative_list = self.index[self.index!=pos_idx]
            neg_idx = random.choice(negative_list)
            
            img_dir = os.path.join(self.root_dir, str(self.node_list.iloc[neg_idx, 0]))
            
            img_path = os.path.join(img_dir, 'img_resnet.pt')
            if self.load_embeddings == True and os.path.exists(img_path):
                neg_image = torch.load(img_path)
                return idx, pos_image, neg_image
            else:
                img_path = os.path.join(img_dir, self.fn)
                neg_image = io.imread(img_path)
                neg_image = self.get_band_from_landsat(neg_image, bands=[2, 1, 0])
                if self.transform:
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

    def __getsampletest__(self, idx):
            # path to the positive sample
            pos_idx = idx
            img_dir = os.path.join(self.root_dir, str(self.node_list.iloc[pos_idx, 0]))
            img_path = os.path.join(img_dir, self.fn)

            pos_image = io.imread(img_path)
            pos_image = self.get_band_from_landsat(pos_image, bands=[2, 1, 0])

            return pos_image
                                 


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
        
        is_valid = False
        while(is_valid == False):
            print(files)
            print()
            try:
                pos_fn = random.choice(files)
                pos_image = np.array(io.imread(pos_fn))
                if len(pos_image.shape) == 3:
                    is_valid = True
                else:
                    print(pos_fn)
            except:
                print(img_dir)
                print(files)
                     
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

    
class MobilityDataset(Dataset):
    """
    Generates node indices for triplet sampling with probability based on edge weights between nodes.
    """
    def __init__(self, node_list_path, is_train=True, threshold=0.5, data=None, node_idx_mapping=None, idx_node_mapping=None):
        node_list = pd.read_csv(node_list_path, dtype={'GEOID': str})
        self.dataset_node_idx_mapping = dict(zip(node_list['GEOID'], node_list.index))
        self.dataset_idx_node_mapping = dict(zip(node_list.index, node_list['GEOID']))
        self.threshold = threshold # distance threshold from which to sample a positive sample
        
        self.graph = data
        self.graph_node_idx_mapping = node_idx_mapping
        self.graph_idx_node_mapping = idx_node_mapping
        
               
        self.is_train = is_train
        
        if self.is_train:
            self.index = node_list.index.values

        print(f'Num Nodes: {self.__len__()}')
        
        # identify rows with only zeros and remove from node_list
        zero_row_idxs = np.where((self.graph != 0).sum(axis=1) == 0)[0]
        zero_rows = len(zero_row_idxs)
        print(f'Number of tracts with 0 edges: {zero_rows}')
        
        print(self.graph[0:10, 0:10])
        

    def __len__(self):
        if len(self.dataset_node_idx_mapping.keys()) != self.graph.shape[0]:
            print('WARNING - Length of node list and edge weight matrix are not equal.')
            print(len(self.dataset_node_idx_mapping.keys()))
            print(self.graph.shape[0])
        return len(self.dataset_node_idx_mapping.keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
 
        # idx is the geoid in self.node_list
        idx_geoid = self.dataset_idx_node_mapping[idx]
            
        # geoid to graph idx
        anchor_idx_graph = self.graph_node_idx_mapping[idx_geoid]
        
        # get dataset idxs of neighbors
        positive_candidate_idx_list = self.return_positive_candidates(anchor_idx_graph)
    
        if len(positive_candidate_idx_list) == 0:
            print(f'ERROR: No positive candidates for geoid:{idx_geoid}.')
            
        # get dataset idxs of non-neighbors
        negative_candidate_idx_list = [i for i in self.dataset_node_idx_mapping.values() if i not in positive_candidate_idx_list]

        # randomly sample positive and negative
        pos_idx = random.choice(positive_candidate_idx_list)
        neg_idx = random.choice(negative_candidate_idx_list) 
        
        return idx, pos_idx, neg_idx


    def return_positive_candidates(self, anchor_idx):        
        # get all the edge weights from this anchor node
        edges = self.graph[anchor_idx]
        
            
#         neighbor_weights = np.reciprocal(edges, where=edges!=0)
        
        # if reciprocol distances is over threshold, it is a neighbor
#         valid = neighbor_weights*1000 < self.threshold # (km)
        threshold_percentile = np.quantile(a=edges, q=self.threshold)
        valid = np.argwhere(edges > threshold_percentile)

        valid = valid.flatten()
        valid = np.delete(valid, np.where(valid == anchor_idx))
        
        return valid
       
    
    
class DistanceDataset(Dataset):
    """
    Generates node indices for triplet sampling with probability based on edge weights between nodes.
    """
    def __init__(self, node_list_path, is_train=True, threshold=0.5, data=None, node_idx_mapping=None, idx_node_mapping=None):
        node_list = pd.read_csv(node_list_path, dtype={'GEOID': str})
        self.dataset_node_idx_mapping = dict(zip(node_list['GEOID'], node_list.index))
        self.dataset_idx_node_mapping = dict(zip(node_list.index, node_list['GEOID']))
        self.threshold = threshold # distance threshold from which to sample a positive sample
        
        self.graph = data
        self.graph_node_idx_mapping = node_idx_mapping
        self.graph_idx_node_mapping = idx_node_mapping
        
        self.is_train = is_train
        if self.is_train:
            self.index = node_list.index.values

        print(f'Num Nodes: {self.__len__()}')
        
        print(self.graph[0:5, 0:5])
      

    def __len__(self):
        if len(self.dataset_node_idx_mapping.keys()) != self.graph.shape[0]:
            print('WARNING - Length of node list and edge weight matrix are not equal.')
            print(f'Dataset length: {len(self.dataset_node_idx_mapping.keys())}')
            print(f'Graph matrix length: {self.graph.shape[0]}')
        return len(self.dataset_node_idx_mapping.keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
 
        # idx is the geoid in self.node_list
        idx_geoid = self.dataset_idx_node_mapping[idx]
            
        # geoid to graph idx
        anchor_idx_graph = self.graph_node_idx_mapping[idx_geoid]
        
        # get dataset idxs of neighbors
        positive_candidate_idx_list = self.return_positive_candidates(anchor_idx_graph)
    
        if len(positive_candidate_idx_list) == 0:
            print(f'ERROR: No positive candidates for geoid:{idx_geoid}.')
            
        # get dataset idxs of non-neighbors
        negative_candidate_idx_list = [i for i in self.dataset_node_idx_mapping.values() if i not in positive_candidate_idx_list and i != idx]

        # randomly sample positive and negative
        pos_idx = random.choice(positive_candidate_idx_list)
        neg_idx = random.choice(negative_candidate_idx_list) 

        return idx, pos_idx, neg_idx


    def return_positive_candidates(self, anchor_idx):
        # get 5 closest neighbors according to paper (just for distance)
        # get all the edge weights from this anchor node
        edges = self.graph[anchor_idx]
#         neighbor_weights = np.reciprocal(edges, where=edges!=0)
        
#         # if reciprocol distances is over threshold, it is a neighbor
#         valid = neighbor_weights*1000 > self.threshold # (km)
#         candidates = edges[valid] 
        candidates = edges
        
       
        # now filter to nearest neighbors and return their indices
        idxs = candidates.argsort()[0:11]
        idxs = np.delete(idxs, np.where(idxs == anchor_idx))

        return idxs