"""
This file preprocesses the data and stores it. This includes randomly sampling 

"""


import torch
from torch_geometric.data import Data
from torchtext.data import get_tokenizer
import numpy as np
import pandas as pd
import copy
from helper_funcs import *
import json
import pickle
import sys
from nltk.corpus import stopwords

sys.path.append('safegraph/utils/')
sys.path.append('yelp/')
from query_api import *
from mobility_processor import *


def create_text_bow_embeddings(region_bow_string):
    # get embedding for each word
    bag = [] # list of embeddings
    words = set() # only unique words are added
    
    token_list = tokenizer(region_bow_string)

    for token in token_list:
        if token in words:
            continue
        try:
            words.add(token)
            emb = embeddings_dict[token].tolist()
            bag.append(emb)

        except: # word is not in table and skip
            pass

        # remove stop words
    return filter(lambda w: not w in s, bag)

def make_poi_text_request(polygon):
    # call api for a region
    bow = ' '
    point = polygon.centroid  # assume that centroid is in shape
    bow += query_api(str(point.x), str(point.y))

    return bow


glv_emb, glv_vocab = load_glove_embeddings()
embeddings_dict = dict(zip(glv_vocab, glv_emb))
s=set(stopwords.words('english')) # define stop words

t = torch.from_numpy(glv_emb).float()
my_embedding_layer = torch.nn.Embedding.from_pretrained(t, freeze=False)
tokenizer = get_tokenizer("basic_english") # very simple tokenizer, maybe select a better one . this one splits each punctuation to be a token. (https://pytorch.org/text/stable/data_utils.html)


# read obj
graph_obj_path = 'safegraph/compute_graph_checkpoints/checkpoint_0.pkl'
with open(graph_obj_path, 'rb') as f:
    g = pickle.load(f) # CensusTractMobility object
idx_node_map = g.get_idx_node() # dictionary = (idx: region_id)
torch.manual_seed(42)

# POI and SV do not require an actual graph structure
num_nodes = len(idx_node_map.keys())
num_node_features = 2 # embedding dimension
node_embedding_length = 200
num_sample = 50

# create bow embeddings; store in dict 
region_bow_map = {}

for geoid, geom in zip(g.tract_data['GEOID'], g.tract_data['geometry']):
    # download poi dtaa
    bow_sentence = make_poi_text_request(geom)
    bow_list_i = create_text_bow_embeddings(bow_sentence)
    region_bow_map[geoid] = bow_list_i.copy()

with open('data_files/poi.json', 'w') as fp:
    json.dump(region_bow_map, fp)
