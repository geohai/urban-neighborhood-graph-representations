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
import nltk 
# nltk.download('stopwords')
from nltk.corpus import stopwords
import string 

sys.path.append('safegraph/utils/')
sys.path.append('yelp/')
from query_api import *
from mobility_processor import *

BASE_DIR = 'yelp/data/'
# createDir(BASE_DIR)

def write_str_to_file(s, fn):
    with open(fn, 'w') as fp:
        fp.write(s)
       
def filter_bow_sentence(region_bow_string):
    # remove punctuation 
    no_punctuation = region_bow_string.translate(str.maketrans('', '', string.punctuation))
    
    # tokenize, remove words not in the vocab, only unique words, and filter out filler/common owrds
    token_list = tokenizer(no_punctuation)
    token_list = list(set(token_list))
    token_list = list(filter(lambda w: not w not in embeddings_dict.keys(), token_list))
    filtered_string = list(filter(lambda w: not w in s, token_list))
    return filtered_string

def create_text_bow_embeddings(region_bow_string):
    # get embedding for each word
    bag = [] # list of embeddings
    for token in region_bow_string:
        emb = embeddings_dict[token].tolist()
        bag.append(emb)

    return bag

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
graph_obj_path = 'safegraph/compute_graph_checkpoints/grandjunction_denver/checkpoint_0.pkl'
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
with open('data_files/final_node_data/poi.json', 'r') as fp:
    region_bow_map = json.load(fp)

found = False
for i, (geoid, geom) in enumerate(zip(g.tract_data['GEOID'], g.tract_data['geometry'])):
    filename = BASE_DIR + str(geoid) + ".txt"
    if geoid in region_bow_map.keys():
        continue
    
    print('---------------------\n\n')
    print(geoid)
    
    # download poi data and extract embeddings
    bow_sentence = make_poi_text_request(geom)
    write_str_to_file(bow_sentence, filename)
    filtered_bow_sentence = filter_bow_sentence(bow_sentence)
    bow_list_i = create_text_bow_embeddings(filtered_bow_sentence)
    region_bow_map[str(geoid)] = bow_list_i.copy()

    with open('data_files/final_node_data/poi.json', 'w') as fp:
        json.dump(region_bow_map, fp)
        print('save')

    with open('data_files/final_node_data/log_poi.txt', 'a') as fp:
        fp.write(f'{i}: {geoid} \n')

