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
sys.path.append('safegraph/utils/')

# from torch_geometric.sampler import (
#     BaseSampler,
#     EdgeSamplerInput,
#     NodeSamplerInput,
#     SamplerOutput,
# )


def create_text_bow_embeddings(poi_list):
    # get embedding for each token; put into set representing bow
    bag = []
    words = set()
    for poi in poi_list:
        token_list = tokenizer(poi)

        for token in token_list:
            if token in words:
                continue
            try:
                words.add(token)
                emb = embeddings_dict[token].tolist()
                bag.append(emb)

            except: # word is not in table and skip
                print(token)
                pass
    return bag

def make_poi_text_request(geometry):
    # call api for a region
        # Get a random bare-bone sentence
    poi_list = [s.bare_bone_sentence() for i in range(10)]


    # poi_list = [["Visited Colorado in June 2018 with a group of friends and knew whitewater rafting had to be something we did while we were there. After a lot of research and review reading, I decided to book with Rocky Mountain. We booked the Beginner trip as I was pretty nervous about doing anything higher than that as a group of newbies. It ended up being a great introduction to whitewater, but we'll definitely be doing the intermediate next time. Booking was easy and they let you do the waivers before hand to help speed up check-in! We originally booked with a group of 5 but one of us got held back by altitude sickness, so we ended up being about 25 minutes late for our call time, but the staff was very cool and understanding about it. Our guide was Ron and he was a very chill, knowledgeable dude. Even though we messed up on our paddling a few times he never made us feel stupid for it lol. Overall it was a great experience and if we find ourselves back in the area we will be booking with them again!", "Oh my goodness, I'm giving this place all of the stars! Wish I could give them a million.", 
    # "I've never been whitewater rafting before so I was obviously nervous about doing it. They explained everything that we needed to do in order to be successful in such great detail! I felt completely at ease."]] # list of text info about all POIs in the region
    return poi_list

def call_api(api_name, geometry):
    if api_name == 'poi_text':
        return make_poi_text_request(geometry)
    return 


# testing
from wonderwords import RandomSentence
s = RandomSentence()

glv_emb, glv_vocab = load_glove_embeddings()
embeddings_dict = dict(zip(glv_vocab, glv_emb))

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
    query = call_api(api_name='poi_text', geometry=geom)
    text_bow_embeddings = create_text_bow_embeddings(query)
    bow_list_i = list(text_bow_embeddings)
    region_bow_map[geoid] = bow_list_i.copy()

with open('data_files/poi.json', 'w') as fp:
    json.dump(region_bow_map, fp)
