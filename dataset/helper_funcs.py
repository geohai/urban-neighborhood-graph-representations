import numpy as np
import os
import urllib.request
import json
import os
import shutil
import argparse
import torch
from torch import nn
import torchvision
from torchvision import transforms

class EncoderInception3(nn.Module):
    def __init__(self, embedding_dim=200):
        super(EncoderInception3, self).__init__()
        self.model = torchvision.models.inception_v3(weights='DEFAULT')
        self.set_parameter_requires_grad(True)  # freeze all but last layer

        # Handle the auxilary net
        self.model.aux_logits = False

        # Handle the primary net
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, images):
        x = self.model(images)  
        return x

    def set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False


def dummy_bbox(bbox):
    bb_new = {}
    bb_new['west'] = bbox['west'] + 0.0001
    bb_new['south'] = bbox['south'] + 0.0001
    bb_new['east'] = bbox['east'] + 0.0001
    bb_new['north'] = bbox['north'] + 0.0001
    return bb_new


def read_glove_file(fp_emb, fp_vocab):
    if os.path.isfile(fp_emb) & os.path.isfile(fp_vocab):
        emb = np.load(fp_emb) 
        vocab = np.load(fp_vocab)
        return emb, vocab
    return None

def load_glove_embeddings(fp_emb='data_files/embeddings.npy', fp_vocab='data_files/vocab.npy'):
    """
    Need to download the embedding files first before running this, and place in data_files directory. I downloaded the Wikipedia 2014 + Gigaword  Glove embeddings bc they were smaller. From here(https://github.com/stanfordnlp/GloVe).
    """
    load = read_glove_file(fp_emb, fp_vocab)
    if load is not None:
        return load
    # load glove embeddings - this code section is taken from Tanmay Garg @ https://medium.com/mlearning-ai/load-pre-trained-glove-embeddings-in-torch-nn-embedding-layer-in-under-2-minutes-f5af8f57416a
    vocab, embeddings = [],[]
    with open('data_files/glove.6B.200d.txt','rt') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)

    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)

    with open(fp_vocab,'wb') as f:
        np.save(f,vocab_npa)

    with open(fp_emb,'wb') as f:
        np.save(f,embs_npa)

    return read_glove_file(fp_emb, fp_vocab)


def createCleanDir(base_path):
  try:
      shutil.rmtree(base_path)
  except:
      pass
  os.mkdir(base_path)


def createDir(dirPath):
  try:
    os.makedirs(dirPath)
  except:
    pass

