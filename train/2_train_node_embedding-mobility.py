from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.multiprocessing as mp

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

import sys
# sys.path.append('../dataset/safegraph/utils/')
sys.path.append('utils/')
from dataset_classes import * 
from models import *
from helper_funcs import *

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

from torch.multiprocessing import Pool, Process, set_start_method
set_start_method('spawn')


##########--------------------------- PARAMETERS -------------------------------#########
root = '/mnt/e/julia/regional-representations-graph-model/'
scenario = 'all_valid_data/'
print(f'Env Var Scenario: {scenario}')

# search = root + f'train/outputs/{scenario}landsat/'
# if os.path.exists(search):
#     for f in os.listdir(search):
#         if 'last.tar' in f:
#             print('FOUND LAST.TAR')
#             pre_trained = search + f
#             break
pre_trained = None
stage = 'mobility/' 
save_scenario = os.getenv('TRAIN_SCENARIO') + '/'
save_dir = 'outputs/' + save_scenario + stage 
data_type = 'mobility'#'distance' # 'mobility' #'poi' #'sv'
save_path = save_dir + "training_log_" + data_type + ".txt"
node_list_path= root + f'dataset/preprocessed_data/{scenario}node_list.csv'
dtype = 'Edge'

createCleanDir(save_dir)

with open(save_path, "w") as file:
    file.write('')
    
# device = "cpu"
current_epoch = None
embedding_dim = 200
num_nodes = pd.read_csv(node_list_path).shape[0]
print(f'Number of nodes: {num_nodes}')
return_best = True
if_early_stop = True
input_size = 500
learning_rate = [0.015]#[0.008]
weight_decay = [0.0005]
batch_size = num_nodes
num_epochs = 80
lr_decay_rate = 0.7
lr_decay_epochs = 6
early_stop_epochs = 18
save_epochs = 7
margin=2

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.cuda.current_device()

# set environmental variables
os.environ['TRAIN_SAVE_PATH'] = save_path
os.environ['TRAIN_DTYPE'] = dtype
os.environ['TRAIN_DATA_TYPE'] = data_type
os.environ['TRAIN_SAVE_DIR'] = save_dir

from train_functions import plot_curves, metrics, train_embedding

#######------------------------ TRAINING ------------------------------######

path = f'/mnt/e/julia/regional-representations-graph-model/dataset/preprocessed_data/{scenario}'

#### MOBILITY #####
if data_type == 'mobility':
    print('Mobility')
    with open(path + 'edge_matrix.pkl', 'rb') as f:
        data = pickle.load(f)
    with open(path +'node_idx_map.pkl', 'rb') as f:
        node_idx = pickle.load(f)
    with open(path +'idx_node_map.pkl', 'rb') as f:
        idx_node = pickle.load(f)
    datasets1 = MobilityDataset(node_list_path=node_list_path, is_train=True, threshold=0.99, data=data, node_idx_mapping=node_idx, idx_node_mapping=idx_node)

    
torch.autograd.set_detect_anomaly(True)
print(device)

best_metric=0
best_lr=-1
best_wr=-1

for i in learning_rate:
    for j in weight_decay:
        dataloaders_dict = DataLoader(datasets1, batch_size=batch_size,shuffle=True, num_workers=0)
        model = NodeEmbeddings(num_nodes, embedding_dim=200)
#         for param in model.parameters():
#             print(param.size())


        model = model.to(device)
        if pre_trained:
            checkpoint=torch.load(pre_trained)
            model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            print(str(checkpoint['model_state_dict']))
            print('Loaded pre-trained weights.')

        optimizer = optim.Adam(model.parameters(), lr=i, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=j, amsgrad=True)
        loss_fn = torch.nn.TripletMarginLoss(reduction="mean", margin=margin)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=lr_decay_rate)

        _, training_log,best_value = train_embedding(model, data_type=data_type, dataloaders=dataloaders_dict, criterion=loss_fn,
                               optimizer=optimizer, metrics=metrics, num_epochs=num_epochs,
                               save_dir=save_dir, verbose=True, return_best=return_best,
                               if_early_stop=if_early_stop, early_stop_epochs=early_stop_epochs, scheduler=scheduler,
                               save_epochs=save_epochs, current_epoch=current_epoch, device=device)
        
        
        print(training_log["metric_value_history"])

    if best_value>best_metric:
        best_metric=best_value
        best_lr=i
        best_wr=j

    with open(save_dir + 'metric_value_history.npy', 'wb') as f:
        np.save(f, training_log["metric_value_history"])
    with open(save_dir + 'loss_history.npy', 'wb') as f:
        np.save(f, training_log["loss_history"])

    best_epoch_loss = np.argmax(training_log["loss_history"])
    best_epoch_metric = np.argmax(training_log["metric_value_history"])
    with open(save_path, "a") as file:
#         for k in range(len(training_log["metric_value_history"])):
#             file.write("epoch:" + str(k)+"\n")
#             file.write("metric_value_history:"+str(training_log["metric_value_history"][k])+"\n")
#             file.write("loss_history:"+str(training_log["loss_history"][k])+"\n")

        file.write("\n\n---BEST---\nbest_lr:"+str(best_lr)+" best_wr:"+str(best_wr)+" best_metric_value:"+str(best_metric))

    print("best_lr:"+str(best_lr)+" best_wr:"+str(best_wr)+" best_metric_value:"+str(best_metric))

    model.print_history()
