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


def plot_curves(loss, metric, best_epoch_loss, best_epoch_metric):
    plt.figure()
    plt.plot([epoch for epoch in range(0, len(loss))], loss, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train Stage: {data_type}')
    plt.axvline(x = best_epoch_loss, color = 'r', linestyle='-')
    plt.savefig(save_dir + 'loss')

    plt.figure()
    plt.plot([epoch for epoch in range(0, len(metric))], metric, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (of triplet learning)')
    plt.title(f'Train Stage: {data_type}')
    plt.axvline(x = best_epoch_metric, color = 'r', linestyle='-')
    plt.savefig(save_dir + 'accuracy')
    
def metrics(stats):
    """
    Self-defined metrics function to evaluate and compare models
    stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    accuracy = (stats['T'] + 0.00001) * 1.0 / (stats['T'] + stats['F'] + 0.00001)
    # print(accuracy)
    return accuracy

def train_embedding(model, data_type, dataloaders, criterion, optimizer, metrics, num_epochs,
                verbose=True, return_best=True, if_early_stop=True, early_stop_epochs=10, scheduler=None,
                save_dir=None, save_epochs=5, current_epoch=None):
    since = time.time()
    training_log = dict()
    training_log['loss_history'] = []
    training_log['metric_value_history'] = []
    training_log['current_epoch'] = -1
    if current_epoch is None:
        current_epoch = training_log['current_epoch'] + 1

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
    best_log = copy.deepcopy(training_log)

    best_metric_value = 0
    nodecrease = 0  # to count the epochs that val loss doesn't decrease
    early_stop = False

    for epoch in tqdm(range(current_epoch, num_epochs)):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        running_loss = 0.0
        stats = {'T':0,'F':0}

        # Iterate over data.
        for pos_index, pos_encoder_embedding, neg_encoder_embedding in dataloaders:
            pos_encoder_embedding = pos_encoder_embedding.to(device)
            neg_encoder_embedding = neg_encoder_embedding.to(device)
            pos_index = pos_index.to(device)
        
            if dtype == 'Edge':
                with torch.no_grad():
                    pos_encoder_embedding = model.return_embedding_by_idx(pos_encoder_embedding)
                    neg_encoder_embedding = model.return_embedding_by_idx(neg_encoder_embedding)
                
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model(pos_index)
                
                print(outputs.shape)
                print(pos_encoder_embedding.shape)
                print(neg_encoder_embedding.shape)

                loss = criterion(outputs, pos_encoder_embedding, neg_encoder_embedding) # anchor, pos, neg

                # evaluate based on whether dist(anchor, pos) < dist(anchor, neg)
                pos_dist = torch.nn.functional.pairwise_distance(outputs, pos_encoder_embedding, p=2.0) 
                neg_dist = torch.nn.functional.pairwise_distance(outputs, neg_encoder_embedding, p=2.0) 

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item()
            stats['T'] += torch.sum(pos_dist < neg_dist).cpu().item() # if pos distance < neg distance, this is good
            stats['F'] += torch.sum(pos_dist > neg_dist).cpu().item()

       
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_metric_value = metrics(stats)
        if verbose:
            print('Loss: {:.8f} Metrics: {:.5f}'.format( epoch_loss, epoch_metric_value))

        training_log['current_epoch'] = epoch
        training_log['metric_value_history'].append(epoch_metric_value)
        training_log['loss_history'].append(epoch_loss)
        
        
        with open(save_path, "a") as file:
            file.write("epoch:" + str(epoch)+"\n")
            file.write("metric_value_history:"+str(epoch_metric_value)+"\n")
            file.write("loss_history:"+str(epoch_loss)+"\n")

        
        if epoch_metric_value > best_metric_value:
            best_metric_value = epoch_metric_value
            best_model_wts = copy.deepcopy(model.state_dict())
            best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
            best_log = copy.deepcopy(training_log)
            nodecrease = 0
        else:
            nodecrease += 1
        if scheduler != None:
            scheduler.step()
        if nodecrease >= early_stop_epochs:
            early_stop = True
        if save_dir and epoch % save_epochs == 0:
            print(str(training_log['current_epoch']))
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_log': training_log
                }
            torch.save(checkpoint,os.path.join(save_dir, data_type + '_' + str(training_log['current_epoch']) + '.tar'))
        if if_early_stop and early_stop:
            print('Early stopped!')
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best metric value: {:5f}'.format(best_metric_value))

    # plot full training curve
    best_epoch_loss = np.argmin(training_log["loss_history"])
    best_epoch_metric = np.argmax(training_log["metric_value_history"])
    plot_curves(training_log["loss_history"], training_log["metric_value_history"], best_epoch_loss=best_epoch_loss, best_epoch_metric=best_epoch_metric)

    # load best model weights
    if return_best:
        model.load_state_dict(best_model_wts)
        optimizer.load_state_dict(best_optimizer_wts)
        training_log = best_log

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_log': training_log
    }

    if save_dir:
        torch.save(checkpoint,
                   os.path.join(save_dir, data_type + '_' + str(training_log['current_epoch']) + '_last.tar')) 
        print(f'Params: {checkpoint}')
    return model, training_log, best_metric_value




##########--------------------------- PARAMETERS -------------------------------#########
root = '/mnt/e/julia/regional-representations-graph-model/'
scenario = 'highres_landsat_experiment/'
current_epoch = None

search = root + f'train/outputs/{scenario}landsat/'
for f in os.listdir(search):
    if 'last.tar' in f:
        pre_trained = search + f
        break
    
# pre_trained=f'outputs/{scenario}mobility_all/landsat_78_last.tar' #'outputs/all_valid_data/landsat_all/landsat_21.tar' #None#'outputs/landsat_1000/landsat_25_last.tar'# last model save
stage = 'mobility/' #'mapillary_1000/' #'mobility_1000/' #'landsat_1000/' 

save_dir = 'outputs/' + scenario + stage 
data_type = 'mobility'#'distance' # 'mobility' #'poi' #'sv'
save_path = save_dir + "training_log_" + data_type + ".txt"
node_list_path= root + f'dataset/preprocessed_data/{scenario}node_list.csv'
dtype = 'Edge'


if not os.path.exists(save_dir):
    createCleanDir(save_dir)

with open(save_path, "w") as file:
    file.write('')
    
print('hello')
# device = "cpu"
embedding_dim = 200
num_nodes = pd.read_csv(node_list_path).shape[0]
print(f'Number of nodes: {num_nodes}')
return_best = True
if_early_stop = False
input_size = 500
learning_rate = [0.015]#[0.008]
weight_decay = [0.0005]
batch_size = num_nodes
num_epochs = 80
lr_decay_rate = 0.7
lr_decay_epochs = 6
early_stop_epochs = 14
save_epochs = 3
margin=2

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.cuda.current_device()


#######------------------------ TRAINING ------------------------------######

# #### LANDSAT #####
# landsat_model = ResNetTransform(device)
# datasets1 = SatelliteImageryDataset(node_list_path=node_list_path, 
#                                     root_image_dir='../dataset/earth_engine/download_landsat_images', 
#                                     fn='least_cloudy_rectangle_highres.tif', is_train=True, transform=landsat_model, 
#                                     load_embeddings=True)

# ##### STREET VIEW #####
# sv_model = Inceptionv3Transform(device)
# datasets1 = StreetViewDataset(node_list_path=node_list_path, 
#                                     root_image_dir='../dataset/mapillary/data/nyc_metro/', 
#                               is_train=True, transform=sv_model)

#### MOBILITY #####
path = f'/mnt/e/julia/regional-representations-graph-model/dataset/preprocessed_data/{scenario}'
print(path + 'edge_matrix.pkl')
with open(path + 'edge_matrix.pkl', 'rb') as f:
    data = pickle.load(f)
    
print(path + 'node_idx_map.pkl')
with open(path +'node_idx_map.pkl', 'rb') as f:
    node_idx = pickle.load(f)
    
with open(path +'idx_node_map.pkl', 'rb') as f:
    idx_node = pickle.load(f)
# datasets1 = MobilityDataset(node_list_path=node_list_path, is_train=True, threshold=0.85, data=data, node_idx_mapping=node_idx, idx_node_mapping=idx_node)

# #### DISTANCE #####
# path = '/mnt/e/julia/regional-representations-graph-model/dataset/preprocessed_data/all_valid_data/'
# with open(path+'distance_node_idx_map.pkl', 'rb') as f:
#     d1 = pickle.load(f)
# with open(path+'distance_idx_node_map.pkl', 'rb') as f:
#     d2 = pickle.load(f)
# with open(path + 'distance_edge_matrix.pkl', 'rb') as f:
#     data = pickle.load(f)
# datasets1 = DistanceDataset(node_list_path=node_list_path, is_train=True, threshold=None, data=data, node_idx_mapping=d1, idx_node_mapping=d2)
    
torch.autograd.set_detect_anomaly(True)
print(device)

best_metric=0
best_lr=-1
best_wr=-1

for i in learning_rate:
    for j in weight_decay:
        dataloaders_dict = DataLoader(datasets1, batch_size=batch_size,shuffle=True, num_workers=0)
        model = NodeEmbeddings(num_nodes, embedding_dim=200)
        for param in model.parameters():
            print(param.size())


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
                               save_epochs=save_epochs, current_epoch=current_epoch)
        
        
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
