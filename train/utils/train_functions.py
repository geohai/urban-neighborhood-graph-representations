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
root = '/mnt/e/julia/regional-representations-graph-model/'
sys.path.append(root + 'train/')
from models import *
sys.path.append(root + 'train/utils/')
from helper_funcs import *

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

# from torch.multiprocessing import Pool, Process, set_start_method
# set_start_method('spawn')

save_path = os.environ['TRAIN_SAVE_PATH']
dtype = os.environ['TRAIN_DTYPE']
data_type = os.environ['TRAIN_DATA_TYPE']
save_dir = os.environ['TRAIN_SAVE_DIR']

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
                save_dir=None, save_epochs=5, current_epoch=None, device='cpu'):
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
                print(f'Outputs: {outputs.shape}')
                print(f'Pos Embeddings: {pos_encoder_embedding.shape}')
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