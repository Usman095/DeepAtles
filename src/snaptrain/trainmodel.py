import random as rand
import sys
import timeit
#from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import progressbar
from apex import amp
import wandb
from tqdm import tqdm

from src.snapconfig import config
from src.snaptrain import process

rand.seed(37)

train_accuracy = []
train_loss = []
test_accuracy = []
test_loss = []
snp_weight = config.get_config(section="ml", key="snp_weight")
ce_weight = config.get_config(section="ml", key="ce_weight")
mse_weight = config.get_config(section="ml", key="mse_weight")
divider = snp_weight + ce_weight# + mse_weight


def train(model, device, train_loader, cross_entropy_loss, optimizer, epoch):
    model.train()

    accurate_labels = 0
    all_labels = 0
    tot_loss = 0
    
    # pbar = tqdm(train_loader, file=sys.stdout)
    # pbar.set_description('Training...')
    for data in train_loader:
        data[0] = data[0].to(device) # mzs
        data[1] = data[1].to(device) # ints
        data[2] = data[2].to(device) # lens
        
        optimizer.zero_grad()

        input_mask = data[0] == 0
        lens = model(data[0], input_mask)
        
        loss = cross_entropy_loss(lens, data[2])
        tot_loss += float(loss)
        
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        optimizer.step()

        accurate_labels += multi_acc(lens, data[2])
        all_labels += len(lens)
        # p_bar.update(idx)
    
    accuracy = 100. * float(accurate_labels) / all_labels
    print('Train accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, tot_loss))
    return loss
    

def test(model, device, test_loader, cross_entropy_loss, epoch):
    model.eval()
    
    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        tot_loss = 0
        # with progressbar.ProgressBar(max_value=len(train_loader)) as p_bar:
        for data in test_loader:

            data[0] = data[0].to(device) # mzs
            data[1] = data[1].to(device) # ints
            data[2] = data[2].to(device) # lens

            input_mask = data[0] == 0
            lens = model(data[0], input_mask)
            
            loss = cross_entropy_loss(lens, data[2])
            tot_loss += float(loss)
            
            accurate_labels += multi_acc(lens, data[2])
            all_labels += len(lens)
                
        accuracy = 100. * float(accurate_labels) / all_labels
        print('Test accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, tot_loss))
        return loss


# TODO: change it. taken from 
# https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
# accessed: 09/18/2020
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float().sum()
    
    return correct_pred