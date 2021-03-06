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

from src.atlesconfig import config
from src.atlestrain import process

rand.seed(37)

train_accuracy = []
train_loss = []
test_accuracy = []
test_loss = []
snp_weight = config.get_config(section="ml", key="snp_weight")
ce_weight = config.get_config(section="ml", key="ce_weight")
mse_weight = config.get_config(section="ml", key="mse_weight")
divider = snp_weight + ce_weight# + mse_weight


def train(model, device, train_loader, mse_loss, ce_loss, optimizer, epoch):
    model.train()

    accurate_cleavs = 0
    accurate_labels_0 = accurate_labels_1 = accurate_labels_2 = 0
    all_labels = 0
    tot_loss = 0
    
    # pbar = tqdm(train_loader, file=sys.stdout)
    # pbar.set_description('Training...')
    for data in train_loader:
        data[0] = data[0].to(device) # m/z's
        data[1] = data[1].to(device) # intensities
        data[2] = data[2].to(device) # peptide lengths
        data[3] = data[3].to(device) # charges
        data[4] = data[4].to(device) # modifications
        data[5] = data[5].to(device) # missed cleavages

        optimizer.zero_grad()

        input_mask = data[0] == 0
        # input_mask = 0
        lens, cleavs = model(data[0], data[1], data[3], input_mask)
        lens = lens.squeeze()
        # cleavs = cleavs.squeeze()
        # print(len(cleavs))
        # print(torch.min(data[5]), torch.max(data[5]))
        mse_loss_val = mse_loss(lens, data[2])
        ce_loss_val = ce_loss(cleavs, data[5])
        loss_lst = [mse_loss_val, ce_loss_val]
        loss = sum(loss_lst) / len(loss_lst)
        # loss = mse_loss_val
        tot_loss += float(loss)
        
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        optimizer.step()

        # accurate_labels += multi_acc(lens, data[2])
        accurate_labels_0 += mse_acc(lens, data[2], err=0)
        accurate_labels_1 += mse_acc(lens, data[2], err=1)
        accurate_labels_2 += mse_acc(lens, data[2], err=2)
        accurate_cleavs += multi_acc(cleavs, data[5])
        all_labels += len(lens)
        # p_bar.update(idx)
    
    # accuracy = 100. * float(accurate_labels) / all_labels
    accuracy_0 = 100. * float(accurate_labels_0) / all_labels
    accuracy_1 = 100. * float(accurate_labels_1) / all_labels
    accuracy_2 = 100. * float(accurate_labels_2) / all_labels
    accuracy_cleavs = 100. * float(accurate_cleavs) / all_labels
    wandb.log({"Train loss": tot_loss}, step=epoch)
    wandb.log({"Train Accuracy Margin-0": accuracy_0}, step=epoch)
    wandb.log({"Train Accuracy Margin-1": accuracy_1}, step=epoch)
    wandb.log({"Train Accuracy Margin-2": accuracy_2}, step=epoch)
    wandb.log({"Train Accuracy Missed Cleavages": accuracy_cleavs}, step=epoch)
    print('Train accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_0, all_labels, accuracy_0, tot_loss/all_labels))
    print('Train accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_1, all_labels, accuracy_1, tot_loss/all_labels))
    print('Train accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_2, all_labels, accuracy_2, tot_loss/all_labels))
    print('Train accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_cleavs, all_labels, accuracy_cleavs, tot_loss/all_labels))
    return loss
    

def test(model, device, test_loader, mse_loss, ce_loss, epoch):
    model.eval()
    
    with torch.no_grad():
        accurate_cleavs = 0
        accurate_labels_0 = accurate_labels_1 = accurate_labels_2 = 0
        all_labels = 0
        tot_loss = 0
        # with progressbar.ProgressBar(max_value=len(train_loader)) as p_bar:
        for data in test_loader:
            data[0] = data[0].to(device) # m/z's
            data[1] = data[1].to(device) # intensities
            data[2] = data[2].to(device) # peptide lengths
            data[3] = data[3].to(device) # charges
            data[4] = data[4].to(device) # modifications
            data[5] = data[5].to(device) # missed cleavages

            input_mask = data[0] == 0
            # input_mask = 0
            lens, cleavs = model(data[0], data[1], data[3], input_mask)
            lens = lens.squeeze()
            # cleavs = cleavs.squeeze()
            mse_loss_val = mse_loss(lens, data[2])
            ce_loss_val = ce_loss(cleavs, data[5])
            loss_lst = [mse_loss_val, ce_loss_val]
            loss = sum(loss_lst) / len(loss_lst)
            tot_loss += float(loss)
            
            # accurate_labels += multi_acc(lens, data[2])
            accurate_labels_0 += mse_acc(lens, data[2], err=0)
            accurate_labels_1 += mse_acc(lens, data[2], err=1)
            accurate_labels_2 += mse_acc(lens, data[2], err=2)
            accurate_cleavs += multi_acc(cleavs, data[5])
            all_labels += len(lens)
                
        # accuracy = 100. * float(accurate_labels) / all_labels
        accuracy_0 = 100. * float(accurate_labels_0) / all_labels
        accuracy_1 = 100. * float(accurate_labels_1) / all_labels
        accuracy_2 = 100. * float(accurate_labels_2) / all_labels
        accuracy_cleavs = 100. * float(accurate_cleavs) / all_labels
        wandb.log({"Test loss": tot_loss}, step=epoch)
        wandb.log({"Test Accuracy Margin-0": accuracy_0}, step=epoch)
        wandb.log({"Test Accuracy Margin-1": accuracy_1}, step=epoch)
        wandb.log({"Test Accuracy Margin-2": accuracy_2}, step=epoch)
        wandb.log({"Test Accuracy Missed Cleavages": accuracy_cleavs}, step=epoch)
        print('Test accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_0, all_labels, accuracy_0, tot_loss/all_labels))
        print('Test accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_1, all_labels, accuracy_1, tot_loss/all_labels))
        print('Test accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_2, all_labels, accuracy_2, tot_loss/all_labels))
        print('Test accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_cleavs, all_labels, accuracy_cleavs, tot_loss/all_labels))
        return loss


# TODO: change it. taken from 
# https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
# accessed: 09/18/2020
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float().sum()
    
    return correct_pred


def mse_acc(y_pred, y_test, err=0):
    correct_pred = (torch.abs( torch.round(y_pred) - torch.round(y_test) ) <= err).float().sum()
    
    return correct_pred