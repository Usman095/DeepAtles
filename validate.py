import argparse
import os
import timeit
import shutil
from os.path import join
import re
import pickle
import random as rand
import numpy as np

import torch
from torch._C import dtype
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.functional import dropout
import torch.optim as optim
from torch.utils.data import BatchSampler
from sklearn.model_selection import train_test_split
import apex
from torch.utils.data.sampler import WeightedRandomSampler
import wandb
torch.manual_seed(1)

from src.atlesconfig import config, wandbsetup
from src.atlestrain import dataset, model, trainmodel, sampler
from src.atlesutils import simulatespectra as sim
import run_train as main

# with redirect_output("deepSNAP_redirect.txtS"):
train_loss     = []
test_loss      = []
train_accuracy = []
test_accuracy  = []

def run_par(rank, world_size):
    model_name = "deepatles" #first k is spec size second is batch size
    print("Validating {}.".format(model_name))
    
    wandb.init(mode="disabled")
    wandb.run.name = "{}-{}-{}".format(model_name, os.environ['SLURM_JOB_ID'], wandb.run.id)
    wandbsetup.set_wandb_config(wandb)

    main.setup(rank, world_size)

    batch_size  = config.get_config(section="ml", key="batch_size")
    val_dir = config.get_config(section='input', key='val_dir')

    val_dataset = dataset.SpectraDataset(join(val_dir, 'val_specs.pkl'))

    weights_all = val_dataset.class_weights_all
    weighted_sampler = WeightedRandomSampler(weights=weights_all, num_samples=len(weights_all), replacement=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, num_workers=0, collate_fn=main.psm_collate,
        batch_size=batch_size, shuffle=False
    )

    lr = config.get_config(section='ml', key='lr')
    num_epochs = config.get_config(section='ml', key='epochs')
    weight_decay = config.get_config(section='ml', key='weight_decay')
    embedding_dim = config.get_config(section='ml', key='embedding_dim')
    encoder_layers = config.get_config(section='ml', key='encoder_layers')
    num_heads = config.get_config(section='ml', key='num_heads')
    dropout = config.get_config(section='ml', key='dropout')

    if rank == 0:
        print("Batch Size: {}".format(batch_size))
        print("Learning Rate: {}".format(lr))
        print("Weigh Decay: {}".format(weight_decay))
        print("Embedding Dim: {}".format(embedding_dim))
        print("Encoder Layers: {}".format(encoder_layers))
        print("Heads: {}".format(num_heads))
        print("Dropout: {}".format(dropout))

    ce_loss = nn.CrossEntropyLoss(reduction="mean")
    mse_loss = nn.MSELoss(reduction="mean")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    model_ = model.Net().to(rank)
    model_ = apex.parallel.DistributedDataParallel(model_)
    # model_.load_state_dict(torch.load("atles-out/15193090/models/deepatles-15193090-1nq92avm-447.pt")["model_state_dict"])
    model_.load_state_dict(torch.load("atles-out/16403437/models/pt-mass-ch-16403437-1toz70vi-472.pt")["model_state_dict"])

    start_time = timeit.default_timer()
    
    _, pred_lens, labl_lens, pred_cleavs, labl_cleavs, pred_mods, labl_mods =\
         trainmodel.test(model_, rank, val_loader, mse_loss, ce_loss, 0)
    
    pred_lens, labl_lens = torch.round(pred_lens), torch.round(labl_lens)
    pred_cleavs, labl_cleavs = multi_class(pred_cleavs, labl_cleavs)
    pred_mods, labl_mods = multi_class(pred_mods, labl_mods)

    torch.save(pred_lens, "pred_lens.pt")
    torch.save(labl_lens, "labl_lens.pt")
    torch.save(pred_cleavs, "pred_cleavs.pt")
    torch.save(labl_cleavs, "labl_cleavs.pt")
    torch.save(pred_mods, "pred_mods.pt")
    torch.save(labl_mods, "labl_mods.pt")
    print(pred_lens.size())
    print(labl_lens.size())
    print(pred_cleavs.size())
    print(labl_cleavs.size())
    print(pred_mods.size())
    print(labl_mods.size())

    elapsed = timeit.default_timer() - start_time
    print("time takes: {} secs.".format(elapsed))

    dist.barrier()
    main.cleanup()


def multi_class(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    return y_pred_tags, y_test


def round_lens(y_pred, y_test, err=0):
    return torch.round(y_pred), torch.round(y_test)


if __name__ == '__main__':

    # Initialize parser 
    parser = argparse.ArgumentParser()
    
    # Adding optional argument 
    parser.add_argument("-j", "--job-id", help="No arguments should be passed. \
        Instead use the shell script provided with the code.") 
    parser.add_argument("-p", "--path", help="Path to the config file.")
    parser.add_argument("-s", "--server-name", help="Which server the code is running on. \
        Options: raptor, comet. Default: comet", default="comet")
    
    # Read arguments from command line 
    args = parser.parse_args() 
    
    if args.job_id: 
        print("job_id: %s" % args.job_id)
        job_id = args.job_id

    if args.path:
        print("job_id: %s" % args.path)
        scratch = args.path

    

    mp.set_start_method('forkserver')
    config.PARAM_PATH = join((os.path.dirname(__file__)), "config.ini")
    
    do_learn = True
    save_frequency = 2
    
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)

    num_gpus = torch.cuda.device_count()
    print("Num GPUs: {}".format(num_gpus))
    # mp.spawn(run_par, args=(num_gpus,), nprocs=num_gpus, join=True)
    run_par(0, 1)

    # model.linear1_1.weight.requires_grad = False
    # model.linear1_1.bias.requires_grad = False
    # model.linear1_2.weight.requires_grad = False
    # model.linear1_2.bias.requires_grad = False