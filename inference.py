import argparse
from tqdm import tqdm
from os.path import join, dirname

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import apex

from src.atlesconfig import config
from src.atlestrain import dataset, model
from src.atlespredict import dbsearch, filter


def run_inference(rank, world_size):
    batch_size  = config.get_config(section="ml", key="batch_size")
    prep_path = config.get_config(section='search', key='prep_path')

    spec_dataset = dataset.SpectraDataset(prep_path)
    spec_loader = torch.utils.data.DataLoader(
        dataset=spec_dataset, num_workers=0, collate_fn=spec_collate,
        batch_size=batch_size, shuffle=True
    )
    
    model_name = "512-embed-2-lstm-SnapLoss2D-80k-nist-massive-no-mc-semi-r2r2r-22.pt"
    print("Using model: {}".format(model_name))
    model_ = model.Net().to(rank)
    model_ = apex.parallel.DistributedDataParallel(model_, device_ids=[rank])
    model_.load_state_dict(torch.load('atles-out/15193090/deepatles-15193090-1nq92avm-250.pt')['model_state_dict'])
    model_ = model_.module
    model_.eval()
    print(model_)

    spec_filts = filter.runModel(spec_loader, model_, rank)

    


def spec_collate(batch):
    specs = torch.cat([item[0] for item in batch], 0)
    chars = torch.FloatTensor([item[1] for item in batch])
    return [specs, chars]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Adding optional argument 
    parser.add_argument("-c", "--config", help="Path to the config file.")
    parser.add_argument("-p", "--preprocess", help="Preprocess data?", default="True")
    
    # Read arguments from command line 
    input_params = parser.parse_args()

    if input_params.config:
        tqdm.write("config: %s" % input_params.path)
    config.PARAM_PATH = input_params.config if input_params.config else join((dirname(__file__)), "config.ini")

    num_gpus = torch.cuda.device_count()
    print("Num GPUs: {}".format(num_gpus))
    mp.spawn(run_inference, args=(2, input_params,), nprocs=2, join=True)
