from dataclasses import dataclass
import read_spectra

import h5py
import re
from pathlib import Path, PurePath
import os
import argparse
from os.path import join, dirname
import time

import progressbar
import torch
from torch.utils import data
from tqdm import tqdm
import itertools
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from collections import defaultdict
import pickle
import numpy as np

from src.atlesconfig import config
from src.atlesutils import simulatespectra as sim
from src.atlespredict import (dbsearch, pepdataset, postprocess, preprocess, specdataset, specollate_model)
from src.atlestrain import dataset, model


out_pin_dir = config.get_config(key="out_pin_dir", section="search")
pep_dir = config.get_config(key="pep_dir", section="search")
pep_index_name = PurePath(pep_dir).name
print(pep_index_name)
index_path = join(config.get_config(key="index_path", section="search"), pep_index_name)
min_pep_len = config.get_config(key="min_pep_len", section="ml")
max_pep_len = config.get_config(key="max_pep_len", section="ml")
max_clvs = config.get_config(key="max_clvs", section="ml")

length_filter = config.get_config(key="length_filter", section="filter")
len_tol_pos = config.get_config(key="len_tol_pos", section="filter") if length_filter else 0
len_tol_neg = config.get_config(key="len_tol_neg", section="filter") if length_filter else 0
missed_cleavages_filter = config.get_config(key="missed_cleavages_filter", section="filter")
modification_filter = config.get_config(key="modification_filter", section="filter")

pep_batch_size = config.get_config(key="pep_batch_size", section="search")


@dataclass
class PepInfo:
    pep_list: list
    prot_list: list
    pep_mass_list: list


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(config.get_config(key="master_port", section="input"))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)


# check if preprocessed folder exisits.
# if not do step: 1 - 2
# do step 3 - 4
# Step 1
def classify_peptides():
    pep_dataset = pepdataset.PeptideDataset(pep_dir, decoy=False)
    pep_classes_path = join(index_path, 'peptide_classes')
    os.mkdir(index_path)

    os.mkdir(pep_classes_path)
    os.mkdir(join(index_path, 'peptide_embeddings'))
    os.mkdir(join(index_path, 'decoy_embeddings'))

    # 1 - classify peptides and write to 144 separate files.
    print('Opening files')
    open_files = {}
    class_offsets = {}
    for length, clv, mod in itertools.product(range(min_pep_len, max_pep_len + 1), range(max_clvs + 1), range(2)):
        file_name = '{}-{}-{}'.format(length, clv, mod)
        open_files[file_name] = open(join(pep_classes_path, file_name), 'a')
        class_offsets[file_name] = 0

    print('Classifying peptides and writing to files')
    for idx, (pep, clv, mod, prot) in enumerate(zip(
            pep_dataset.pep_list, pep_dataset.missed_cleavs, pep_dataset.pep_modified_list, pep_dataset.prot_list)):
        pep_len = sum(map(str.isupper, pep))
        if min_pep_len <= pep_len <= max_pep_len and 0 <= clv <= max_clvs:
            file_name = '{}-{}-{}'.format(int(pep_len), int(clv), int(mod))
            if file_name in open_files:
                f = open_files[file_name]
                f.write('>' + prot + '\n')
                f.write(pep + '\n')
                class_offsets[file_name] += 1

    print('Closing files')
    for _, f in open_files.items():
        f.close()
    
    cum = 0
    for length, clv, mod in itertools.product(range(min_pep_len, max_pep_len + 1), range(max_clvs + 1), range(2)):
        file_name = '{}-{}-{}'.format(length, clv, mod)
        offset = class_offsets[file_name]
        class_offsets[file_name] = cum
        cum += offset

    # need to get decoy offset later
    pickle.dump(class_offsets, open(join(index_path, '{}'.format('peptide_class_offsets.pkl')), 'wb'))


def get_snap_model(rank):
    model_name = config.get_config(key="model_name", section="search")
    print("Using model: {}".format(model_name))
    snap_model = specollate_model.Net(vocab_size=30, embedding_dim=512, hidden_lstm_dim=512, lstm_layers=2).to(rank)
    snap_model = nn.parallel.DistributedDataParallel(snap_model, device_ids=[rank])
    # snap_model.load_state_dict(torch.load('models/32-embed-2-lstm-SnapLoss2-noch-3k-1k-152.pt')['model_state_dict'])
    # below one has 26975 identified peptides.
    # snap_model.load_state_dict(torch.load('models/512-embed-2-lstm-SnapLoss-noch-80k-nist-massive-52.pt')['model_state_dict'])
    # below one has 27.5k peps
    # snap_model.load_state_dict(torch.load('models/hcd/512-embed-2-lstm-SnapLoss2D-inputCharge-80k-nist-massive-116.pt')['model_state_dict'])
    snap_model.load_state_dict(torch.load('specollate-model/{}'.format(model_name))['model_state_dict'])
    snap_model = snap_model.module
    snap_model.eval()
    print(snap_model)
    return snap_model


# 2 - load each class and process it using specollate, save embeddings for each class separately
def process_peptides(rank, snap_model):
    class_offsets = {}
    cum = 0
    for length, clv, mod in itertools.product(range(min_pep_len, max_pep_len + 1), range(max_clvs + 1), range(2)):
        file_name = '{}-{}-{}'.format(length, clv, mod)
        class_offsets[file_name] = cum
        pep_classes_path = join(index_path, 'peptide_classes')
        pep_file_path = join(pep_classes_path, file_name)
        if os.path.exists(pep_file_path):
            if not os.path.getsize(pep_file_path):
                os.remove(pep_file_path)
                continue
            print('Processing file: {}'.format(file_name))
            # process peptides
            pep_dataset = pepdataset.PeptideDataset(pep_dir, pep_file_path, decoy=rank == 1)
            cum += len(pep_dataset)
            pep_loader = torch.utils.data.DataLoader(
                dataset=pep_dataset, batch_size=pep_batch_size,
                collate_fn=dbsearch.pep_collate)
            
            print("Processing {}...".format("Peptides" if rank == 0 else "Decoys"))
            e_peps = dbsearch.runSpeCollateModel(pep_loader, snap_model, "peps", rank)
            print("Peptides done!")

            # save embeddings
            print('Saving embeddings at {}'.format(join(index_path, '{}'.format(
                'peptide_embeddings' if rank == 0 else 'decoy_embeddings'), file_name)))
            torch.save(e_peps, join(index_path, '{}'.format(
                'peptide_embeddings' if rank == 0 else 'decoy_embeddings'), file_name))
            print('Done \n')

    pickle.dump(class_offsets, open(join(index_path, '{}'.format(
        'peptide_class_offsets.pkl' if rank == 0 else 'decoy_class_offsets.pkl')), 'wb'))


def run_atles(rank, world_size, spec_loader):
    model_ = model.Net().to(rank)
    model_ = nn.parallel.DistributedDataParallel(model_, device_ids=[rank])
    # model_.load_state_dict(torch.load('atles-out/16403437/models/pt-mass-ch-16403437-1toz70vi-472.pt')['model_state_dict'])
    # model_.load_state_dict(torch.load(
    #     '/lclhome/mtari008/DeepAtles/atles-out/123/models/pt-mass-ch-123-2zgb2ei9-385.pt'
    #     )['model_state_dict'])
    model_.load_state_dict(torch.load(
        '/lclhome/mtari008/DeepAtles/atles-out/1382/models/nist-massive-deepnovo-mass-ch-1382-c8mlqbq7-157.pt'
    )['model_state_dict'])
    model_ = model_.module
    model_.eval()
    print(model_)

    lens, cleavs, mods = dbsearch.runAtlesModel(spec_loader, model_, rank)
    pred_cleavs_softmax = torch.log_softmax(cleavs, dim=1)
    _, pred_cleavs = torch.max(pred_cleavs_softmax, dim=1)
    pred_mods_softmax = torch.log_softmax(mods, dim=1)
    _, pred_mods = torch.max(pred_mods_softmax, dim=1)

    return (
        torch.round(lens).type(torch.IntTensor).squeeze().tolist(),
        pred_cleavs.squeeze().tolist(),
        pred_mods.squeeze().tolist()
    )


def process_spectra(rank, snap_model):
    prep_path = config.get_config(section='search', key='prep_path')
    spec_batch_size = config.get_config(key="spec_batch_size", section="search")
    spec_dataset = specdataset.SpectraDataset(join(prep_path, "specs.pkl"))
    spec_loader = torch.utils.data.DataLoader(
        dataset=spec_dataset, batch_size=spec_batch_size,
        collate_fn=dbsearch.spec_collate)

    print("Processing spectra...")
    e_specs = dbsearch.runSpeCollateModel(spec_loader, snap_model, "specs", rank)
    print("Spectra done!")

    atles_start_time = time.time()
    lens, cleavs, mods = run_atles(rank, 1, spec_loader)
    atles_end_time = time.time()
    atles_time = atles_end_time - atles_start_time
    print("Atles time: {}".format(atles_time))
    return e_specs, lens, cleavs, mods, spec_dataset.masses, spec_dataset.charges


# 3 - Loop over spectra classes, load embeddings for peptides, peform db search
def create_spectra_dict(lens, cleavs, mods, e_specs, spec_masses):
    print("Creating spectra filtered dictionary.")
    spec_filt_dict = defaultdict(list)
    for idx, (l, clv, mod) in enumerate(zip(lens, cleavs, mods)):
        if min_pep_len <= l <= max_pep_len and 0 <= clv <= max_clvs:
            key = '{}-{}-{}'.format(int(l), int(clv), int(mod))
            # FIXME: needs to add actual spectra embeddings
            spec_filt_dict[key].append([idx, e_specs[idx], spec_masses[idx]])

    return spec_filt_dict


def write_to_pin(rank, pep_inds, psm_vals, spec_inds, l_pep_dataset, spec_charges, cols):
    if rank == 0:
        print("Generating percolator pin files...")
    global_out = postprocess.generate_percolator_input(
        pep_inds, psm_vals, spec_inds, l_pep_dataset, spec_charges, "target" if rank == 0 else "decoy")
    df = pd.DataFrame(global_out, columns=cols)
    df.sort_values(by="SNAP", inplace=True, ascending=False)
    with open(join(out_pin_dir, "target.pin" if rank == 0 else "decoy.pin"), 'a') as f:
        df.to_csv(f, sep="\t", index=False, header=not f.tell())

    if rank == 0:
        print("Wrote percolator files: ")
    # dist.barrier()
    print("{}".format(join(out_pin_dir, "target.pin") if rank == 0 else join(out_pin_dir, "decoy.pin")))


def search_database(rank, spec_filt_dict, spec_charges):
    search_spec_batch_size = config.get_config(key="search_spec_batch_size", section="search")
    # dist.barrier()

    if rank == 0:
        search_start_time = time.time()
    # Run database search for each dict item
    unfiltered_time = 0

    class_offsets = pickle.load(
        open(join(index_path, '{}'.format(
            'peptide_class_offsets.pkl' if rank == 0 else 'decoy_class_offsets.pkl')), 'rb'))

    pin_charge = config.get_config(section="search", key="charge")
    charge_cols = [f"charge-{ch+1}" for ch in range(pin_charge)]
    cols = ["SpecId", "Label", "ScanNr", "SNAP", "ExpMass", "CalcMass", "deltCn",
            "deltLCn"] + charge_cols + ["dM", "absdM", "enzInt", "PepLen", "Peptide", "Proteins"]
            
    print("Running filtered {} database search.".format("target" if rank == 0 else "decoy"))
    for key in spec_filt_dict:
        print('Searching for key {}.'.format(key))
        spec_inds = []
        pep_inds = []
        psm_vals = []
        pep_info = PepInfo([], [], [])
        for tol in range(len_tol_neg, len_tol_pos + 1):
            key_len, key_clv, key_mod = int(key.split('-')[0]), int(key.split('-')[1]), int(key.split('-')[2])
            file_name = '{}-{}-{}'.format(key_len + tol, key_clv, key_mod)
            pep_classes_path = join(index_path, 'peptide_classes')
            pep_file_path = join(pep_classes_path, file_name)
            if not os.path.exists(pep_file_path):
                print("Key {} not found in pep_dataset".format(pep_file_path))
                continue
            print('Processing file: {}'.format(file_name))
            # process peptides
            pep_dataset = pepdataset.PeptideDataset(pep_dir, pep_file_path, decoy=rank == 1)
            # pep_loader = torch.utils.data.DataLoader(
            #     dataset=pep_dataset, batch_size=pep_batch_size,
            #     collate_fn=dbsearch.pep_collate)
            # pep_info.pep_list += pep_dataset.pep_list
            # pep_info.prot_list += pep_dataset.prot_list
            # pep_info.pep_mass_list += pep_dataset.pep_mass_list

            # load embeddings
            pep_embeddings_path = join(index_path, 'peptide_embeddings' if rank == 0 else 'decoy_embeddings')
            embedding_file_path = join(pep_embeddings_path, file_name)
            e_peps = torch.load(embedding_file_path)
            # pep_data = [[idx + class_offsets[file_name], e_pep, mass] \
            #     for idx, (e_pep, mass) in enumerate(zip(e_peps, pep_dataset.pep_mass_list))]
            pep_data = [[idx + class_offsets[file_name], e_pep, mass]
                        for idx, (e_pep, mass) in enumerate(zip(e_peps, pep_dataset.pep_mass_list))]

            print("Searching against key {} with {} peptides.".format(file_name, len(pep_dataset.pep_mass_list)))
            spec_subset = spec_filt_dict[key]
            search_loader = torch.utils.data.DataLoader(
                dataset=spec_subset, num_workers=0, batch_size=search_spec_batch_size, shuffle=False)
            pep_dataset = None
            unfiltered_start_time = time.time()
            l_spec_inds, l_pep_inds, l_psm_vals = dbsearch.filtered_parallel_search(
                search_loader, pep_data, rank)
            unfiltered_time += time.time() - unfiltered_start_time

            if not l_spec_inds:
                continue
            # spec_inds.extend(l_spec_inds)
            # pep_inds.append(l_pep_inds)
            # psm_vals.append(l_psm_vals)

        # if not l_spec_inds:
        #     continue
        # spec_inds.extend(l_spec_inds)
        # pep_inds.append(l_pep_inds)
        # psm_vals.append(l_psm_vals)

        # pep_inds = torch.cat(pep_inds, 0)
        # psm_vals = torch.cat(psm_vals, 0)

        print("{} PSMS: {}".format("Target" if rank == 0 else "Decoy", len(pep_inds)))

        # 4 - Write PSMs to pin file
        # write_to_pin(rank, pep_inds, psm_vals, spec_inds, pep_info, spec_charges, cols)


def run_atles_search(rank, world_size):
    setup(rank, world_size)
    snap_model = get_snap_model(rank)
    if not os.path.exists(index_path):
        if rank == 0:
            classify_peptides()
        dist.barrier()
        process_peptides(rank, snap_model)

    t_time = time.time()
    e_specs, lens, cleavs, mods, spec_masses, spec_charges = process_spectra(rank, snap_model)
    spec_filt_dict = create_spectra_dict(lens, cleavs, mods, e_specs, spec_masses)
    start_time = time.time()
    search_database(rank, spec_filt_dict, spec_charges)
    print("Search time: {}".format(time.time() - start_time))
    print("Total time: {}".format(time.time() - t_time))


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
    start_time = time.time()
    mp.spawn(run_atles_search, args=(2,), nprocs=2, join=True)
    # run_atles_search(0, 1)
    print("Total time: {}".format(time.time() - start_time))