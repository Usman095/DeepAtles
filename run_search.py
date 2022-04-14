import os
import argparse
from tqdm import tqdm
from os.path import join, dirname
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import apex

from src.atlesconfig import config
from src.atlestrain import dataset, model
from src.atlespredict import dbsearch, specdataset, pepdataset, preprocess, postprocess


def run_atles(rank, world_size, spec_loader):
    model_ = model.Net().to(rank)
    model_ = nn.parallel.DistributedDataParallel(model_, device_ids=[rank])
    model_.load_state_dict(torch.load('atles-out/15193090/models/deepatles-15193090-1nq92avm-250.pt')['model_state_dict'])
    model_ = model_.module
    model_.eval()
    print(model_)

    lens, cleavs, mods = dbsearch.runAtlesModel(spec_loader, model_, rank)

    return lens, cleavs, mods
    

def run_specollate_par(rank, world_size):
    setup(rank, world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    mgf_dir     = config.get_config(key="mgf_dir", section="search")
    prep_dir    = config.get_config(key="prep_dir", section="search")
    pep_dir     = config.get_config(key="pep_dir", section="search")
    out_pin_dir = config.get_config(key="out_pin_dir", section="search")

    # scratch_loc = "/scratch/mtari008/job_" + os.environ['SLURM_JOB_ID'] + "/"

    # mgf_dir     = scratch_loc + mgf_dir
    # prep_dir    = scratch_loc + prep_dir
    # pep_dir     = scratch_loc + pep_dir
    # out_pin_dir = scratch_loc + out_pin_dir

    if rank == 0:
        tqdm.write("Reading input files...")
    
    batch_size  = config.get_config(section="ml", key="batch_size")
    prep_path = config.get_config(section='search', key='prep_path')
    spec_batch_size = config.get_config(key="spec_batch_size", section="search")
    spec_dataset = specdataset.SpectraDataset(join(prep_path, "specs.pkl"))
    spec_loader = torch.utils.data.DataLoader(
        dataset=spec_dataset, batch_size=spec_batch_size,
        collate_fn=dbsearch.spec_collate)
    
    lens, cleavs, mods = run_atles(rank, 1, spec_loader)
    
    pep_batch_size  = config.get_config(key="pep_batch_size", section="search")
    if rank == 0:
        pep_dataset  = pepdataset.PeptideDataset(pep_dir)
        pep_loader = torch.utils.data.DataLoader(
            dataset=pep_dataset, batch_size=pep_batch_size,
            collate_fn=dbsearch.pep_collate)
    if rank == 1:
        dec_dataset  = pepdataset.PeptideDataset(pep_dir, decoy=True)
        dec_loader = torch.utils.data.DataLoader(
        dataset=dec_dataset, batch_size=pep_batch_size,
        collate_fn=dbsearch.pep_collate)

    dist.barrier()

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12350'
    # dist.init_process_group(backend='nccl', world_size=1, rank=0)
    # model_name = "512-embed-2-lstm-SnapLoss2D-80k-nist-massive-no-mc-semi-randbatch-62.pt" # 28.8k
    model_name = "512-embed-2-lstm-SnapLoss2D-80k-nist-massive-no-mc-semi-r2r-18.pt" # 28.975k
    model_name = "512-embed-2-lstm-SnapLoss2D-80k-nist-massive-no-mc-semi-r2r2r-22.pt"
    print("Using model: {}".format(model_name))
    snap_model = model.Net(vocab_size=30, embedding_dim=512, hidden_lstm_dim=512, lstm_layers=2).to(rank)
    snap_model = nn.parallel.DistributedDataParallel(snap_model, device_ids=[rank])
    # snap_model.load_state_dict(torch.load('models/32-embed-2-lstm-SnapLoss2-noch-3k-1k-152.pt')['model_state_dict'])
    # below one has 26975 identified peptides.
    # snap_model.load_state_dict(torch.load('models/512-embed-2-lstm-SnapLoss-noch-80k-nist-massive-52.pt')['model_state_dict'])
    # below one has 27.5k peps
    # snap_model.load_state_dict(torch.load('models/hcd/512-embed-2-lstm-SnapLoss2D-inputCharge-80k-nist-massive-116.pt')['model_state_dict'])
    snap_model.load_state_dict(torch.load('models/hcd/{}'.format(model_name))['model_state_dict'])
    snap_model = snap_model.module
    snap_model.eval()
    print(snap_model)

    print("Processing spectra...")
    e_specs = dbsearch.runSpeCollateModel(spec_loader, snap_model, "specs", rank)
    print("Spectra done!")

    dist.barrier()

    if rank == 0:
        print("Processing peptides...")
        e_peps = dbsearch.runSpeCollateModel(pep_loader, snap_model, "peps", rank)
        print("Peptides done!")

    if rank == 1:
        print("Processing decoys...")
        e_decs = dbsearch.runSpeCollateModel(dec_loader, snap_model, "peps", rank)
        print("Decoys done!")

    dist.barrier()

    min_pep_len = config.get_config(key="min_pep_len", section="ml")
    max_pep_len = config.get_config(key="max_pep_len", section="ml")
    max_clvs = config.get_config(key="max_clvs", section="ml")
    spec_dataset.filt_dict = defaultdict(list)
    for idx, l, clv, mod in enumerate(zip(lens, cleavs, mods)):
        if min_pep_len <= l <= max_pep_len and 0 <= clv <= max_clvs:
            key = '{}-{}-{}'.format(l, clv, int(mod))
            spec_dataset.filt_dict[key].append([idx, e_specs[idx], spec_dataset.masses[idx]]) #FIXME: needs to add actual spectra embeddings

    pep_batch_size  = config.get_config(key="pep_batch_size", section="search")
    pep_dataset  = pepdataset.PeptideDataset(pep_dir, decoy=rank==1)
    pep_dataset.filt_dict = defaultdict(list)
    for idx, pep, clv, mod in enumerate(zip(
        pep_dataset.pep_list, pep_dataset.missed_cleavs, pep_dataset.pep_modified_list)):
        pep_len = sum(map(str.isupper, pep))
        if min_pep_len <= pep_len <= max_pep_len and 0 <= clv <= max_clvs:
            key = '{}-{}-{}'.format(pep_len, clv, int(mod))
            pep_dataset.filt_dict[key].append([idx, e_peps[idx], pep_dataset.pep_mass_list[idx]]) #FIXME: needs to add actual peptide embeds

    search_spec_batch_size = config.get_config(key="search_spec_batch_size", section="search")
    # Run database search for each dict item
    g_inds = []
    g_vals = []
    for key, spec_subset in spec_dataset.filt_dict.items():
        if key not in pep_dataset: continue
        search_loader = torch.utils.data.DataLoader(
            dataset=spec_subset, num_workers=0, batch_size=search_spec_batch_size, shuffle=False)

        inds, vals = dbsearch.filtered_parallel_search(search_loader, pep_dataset.filt_dict[key], rank)
        g_inds.append(inds)
        g_vals.append(vals)
    inds = torch.cat(g_inds, 0)
    vals = torch.cat(g_vals, 0)

    dist.barrier()

    pin_charge = config.get_config(section="search", key="charge")
    charge_cols = [f"charge-{ch+1}" for ch in range(pin_charge)]
    cols = ["SpecId", "Label", "ScanNr", "SNAP", "ExpMass", "CalcMass", "deltCn", "deltLCn"] + charge_cols + ["dM", "absdM", "enzInt", "PepLen", "Peptide", "Proteins"]
    
    dist.barrier()

    if rank == 0:
        print("Generating percolator pin files...")
        global_out = postprocess.generate_percolator_input(inds, vals, pep_dataset, spec_dataset, "target")
        df = pd.DataFrame(global_out, columns=cols)
        df.sort_values(by="SNAP", inplace=True, ascending=False)
        df.to_csv(join(out_pin_dir, "target.pin"), sep="\t", index=False)

    if rank == 1:
        global_out = postprocess.generate_percolator_input(inds, vals, dec_dataset, spec_dataset, "decoy")
        df = pd.DataFrame(global_out, columns=cols)
        df.sort_values(by="SNAP", inplace=True, ascending=False)
        df.to_csv(join(out_pin_dir, "decoy.pin"), sep="\t", index=False)
        print("Wrote percolator files: \n{}\n{}".format(
            join(out_pin_dir, "target.pin"), join(out_pin_dir, "decoy.pin")))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)


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
    mp.spawn(run_specollate_par, args=(2,), nprocs=2, join=True)
