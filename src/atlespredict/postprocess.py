import os
from os.path import join
import pandas as pd

import torch

from src.atlesconfig import config


def generate_percolator_input(l_pep_inds, l_pep_vals, l_spec_inds, pd_dataset, spec_charges, res_type):
    assert res_type == "target" or res_type == "decoy"
    assert len(l_pep_inds) == len(l_pep_vals) == len(l_spec_inds)
    pin_charge = config.get_config(section="search", key="charge")
    l_global_out = []
    tot_count = 0
    max_snap = torch.max(l_pep_vals).item()
    for l_spec_idx, pep_inds_row, pep_vals_row in zip(l_spec_inds, l_pep_inds, l_pep_vals):
        l_spec_idx = l_spec_idx.item()
        # Reminder: pep_inds_row length is one less than pep_vals_row
        for iidx in range(len(pep_inds_row) - 1):
            pep_ind = pep_inds_row[iidx]
            pep_val = pep_vals_row[iidx]
            if pep_val.item() > 0:
                charge = [0] * pin_charge
                ch_idx = min(spec_charges[l_spec_idx], pin_charge)
                charge[ch_idx - 1] = 1
                label = 1 if res_type == "target" else -1
                out_row = [f"{res_type}-{tot_count}", label, l_spec_idx, pep_val.item()]
                spec_mass = spec_charges[l_spec_idx]
                pep_mass = pd_dataset.pep_mass_list[pep_ind.item()]
                out_row.append(spec_mass)
                out_row.append(pep_mass)
#                 out_row.append(((pep_val - pep_vals_row[iidx + 1]).item()) / max_snap)
#                 out_row.append(((pep_val - pep_vals_row[-1]).item()) / max_snap)
                out_row.append(((pep_val - pep_vals_row[-1]).item()) / max(pep_val.item(), 1.0))
                out_row.append(((pep_val - pep_vals_row[iidx + 1]).item()) / max(pep_val.item(), 1.0))
                out_row.extend(charge)
                out_row.append(spec_mass - pep_mass)
                out_row.append(abs(spec_mass - pep_mass))

                out_pep = pd_dataset.pep_list[pep_ind.item()]
                out_pep_array = []
                for aa in out_pep:
                    if aa.islower():
                        out_pep_array.append("[" + str(config.AAMass[aa]) + "]")
                    else:
                        out_pep_array.append(aa)
                out_pep = "".join(out_pep_array)
                out_row.append((out_pep.count("K") + out_pep.count("R")
                                ) - (out_pep.count("KP") + out_pep.count("RP")) - 1)
                out_prot = pd_dataset.prot_list[pep_ind.item()]
                pep_len = sum([a.isupper() for a in out_pep])
                out_row.append(pep_len)
                out_row.append(out_pep)
                out_row.append(out_prot)
                l_global_out.append(out_row)
                tot_count += 1
    return l_global_out


def write_to_pin(rank, pep_inds, psm_vals, spec_inds, l_pep_dataset, spec_charges, cols, out_pin_dir):
    os.makedirs(out_pin_dir, exist_ok=True)
    if rank == 0:
        print("Generating percolator pin files...")
    global_out = generate_percolator_input(
        pep_inds,
        psm_vals,
        spec_inds,
        l_pep_dataset,
        spec_charges,
        "target" if rank == 0 else "decoy",
    )
    df = pd.DataFrame(global_out, columns=cols)
    df.sort_values(by="SNAP", inplace=True, ascending=False)
    with open(join(out_pin_dir, "target.pin" if rank == 0 else "decoy.pin"), "a") as f:
        df.to_csv(f, sep="\t", index=False, header=not f.tell())

    if rank == 0:
        print("Wrote percolator files: ")
    # dist.barrier()
    print("{}".format(join(out_pin_dir, "target.pin") if rank == 0 else join(out_pin_dir, "decoy.pin")))