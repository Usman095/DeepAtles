from os.path import join
from pathlib import Path
import re
import random
import pickle

import numpy as np
import torch
from torch._C import dtype
from torch.utils import data
from sklearn import preprocessing

from src.atlesconfig import config
from src.atlesutils import simulatespectra as sim


class SpectraDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir_path):
        'Initialization'
        
        in_path = Path(dir_path)
        assert in_path.exists()
        
        with open(dir_path, 'rb') as f:
            data = pickle.load(f)
        random.shuffle(data)

        parent_dir = in_path.parent.absolute()
        means = np.load(join(parent_dir, "means.npy"))
        stds = np.load(join(parent_dir, "stds.npy"))
        self.means = torch.from_numpy(means).float()
        self.stds = torch.from_numpy(stds).float()

        self.charge = config.get_config(section='input', key='charge')
        self.max_pep_len = config.get_config(section='ml', key='max_pep_len')
        self.min_pep_len = config.get_config(section='ml', key='min_pep_len')
        self.spec_size = config.get_config(section='input', key='spec_size')

        self.mzs = []
        self.ints = []
        self.lens = []
        self.charges = []
        self.is_mods = []
        self.miss_cleavs = []
        for spec_data in data:
            self.mzs.append(spec_data[0])
            self.ints.append(spec_data[1])
            self.lens.append(spec_data[2])
            self.charges.append(spec_data[3])
            self.is_mods.append(spec_data[4])
            self.miss_cleavs.append(spec_data[5])

        num_classes = self.max_pep_len - self.min_pep_len
        class_counts = [0] * num_classes
        for cla in self.lens:
            class_counts[cla] += 1
        class_weights = 1./torch.FloatTensor(class_counts)
        self.class_weights_all = class_weights[self.lens]
        
        print('dataset size: {}'.format(len(data)))
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.mzs)


    def __getitem__(self, index):
        'Generates one sample of data'
        max_spec_len = config.get_config(section='ml', key='max_spec_len')
        charge = config.get_config(section='input', key='charge')
        spec_mz = self.pad_right(self.mzs[index], max_spec_len)
        spec_intensity = self.pad_right(self.ints[index], max_spec_len)

        # ind = torch.LongTensor([[0]*len(spec_mz), spec_mz])
        # val = torch.FloatTensor(spec_intensity)
        
        # torch_spec = torch.sparse_coo_tensor(
        #     ind, val, torch.Size([1, self.spec_size])).to_dense()
        # torch_spec = (torch_spec - self.means) / self.stds

        pep_len = self.lens[index]
        ch_vec = [0] * charge
        l_ch = self.charges[index]
        for ch in range(l_ch):
            ch_vec[ch] = 1
        mod_vec = [0, 0]
        l_mod = self.is_mods[index]
        mod_vec[l_mod] = 1

        return spec_mz, spec_intensity, pep_len, ch_vec, mod_vec
        # return torch_spec, pep_len


    def pad_right(self, lst, max_len):
        lst_len = len(lst)
        zeros = [0] * (max_len - lst_len)
        return list(lst) + zeros

