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

from src.snapconfig import config
from src.snaputils import simulatespectra as sim


class SpectraDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir_path):
        'Initialization'
        
        in_path = Path(dir_path)
        assert in_path.exists()
        
        with open(dir_path, 'rb') as f:
            data = pickle.load(f)

        self.charge = config.get_config(section='input', key='charge')
        self.max_pep_len = config.get_config(section='ml', key='max_pep_len')

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
        
        print('dataset size: {}'.format(len(data)))
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.mzs)


    def __getitem__(self, index):
        'Generates one sample of data'
        max_spec_len = config.get_config(section='ml', key='max_spec_len')
        spec_mz = self.pad_right(self.mzs[index], max_spec_len)
        spec_intensity = self.pad_right(self.ints[index], max_spec_len)
        pep_len = self.lens[index]

        return spec_mz, spec_intensity, pep_len


    def pad_right(self, lst, max_len):
        lst_len = len(lst)
        zeros = [0] * (max_len - lst_len)
        return list(lst) + zeros

