import os
from os.path import join
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.snapconfig import config

#adding useless comment
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.spec_size = config.get_config(section='input', key='spec_size')
        self.charge = config.get_config(section='input', key='charge')
        self.max_spec_len = config.get_config(section='ml', key='max_spec_len')
        self.max_pep_len = config.get_config(section='ml', key='max_pep_len')
        self.min_pep_len = config.get_config(section='ml', key='min_pep_len')
        self.embedding_dim = config.get_config(section='ml', key='embedding_dim')
        self.num_encoder_layers = config.get_config(section='ml', key='encoder_layers')
        self.num_heads = config.get_config(section='ml', key='num_heads')
        do = config.get_config(section="ml", key="dropout")
        
        ################### Spectra branch ###################
        # self.spec_embedder = nn.Embedding(self.spec_size, self.embedding_dim)
        # self.spec_pos_encoder = PositionalEncoding(self.embedding_dim, dropout=do, max_len=self.max_spec_len)
        # encoder_layers = nn.TransformerEncoderLayer(self.embedding_dim, nhead=self.num_heads, dropout=do, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_encoder_layers)
        # self.bn1 = nn.BatchNorm1d(num_features=self.embedding_dim * self.max_spec_len)

        # self.linear1_1 = nn.Linear(self.embedding_dim * self.max_spec_len, 1024)
        self.linear1_1 = nn.Linear(self.spec_size, 1024)
        # self.bn1 = nn.BatchNorm1d(num_features=1024)
        
        self.linear1_2 = nn.Linear(1024, 512)
        # self.bn2 = nn.BatchNorm1d(num_features=512)

        self.linear1_3 = nn.Linear(512, 256)
        # self.bn3 = nn.BatchNorm1d(num_features=256)

        self.linear_out = nn.Linear(256, self.max_pep_len - self.min_pep_len)

        self.dropout = nn.Dropout(do)
        
    def forward(self, data, mask):

        # data = self.spec_embedder(data) * math.sqrt(self.embedding_dim)
        # data = self.spec_pos_encoder(data)
        
        # out = self.encoder(data, src_key_padding_mask=mask)
        
        # out = F.relu(self.bn2(self.linear1_1(out.view(-1, self.embedding_dim * self.max_spec_len))))
        out = F.relu((self.linear1_1(data.view(-1, self.spec_size))))
        out = self.dropout(out)

        out = F.relu((self.linear1_2(out)))
        out = self.dropout(out)

        out = F.relu((self.linear1_3(out)))
        out = self.dropout(out)

        out = self.linear_out(out)
        
        return out
    
    def name(self):
        return "Net"


# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html 6/25/2021
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

