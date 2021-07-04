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
    def __init__(self, embedding_dim=512):
        super(Net, self).__init__()
        self.spec_size = config.get_config(section='input', key='spec_size')
        self.charge = config.get_config(section='input', key='charge')
        self.max_spec_len = config.get_config(section='ml', key='max_spec_len')
        self.max_pep_len = config.get_config(section='ml', key='max_pep_len')
        self.min_pep_len = config.get_config(section='ml', key='min_pep_len')
        do = config.get_config(section="ml", key="dropout")
        self.embedding_dim = embedding_dim
        
        ################### Spectra branch ###################
        self.spec_embedder = nn.Embedding(self.spec_size, embedding_dim)
        self.spec_pos_encoder = PositionalEncoding(embedding_dim, dropout=do, max_len=self.max_spec_len)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead=8, dropout=do, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.bn1 = nn.BatchNorm1d(num_features=embedding_dim * self.max_spec_len)

        self.linear1_1 = nn.Linear(embedding_dim * self.max_spec_len, 1024)
        self.bn2 = nn.BatchNorm1d(num_features=1024)
        
        self.linear1_2 = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(num_features=256)

        self.linear1_3 = nn.Linear(256, self.max_pep_len - self.min_pep_len)

        self.dropout_conv1_1 = nn.Dropout(do)
        self.dropout1 = nn.Dropout(do)
        self.dropout2 = nn.Dropout(do)
        self.dropout3 = nn.Dropout(do)
        
    def forward(self, data, mask):

        data = self.spec_embedder(data) * math.sqrt(self.embedding_dim)
        data = self.spec_pos_encoder(data)
        
        out = self.encoder(data, src_key_padding_mask=mask)
        
        out = F.relu(self.bn2(self.linear1_1(out.view(-1, self.embedding_dim * self.max_spec_len))))
        out = self.dropout1(out)

        out = F.relu(self.bn3(self.linear1_2(out)))
        out = self.dropout2(out)

        out = F.relu(self.linear1_3(out))
        
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

