from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import pickle
import time
import math, copy
from random import randint
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import pdb
import tensorflow as tf

import matplotlib.pyplot as plt
plt.switch_backend('agg')

# from model_architectures import Encoder_RNN, Decoder_RNN
from misc import timeSince, load_cpickle_gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embed = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embed, hidden)
        # output and hidden are the same vectors
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    

class Decoder_RNN(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder_RNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        pdb.set_trace()
        embed = self.embedding(input).view(1, 1, -1)
        embed = F.relu(embed)
        output, hidden = self.gru(embed, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Encoder_Batch_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder_Batch_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, sents, sent_lengths):
        '''
            sents is a tensor with the shape (batch_size, padded_length )
            when we evaluate sentence by sentence, you evaluate it with batch_size = 1, padded_length.
            [[1, 2, 3, 4]] etc. 
        '''
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # We sort and then do pad packed sequence here. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = np.array(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = self.init_hidden(batch_size)
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        # change the order back
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device)) 
        
        # **TODO**: What is rnn_out - for attention. 
        return rnn_out, self.hidden

class EncoderRNNBidirectionalBatch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNNBidirectionalBatch, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, dropout=0.1)

    def forward(self, sents, sent_lengths):
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # We sort and then do pad packed sequence here. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = np.array(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = self.init_hidden(batch_size)
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        # change the order back
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device)) 
        return rnn_out, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=device)


class Decoder_RNN(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder_RNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embed = self.embedding(input).view(1, 1, -1)
        embed = F.relu(embed)
        output, hidden = self.gru(embed, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class SupEncoder(nn.Module):

    """
    A super class of Encoder that learns embeddings
    """
    def __init__(self, encoder, src_embed):
        super(SupEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed 
        
    def forward(self, src, src_mask):
        "Take in and process masked src sequences."
        a = self.encode(src, src_mask)
        a = a[:,0,:]
        a = a.unsqueeze(1)
        a = a.cuda()
#         pdb.set_trace()
#         print(np.shape(a))
        return a
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    

def clones(module, N):
    "Produce N identical layers."
#     module = module.cpu()
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        x = x.cuda()
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        features = features
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        x = x.cuda()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (self.a_2 * (x - mean) / (std + self.eps) + self.b_2).cuda()

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x = x.cuda()
        return (x + self.dropout(sublayer(self.norm(x))))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = x.cuda()
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        mask = mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.cuda()
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff).cuda()
        self.w_2 = nn.Linear(d_ff, d_model).cuda()
        self.dropout = nn.Dropout(dropout).cuda()

    def forward(self, x):
        x = x.cuda()
        return self.w_2(self.dropout(F.relu(self.w_1(x)).cuda()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model).cuda()
        self.d_model = d_model

    def forward(self, x):
        # x is the weights here
        x = x.cuda()
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).cuda()
        position = torch.arange(0., max_len).unsqueeze(1).cuda()
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        div_term = div_term.cuda()
#         pdb.set_trace()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
#         pdb.set_trace()
        x = x.cuda()
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)

        return self.dropout(x)




