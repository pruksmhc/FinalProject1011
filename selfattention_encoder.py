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
import sys
import gc
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# from model_architectures import Encoder_RNN, Decoder_RNN
from data_prep import * 
from misc import timeSince, load_cpickle_gc
from logistics import *
from inference import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
PAD_token = 0
PAD_TOKEN = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
teacher_forcing_ratio = 1.0
attn_model = 'dot'

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: PAD_token, 1: SOS_token, 2: EOS_token, 3:UNK_token}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, count):
#     encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    err_real = 0
    loss = 0
#     encoder_outputs = torch.zeros(max_length, decoder.hidden_size, device=device)

    # iterate GRU over words --> final hidden state is representation of source sentence. 
    for ei in range(input_length):
#         encoder_output = encoder(input_tensor[ei], encoder_hidden)
        encoder_output = encoder(input_tensor[ei],None)
#         encoder_outputs[ei] = encoder_output[0,0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_output
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(1, target_length-1):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
#             err_real += loss.data[0]
#             err_real += loss.item()
            count += 1
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(1, target_length-1):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
#             err_real += loss.data[0]
#             err_real += loss.item()
            count += 1
            if decoder_input.item() == EOS_token:
                break
                
    if type(loss) != torch.Tensor:
#         pdb.set_trace()
        loss = torch.Tensor(loss)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, count

def trainIters(encoder, decoder,n_epochs, validation_pairs, pairs, lang1, lang2, max_length, max_length_generation, title, print_every=5000, plot_every=5000, learning_rate=3e-4, search="beam"):
    """
    lang1 is the Lang o|bject for language 1 
    Lang2 is the Lang object for language 2
    n_iters is the number of training pairs per epoch you want to train on
    """
    
    start = time.time()
    training_pairs = pairs
    n_iters = len(pairs)
    plot_losses, val_losses = [], []
    val_losses = [] 
    count, print_loss_total, plot_loss_total, val_loss_total, plot_val_loss = 0, 0, 0, 0, 0 
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss(ignore_index=PAD_token)
    plot_loss =[]
    val_loss = []
    
    for i in range(n_epochs):
        plot_loss =[]
        val_loss = []
        # framing it as a categorical loss function. 
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1] 
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            input_length = input_tensor.size(0)
            if target_tensor.size(0) < 3:
                continue
            loss_value, count = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, count)
            print_loss_total += loss_value 
            plot_loss_total += loss_value
            
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / count
                count = 0
                print_loss_total = 0
                print('TRAIN SCORE %s (%d %d%%) %.4f' % (timeSince(start, iter / n_epochs),
                                             iter, iter / n_epochs * 100, print_loss_avg))
                plot_loss.append(print_loss_avg)
                plot_loss_total = 0
                with torch.no_grad():
                    v_loss = test_model(encoder, decoder, search, validation_pairs, lang2, max_length=None, no_attention=False)
                # returns bleu score
                print("VALIDATION BLEU SCORE: "+str(v_loss))
                sys.stdin.flush()
                val_loss.append(v_loss)
                save_model(encoder,decoder, title)
        plot_losses.append(plot_loss)
        val_losses.append(val_loss)
        save_model(encoder,decoder, title)
        make_graph(encoder, decoder, val_losses, plot_losses, title)

   

train_idx_pairs = load_cpickle_gc("preprocessed_data_no_elmo/iwslt-zh-eng/preprocessed_no_indices_pairs_train_tokenized")
input_lang = load_cpickle_gc("preprocessed_data_no_elmo/iwslt-zh-eng/preprocessed_no_elmo_zhlang")
target_lang = load_cpickle_gc("preprocessed_data_no_elmo/iwslt-zh-eng/preprocessed_no_elmo_englang")
val_idx_pairs =  pickle.load(open("preprocessed_data_no_elmo/iwslt-zh-eng/preprocessed_no_indices_pairs_validation_tokenized", 'rb'))
hidden_size = 256
# number of duplicate layers in encoder
N = 1
# number of heads
h=8
dropout=0.1
"Helper: Construct a model from hyperparameters."
attn = MultiHeadedAttention(h, hidden_size).cuda()
ff = PositionwiseFeedForward(hidden_size,input_lang.n_words, dropout).cuda()
position = PositionalEncoding(hidden_size, dropout).cuda()
src_embed = nn.Sequential(Embeddings(hidden_size, input_lang.n_words), position).cuda()
encoder1 = SupEncoder(Encoder(EncoderLayer(hidden_size, attn, ff, dropout), N),src_embed).cuda()

decoder1 = Decoder_RNN(target_lang.n_words,hidden_size).cuda()
args = {
    'n_epochs': 10,
    'learning_rate': 0.001,
    'search': 'beam',
    'encoder': encoder1,
    'decoder': decoder1,
    'lang1': input_lang, 
    'lang2': target_lang,
    "pairs":train_idx_pairs, 
    "validation_pairs": val_idx_pairs[:200], 
    "title": "Training Curve for Basic Self Encoder With LR = 0.0001",
    "max_length": 100,
    "max_length_generation": 20, 
    "plot_every": 500, 
    "print_every": 500
}

"""
We follow https://arxiv.org/pdf/1406.1078.pdf 
and use the Adadelta optimizer

"""
print(BATCH_SIZE)

trainIters(**args)

