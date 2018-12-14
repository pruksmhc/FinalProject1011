from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import pickle
import time
import sys
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pdb
from random import randint
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from model_architectures import *
from data_prep import *
from misc import timeSince, load_cpickle_gc
from inference import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
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

hidden_size = 256

input_lang = pickle.load(open("preprocessed_data_no_elmo/iwslt-vi-eng/preprocessed_no_elmo_vilang", "rb"))
target_lang = pickle.load(open("preprocessed_data_no_elmo/iwslt-vi-eng/preprocessed_no_elmo_englang", "rb"))
test_pairs = pickle.load(open("preprocessed_data_no_elmo/iwslt-vi-eng/preprocessed_no_indices_pairs_test_tokenized", "rb"))
# is there anything in the train_idx_pairs that is only 0s right noww instea dof padding. 

encoder1 = Encoder_Batch_RNN(input_lang.n_words, hidden_size).to(device)
encoder1.load_state_dict(torch.load("output/TrainingCurveforBasic1-DirectionalEncoderDecoderModelWithLR=0.001nodecoderbatchingencodermodel_states"))
decoder1 = Decoder_RNN(target_lang.n_words, hidden_size).to(device)
decoder1.load_state_dict(torch.load("output/TrainingCurveforBasic1-DirectionalEncoderDecoderModelWithLR=0.001nodecoderbatchingdecodermodel_states"))

"""
We take the input sentence as the length of the maximum generating sentence 
We follow https://arxiv.org/pdf/1406.1078.pdf 
and use the Adadelta optimizer
Have max_length_generation

"""


test_model(encoder1, decoder1, "beam", test_pairs[:100], target_lang, max_length=None, no_attention=True)


