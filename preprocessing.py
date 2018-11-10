# now, we tokenize our current dataset
import pickle
import pdb 
import numpy as np
import pandas as pd
import pprint
import numpy as np
from operator import itemgetter
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from allennlp.commands.elmo import ElmoEmbedder


def load_elmo(all_tokens):
    # here, we will load ELMO. 
    all_tokens = list(set(all_tokens))
    current_word2idx = {all_tokens[i]: i+4 for i in range(len(all_tokens))}
    current_word2idx["pad"] = 0
    current_word2idx["unk"] = 1
    current_word2idx["<SOS>"] = 2
    current_word2idx["<EOS>"] = 3
    weights = []
    elmo = ElmoEmbedder()
    weights = elmo.embed_sentence(all_tokens)
    # take the average
    weights = np.mean(weights, axis=0)
    pdb.set_trace()
    final_weights = []
    final_weights.append([0]* EMBED_DIM)
    unknown_vector = list(np.random.normal( size=(EMBED_DIM, )))
    start_vector = list(np.random.normal(size=(EMBED_DIM, )))
    end_vector = list(np.random.normal(size=(EMBED_DIM, )))
    final_weights.append(unknown_vector)
    final_weights.append(start_vector)
    final_weights.append(end_vector)
    final_weights.extend(weights)
    return current_word2idx, final_weights




SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

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

            