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
import unicodedata
import re

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
    # This class counts the index to word. 
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "sos", 1: "eos"}
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

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
# so normalize the string. 

def readLangs(lang1, lang2, dataset_type, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lang1_file = open('iwslt-%s-%s/%s.tok.%s' % (lang1, lang2, dataset_type, lang1), encoding='utf-8').\
        read().strip().split('\n')

    lang2_file = open('iwslt-%s-%s/%s.tok.%s' % (lang1, lang2, dataset_type, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [normalizeString(lang1_file[i] +"EOS" + lang2_file[i] +"EOS") for i in range(len(lang1_file))]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    for i in range(len(lang1_file)):
        input_lang.addSentence(lang1_file[i])
        output_lang.addSentence(lang2_file[i])

    pickle.dump(pairs, open("preprocessed_data/iwslt-%s-%s/%s" % (lang1, lang2, dataset_type), "wb"))
    pickle.dump(input_lang, open("preprocessed_data/iwslt-%s-%s/%s_input" % (lang1, lang2, dataset_type), "wb"))
    pickle.dump(output_lang, open("preprocessed_data/iwslt-%s-%s/%s_output" % (lang1, lang2, dataset_type), "wb"))
    return input_lang, output_lang, pairs


readLangs("vi", "en", 'dev')
readLangs("vi", "en", 'train')
readLangs("vi", "en", 'test')


readLangs("zh", "en", 'dev')
readLangs("zh", "en", 'train')
readLangs("zh", "en", 'test')

input_pair = pickle.load("preprocessed_data/iwslt-%s-%s/%s_input" % ("vi", "en", "train"))
output_pair = pickle.load("preprocessed_data/iwslt-%s-%s/%s_output" % ("vi", "en", "train"))
# load the ELMO weight embedidngs for ONLY train vocabulary
all_tokens_en_1 = load_elmo(input_pair.word2count.keys())
all_tokens_vi = load_elmo(input_pair.word2count.keys())
pickle.dump(all_tokens_en_1, open("weights_train_en1", "wb"))
pickle.dump(all_tokens_vi, open("weights_train_vi", "wb"))


