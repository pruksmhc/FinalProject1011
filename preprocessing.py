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
from allennlp.commands.elmo import ElmoEmbedder
import unicodedata
import re
import spacy
import numpy as np
from underthesea import word_tokenize
import jieba
import pdb
import os
from elmoformanylangs import Embedder
import math
EMBED_DIM = 1024

def preprocess_weights(weights):
    final_weights = []
    for i in range(len(weights)):
        try:
            if math.isnan(weights[i]):
                final_weights.append([0]* 1024)
        except:
            final_weights.append(np.mean(weights[i], axis=0))
    return final_weights

def load_elmo(all_tokens, language=""):
    # here, we will load ELMO. 
    # For Chinese + Vietnamese, we use pretrained https://github.com/HIT-SCIR/ELMoForManyLangs
    all_tokens = list(set(all_tokens))
    current_word2idx = {all_tokens[i]: i+4 for i in range(len(all_tokens))}
    current_word2idx["pad"] = 0
    current_word2idx["unk"] = 1
    current_word2idx["<SOS>"] = 2
    current_word2idx["<EOS>"] = 3
    weights = []
    elmo = ElmoEmbedder()
    if language == "zh":
        e = Embedder('179')
        weights = e.sents2elmo(all_tokens)
        weights = preprocess_weights(weights)
        # so this basically returns a 3x 1024 for any words that are 
        # in the token_embedding
        # and nan for any words not in the embedding
    elif language == "vi":
        pdb.set_trace()
        e = Embedder('178')
        weights = e.sents2elmo(all_tokens)
        weights = preprocess_weights(weights)
    else: 
        weights = elmo.embed_sentence(all_tokens)
        weights = np.mean(weights, axis=0)
    # take the average
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

def transform_dict(curr_dict):
    new_dict = {}
    new_dict[0] = "pad"
    new_dict[1] = "unk"
    for key, value in list(curr_dict.items()):
        new_dict[key+2] = value
    new_dict[2] = "<SOS>"
    new_dict[3] = "<EOS>"
    return new_dict

class Lang:
    # This class counts the index to word. 
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "pad", 1: "unk", 2: "<SOS>", 3: "<EOS>"}
        self.n_words = 2  # so now, let'scount this again. 

    def addSentence(self, sentence, language):
        # and we have the tokenization here.
        tokens = token_funcs[language](sentence)
        for word in tokens:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def tokenize_vi(sentence):
    tokens = word_tokenize(sentence)
    return tokens


def tokenize_zh(sentence):
    tokens = jieba.cut(sentence, cut_all=True)
    return tokens
tokenizer = spacy.load('en_core_web_sm')
def tokenize_en(sentence):
    tokens = tokenizer(sentence)
    final = [t.text for t in tokens]
    return final

token_funcs = {"en": tokenize_en, "vi": tokenize_vi, "zh": tokenize_zh}
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
def tokenize_indices(lang1, lang2, dataset, index, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    text_file = pickle.load(open("preprocessed_data/iwslt-%s-%s/%s_%s" % (lang1, lang2, dataset, index), "rb"))
    output_pair = pickle.load(open("preprocessed_data/iwslt-%s-%s/%s_output" % ("zh", "en", "train"), "rb"))
    input_pair = pickle.load(open("preprocessed_data/iwslt-%s-%s/%s_input" % ("vi", "en", "train"), "rb"))
    new_file = []
    for i in range(len(text_file)):
        #eos sos, eos sos. 
        text = text_file[i]
        word2idx = input_pair.word2index
        new = []
        for word in text:
            if word == "<EOS>":
                word2idx = output_pair.word2index
            if word[0] in word2idx.keys():
                new.append(word2idx[word[0]])
            else:
                new.append(word2idx["unk"])
        new_file.append(new)
    pdb.set_trace()
    pickle.dump(new_file, open("preprocessed_data/iwslt-%s-%s/%s_indexed_%s" % (lang1, lang2, dataset, index), "wb"))
    return new_file

def readLangs(lang1, lang2, dataset_type, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lang1_file = open('iwslt-%s-%s/%s.tok.%s' % (lang1, lang2, dataset_type, lang1), encoding='utf-8').\
        read().strip().split('\n')

    lang2_file = open('iwslt-%s-%s/%s.tok.%s' % (lang1, lang2, dataset_type, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = []
    dirlink = "preprocessed_data/iwslt-%s-%s/%s_11" % (lang1, lang2, dataset_type)
    for i in range(132736, len(lang1_file)):
        print(i)
        pair = ["<SOS>"]
        lang1_text = token_funcs[lang1](normalizeString(lang1_file[i]))
        pair.extend(lang1_text)
        pair.extend(["<EOS>", "<SOS>"])
        lang2_text = token_funcs[lang2](normalizeString(lang2_file[i]))
        pair.extend(lang2_text)
        pair.extend(["<EOS>"])
        pairs.append(pair)
        pickle.dump(pairs, open(dirlink, "wb"))
    return pairs

"""
output_pair = pickle.load(open("preprocessed_data/iwslt-%s-%s/%s_output" % ("vi", "en", "train"), "rb"))
# load the ELMO weight embedidngs for ONLY train vocabulary
all_tokens_vicurrent_word2idx, all_tokens_vifinal_weights = load_elmo(output_pair.word2count.keys())
pickle.dump(all_tokens_vifinal_w=eights, open("weights_train_en1", "wb"))

output_pair = pickle.load(open("preprocessed_data/iwslt-%s-%s/%s_output" % ("zh", "en", "train"), "rb"))

pdb.set_trace()
all_tokens_vicurrent_word2idx, all_tokens_vifinal_weights = load_elmo(list(output_pair.word2count.keys())[5000:])
pickle.dump(all_tokens_vifinal_weights, open("weights_train_en2_rest", "wb"))

pdb.set_trcae()<
input_lang = pickle.load(open("preprocessed_data/iwslt-%s-%s/train_input" % ("vi", "en"), "rb"))
output_lang = pickle.load(open("preprocessed_data/iwslt-%s-%s/train_output" % ("vi", "en"), "rb"))
pdb.set_trace()
new_train = []
curr_length = 0
i = 1
train = pickle.load(open("preprocessed_data/iwslt-%s-%s/%s_indexed_%s" % ("vi", "en", "train", str(i)), "rb"))

new_train.extend(train)
pdb.set_trace()
pickle.dump(new_train, open("preprocessed_data/iwslt-vi-en/train_indexed", "wb"))

lang1 = "vi"
lang2 = "en"
dataset = "train"
indices1 = tokenize_indices(lang1, lang2, dataset,"1")
indices2 = tokenize_indices(lang1, lang2, dataset,"2")
indices4 =tokenize_indices(lang1, lang2, dataset,"4")
indices5 =tokenize_indices(lang1, lang2, dataset,"5")
indices6 =tokenize_indices(lang1, lang2, dataset,"6")
indices7 =tokenize_indices(lang1, lang2, dataset,"7")
indices8 =tokenize_indices(lang1, lang2, dataset,"8")
indices9 =tokenize_indices(lang1, lang2, dataset,"9")
indices10 =tokenize_indices(lang1, lang2, dataset,"10")
indices11 =tokenize_indices(lang1, lang2, dataset,"11")
"""
indices1 = pickle.load(open("preprocessed_data/iwslt-vi-en/train_1", "rb"))
indices2 = pickle.load(open("preprocessed_data/iwslt-vi-en/train_2", "rb"))
indices4 = pickle.load(open("preprocessed_data/iwslt-vi-en/train_4","rb"))
indices5 = pickle.load(open("preprocessed_data/iwslt-vi-en/train_5", "rb"))
indices6 = pickle.load(open("preprocessed_data/iwslt-vi-en/train_6", "rb"))
indices7 = pickle.load(open("preprocessed_data/iwslt-vi-en/train_7", "rb"))
indices8 = pickle.load(open("preprocessed_data/iwslt-vi-en/train_8", "rb"))
indices9 = pickle.load(open("preprocessed_data/iwslt-vi-en/train_9", "rb"))
indices10 = pickle.load(open("preprocessed_data/iwslt-vi-en/train_10", "rb"))
indices11 = pickle.load(open("preprocessed_data/iwslt-vi-en/train_11", "rb"))
indices1.extend(indices2)
indices1.extend(indices4)
indices1.extend(indices5)
indices1.extend(indices6)
indices1.extend(indices7)
indices1.extend(indices8)
indices1.extend(indices9)
indices1.extend(indices10)
indices1.extend(indices11)
pickle.dump(indices1, open("preprocessed_data/iwslt-vi-en/train_tokenized", "wb"))




