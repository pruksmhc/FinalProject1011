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
        all_tokens = all_tokens[:10]
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
    pdb.set_trace()
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

def tokenize_en(sentence):
    tokenizer = spacy.load('en_core_web_sm')
    tokens = tokenizer(sentence)
    return tokens

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
        # tokenizes based on the language. 
        input_lang.addSentence(lang1_file[i], lang1)
        output_lang.addSentence(lang2_file[i], lang2)

    pickle.dump(pairs, open("preprocessed_data/iwslt-%s-%s/%s" % (lang1, lang2, dataset_type), "wb"))
    pickle.dump(input_lang, open("preprocessed_data/iwslt-%s-%s/%s_input" % (lang1, lang2, dataset_type), "wb"))
    pickle.dump(output_lang, open("preprocessed_data/iwslt-%s-%s/%s_output" % (lang1, lang2, dataset_type), "wb"))
    return input_lang, output_lang, pairs
"""
output_pair = pickle.load(open("preprocessed_data/iwslt-%s-%s/%s_output" % ("vi", "en", "train"), "rb"))
# load the ELMO weight embedidngs for ONLY train vocabulary
all_tokens_vicurrent_word2idx, all_tokens_vifinal_weights = load_elmo(output_pair.word2count.keys())
pickle.dump(all_tokens_vifinal_w=eights, open("weights_train_en1", "wb"))

output_pair = pickle.load(open("preprocessed_data/iwslt-%s-%s/%s_output" % ("zh", "en", "train"), "rb"))

pdb.set_trace()
all_tokens_vicurrent_word2idx, all_tokens_vifinal_weights = load_elmo(list(output_pair.word2count.keys())[5000:])
pickle.dump(all_tokens_vifinal_weights, open("weights_train_en2_rest", "wb"))

pdb.set_trcae()
"""
input_pair = pickle.load(open("preprocessed_data/iwslt-%s-%s/%s_input" % ("zh", "en", "train"), "rb"))

all_tokens_zhcurrent_word2idx, all_tokens_zhfinal_weights = load_elmo(input_pair.word2count.keys(), "zh")
pickle.dump(all_tokens_zhfinal_weights, open("weights_train_zh", "wb"))


input_pair = pickle.load(open("preprocessed_data/iwslt-%s-%s/%s_input" % ("vi", "en", "train"),  "rb"))
all_tokens_vicurrent_word2idx, all_tokens_vifinal_weights = load_elmo(input_psair.word2count.keys(), "vi")
pickle.dump(all_tokens_vifinal_weights, open("weights_train_vi", "wb"))


