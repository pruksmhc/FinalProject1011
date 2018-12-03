from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 30

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
            
            

# Turn a Unicode string to plain ASCII: http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def readLangs(input_file, target_file, input_lang, target_lang, size=None):
    print("Reading lines...")

    # Read the file and split into lines
    with open(input_file, encoding='utf-8') as file:
        if size == None:
            input_lines = open(input_file, encoding='utf-8').read().strip().split("\n")
        else:
            input_lines = [next(file).strip() for x in range(size)]
        
    with open(target_file, encoding='utf-8') as file:
        if size == None:
            target_lines = open(target_file, encoding='utf-8').read().strip().split("\n")
        else:
            target_lines = [next(file).strip() for x in range(size)]
        
    if input_lang == "zh":
        target_pairs = [normalizeString(s) for s in target_lines]
        pairs = list(zip(input_lines, target_pairs))
    else:
        lines = list(zip(input_lines, target_lines))
        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in l] for l in lines]
    print(pairs[0])

    input_lang = Lang(input_lang)
    target_lang = Lang(target_lang)

    return input_lang, target_lang, pairs


def prepareData(input_file, target_file, input_lang, target_lang, size=None):
    
    input_lang, target_lang, pairs = readLangs(input_file, target_file, input_lang, target_lang, size)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    print(pairs[0])
    for pair in pairs:
        input_lang.addSentence(pair[0])
        target_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(target_lang.name, target_lang.n_words)
    return input_lang, target_lang, pairs

def indexesFromSentence(lang, sentence):
    words = sentence.split(' ')
    indices = []
    for word in words:
        if lang.word2index.get(word) is not None:
            indices.append(lang.word2index[word])
        else:
            indices.append(1) # UNK_INDEX
    return indices

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, target_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(target_lang, pair[1])
    return (input_tensor, target_tensor)