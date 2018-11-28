teacher_forcing_ratio = 0.5
import time
import math
import pdb
import pickle
from Model1 import * 
import torch
import torch.nn as nn
import preprocessing
from preprocessing import Lang
import random
import itertools

MAX_LENGTH_VI_EN = 1310
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

SOS_token = 2
EOS_token = 3

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

# 
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length,attention=False):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    criterion = nn.NLLLoss()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    # waht is encoder-decoder methodology? 
    loss = 0

    for ei in range(input_length):
        # pass in pair by pair rather than using dataloader. 
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    # add SOS before decoding. <eng> EOS SOS <fra>
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden, = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden, = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            # so only sotp when you see an EOS_token
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def isplit(iterable,splitters):
    # gotten from https://stackoverflow.com/questions/4322705/split-a-list-into-nested-lists-on-a-value
    return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]


def indexesFromSentence(lang, sentence):
    sentence_new = []
    keys = lang.word2index.keys()
    for word in sentence:
        if word in keys:
            sentence_new.append(lang.word2index[word])
        else:
            sentence_new.append(1)
    return sentence_new


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def trainIters(encoder, decoder, n_iters,n_epochs,  lang1, lang2,  print_every=1000, plot_every=100, learning_rate=0.001):
    pairs = pickle.load(open("preprocessed_data/iwslt-%s-%s/%s" % (lang1.name, lang2.name, "train_tokenized"), "rb"))
    dev_pairs = pickle.load(open("preprocessed_data/iwslt-vi-en/dev_indexed", "rb"))
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    for i in range(n_epochs):
        # so now, becuase we've already indeed, 
        training_pairs = [random.choice(pairs) for i in range(n_iters)] # samples the pairs. ?
        criterion = nn.NLLLoss()
        # framing it as a categorical loss function. 
        for iter in range(1, n_iters + 1):
            print(inter)
            training_pair = training_pairs[iter - 1] 
            training_pair_final = isplit(training_pair,("<EOS>",))
            # and now we split by the <EOS> tag sicne we hav ethat 
            d_input_tensor = training_pair_final[0]
            d_target_tensor = training_pair_final[1]
            input_tensor = tensorFromSentence(lang1, d_input_tensor)
            target_tensor = tensorFromSentence(lang2, d_target_tensor)
            # these eshould be indexed, and wth EOS at the end. 

            # so you have to output the EOS ss well when you're done. 
            # input is the source language, target is the target language. 
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH_VI_EN)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('TRAIN SCORE %s (%d %d%%) %.4f' % (timeSince(start, iter / n_epochs),
                                             iter, iter / n_epochs * 100, print_loss_avg))
                VAL_loss = train(input_val_tensor, target_val_tensor, encoder,
                 decoder, encoder_optimizer, decoder_optimizer, criterion)

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)
# get the tokenzied parts. 
hidden_size = 1024
train_data = pickle.load(open("preprocessed_data/iwslt-vi-en/train_indexed", "rb"))
lang_object_input =  pickle.load(open("preprocessed_data/iwslt-vi-en/train_input", "rb"))
lang_object_output =  pickle.load(open("preprocessed_data/iwslt-vi-en/train_output", "rb"))
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

weights_en_torch = torch.load("weights_en_torch")
weights_vi_torch = torch.load("weights_vi_torch")
encoder = EncoderRNN(lang_object_input.n_words, hidden_size, weights_vi_torch).to(device)
decoder = DecoderRNN(hidden_size, lang_object_output.n_words, weights_en_torch).to(device)
trainIters(encoder, decoder, 750, 10, lang_object_input, lang_object_output, print_every=5) #train on a small subset

# i hsould have called it dev_tokenied, adn then dev_indexed. 


