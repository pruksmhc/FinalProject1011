from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pdb

import pickle
import _pickle as cPickle
import gc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
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



def indexesFromSentence(lang, sentence, i):
    try:
        words = sentence.split(' ')
    except Exception:
        pdb.set_trace()
    indices = []
    for word in words:
        if lang.word2index.get(word) is not None:
            indices.append(lang.word2index[word])
        else:
            indices.append(UNK_token) # UNK_INDEX
    return indices


def tensorFromSentence(lang, sentence, i):
    indexes = indexesFromSentence(lang, sentence, i)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, target_lang, i):
    input_tensor = tensorFromSentence(input_lang, pair[0], i)
    target_tensor = tensorFromSentence(target_lang, pair[1], i)
    return (input_tensor, target_tensor)




SOS_token = 0
EOS_token = 1
UNK_token = 2


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        # output and hidden are the same vectors
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5


# example of input_tensor: [2, 43, 23, 9, 19, 4]. Indexed on our vocabulary. 
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # iterate GRU over words --> final hidden state is representation of source sentence. 
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


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



def load_cpickle_gc(dirlink):
    # https://stackoverflow.com/questions/26860051/how-to-reduce-the-time-taken-to-load-a-pickle-file-in-python
    output = open(dirlink, 'rb')

    # disable garbage collector
    gc.disable()

    mydict = pickle.load(output)

    # enable garbage collector again
    gc.enable()
    output.close()
    return mydict

def trainIters(encoder, decoder, n_iters,n_epochs,  lang1, lang2, max_length, print_every=1000, plot_every=100, learning_rate=0.001, search="greedy"):
    """
    lang1 is the Lang o|bject for language 1 
    Lang2 is the Lang object for language 2
    n_iters is the number of training pairs per epoch you want to train on
    """
    training_pairs = load_cpickle_gc("preprocessed_data_no_elmo/iwslt-"+lang1.name+"-"+lang2.name+"/preprocessed_no_indices_pairs_train_tokenized")
    validation_pairs = load_cpickle_gc("preprocessed_data_no_elmo/iwslt-zh-eng/preprocessed_no_indices_pairs_validation_tokenized")

    start = time.time()
    plot_losses = []
    val_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0
    val_loss_total = 0
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    #training_pairs = [tensorsFromPair(pair, lang1, lang2, 0) for pair in pairs]
    for i in range(n_epochs):
        criterion = nn.NLLLoss()
        # framing it as a categorical loss function. 
        for iter in range(1, n_iters + 1):
            if iter % 100 == 0:
                print(iter)
            training_pair = training_pairs[iter - 1] 
            d_input_tensor = training_pair[0]
            d_target_tensor = training_pair[1]
            loss = train(d_input_tensor, d_target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('TRAIN SCORE %s (%d %d%%) %.4f' % (timeSince(start, iter / n_epochs),
                                             iter, iter / n_epochs * 100, print_loss_avg))
                val_loss = test_model(encoder, decoder,search, validation_pairs, lang1, max_length)

                # retursn teh bleu score
                print("VALIDATION BLEU SCORE: "+str(val_loss))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                val_loss_avg = val_loss_total/plot_every
                plot_losses.append(plot_loss_avg)
                val_losses.append(val_loss_avg)
                plot_loss_total = 0
                val_loss_total = 0
        pickle.dump(encoder, open("encoder_"+str(num_epochs), "wb"))
        pickle.dump(decoder, open("decoder_"+str(num_epochs), "wb"))
        pickle.dump(plot_loss_avg, open("training_loss", "wb"))
        pickle.dump(val_loss_avg, open("val_loss", "wb"))
        pdb.set_trace()
    showPlot(plot_losses)

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

def greedy_search(decoder, decoder_input, hidden, max_length):
    translation = []
    for i in range(max_length):
        next_word_softmax, hidden = decoder(decoder_input, hidden)
        best_idx = torch.max(next_word_softmax, 1)[1].squeeze().item()

        # convert idx to word
        best_word = target_lang.index2word[best_idx]
        translation.append(best_word)
        decoder_input = torch.tensor([[best_idx]], device=device)
        
        if best_word == 'EOS':
            break
    return translation


def beam_search(decoder, decoder_input, hidden, max_length, k = 2):
    
    candidates = [(decoder_input, 0, hidden)]
    potential_candidates = []
    completed_translations = []

    # put a cap on the length of generated sentences
    for m in range(max_length):
        for c in candidates:
            # unpack the tuple
            c_sequence = c[0]
            c_score = c[1]
            c_hidden = c[2]
            # EOS token
            if c_sequence[-1] == 1:
                completed_translations.append((c_sequence, c_score))
                k = k - 1
            else:
                next_word_probs, hidden = decoder(c_sequence[-1], c_hidden)
                # in the worst-case, one sequence will have the highest k probabilities
                # so to save computation, only grab the k highest_probability from each candidate sequence
                top_probs, top_idx = torch.topk(next_word_probs, k)
                for i in range(len(top_probs[0])):
                    word = torch.from_numpy(np.array([top_idx[0][i]]).reshape(1, 1)).to(device)
                    new_score = c_score + top_probs[0][i]
                    potential_candidates.append((torch.cat((c_sequence, word)).to(device), new_score, c_hidden))

        candidates = sorted(potential_candidates, key= lambda x: x[1])[0:k] 
        potential_candidates = []

    completed = completed_translations + candidates
    completed = sorted(completed, key= lambda x: x[1])[0] 
    final_translation = []
    for x in completed[0]:
        final_translation.append(target_lang.index2word[x.squeeze().item()])
    return final_translation


def evaluate(encoder, decoder, sentence, max_length, search="greedy", k= None):
    """
    Function that generate translation.
    First, feed the source sentence into the encoder and obtain the hidden states from encoder.
    Secondly, feed the hidden states into the decoder and unfold the outputs from the decoder.
    Lastly, for each outputs from the decoder, collect the corresponding words in the target language's vocabulary.
    And collect the attention for each output words.
    @param encoder: the encoder network
    @param decoder: the decoder network
    @param sentence: string, a sentence in source language to be translated
    @param max_length: the max # of words that the decoder can return
    @output decoded_words: a list of words in target language
    @output decoder_attentions: a list of vector, each of which sums up to 1.0
    """    
    # process input sentence
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        # encode the source lanugage
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        # decode the context vector
        decoder_hidden = encoder_hidden # decoder starts from the last encoding sentence
        # output of this function
        decoder_attentions = torch.zeros(max_length, max_length)
        decoded_words = []
        if search == 'greedy':
            decoded_words = greedy_search(decoder, decoder_input, decoder_hidden, max_length)
        elif search == 'beam':
            if k == None:
                k = 2
            decoded_words = beam_search(decoder, decoder_input, decoder_hidden, max_length, k)  
        return decoded_wordsm_search(decoder, decoder_input, decoder_hidden, max_length)  
        return decoded_words


import sacrebleu
def calculate_bleu(predictions, labels):
	"""
	Only pass a list of strings 
	"""
	# tthis is ony with n_gram = 4

	bleu = sacrebleu.raw_corpus_bleu(predictions, [labels], .01).score
	return bleu

def test_model(encoder, decoder,search, test_pairs, lang1,max_length):
    # for test, you only need the lang1 words to be tokenized,
    # lang2 words is the true labels
    encoder_inputs = [pair[0] for pair in test_pairs]
    true_labels = [pair[1] for pair in test_pairs]
    translated_predictions = []
    for i in range(len(encoder_inputs)): 
        if i% 100== 0:
            print(i)
        e_input = encoder_inputs[i]
        decoded_words = evaluate(encoder, decoder, torch.cuda.LongTensor(e_input), max_length)
        translated_predictions.append(" ".join(decoded_words))
    return calculate_bleu(translated_predictions, true_labels)

input_lang = load_cpickle_gc("preprocessed_data_no_elmo/iwslt-zh-eng/preprocessed_no_elmo_zhlang")
target_lang = load_cpickle_gc("preprocessed_data_no_elmo/iwslt-zh-eng/preprocessed_no_elmo_englang")
pairs =  load_cpickle_gc("preprocessed_data_no_elmo/iwslt-zh-eng/preprocessed_no_indices_pairs_train")
lengths = [len(s[0].split()) for s in pairs]
print(max(lengths))

hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(target_lang.n_words, hidden_size).to(device)
n_iters = 213376 # length of chinese 
n_epochs = 3
max_length = 530 # for chinese
trainIters(encoder, decoder, n_iters,n_epochs, input_lang, target_lang, max_length)




