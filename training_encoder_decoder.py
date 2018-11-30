# For the normal tokenization, we use iwslt-vi-en/dev.tok._language
teacher_forcing_ratio = 0.5
import time
import math
import pdb
import pickle
from ModelsWithoutElmo import * 
import torch
import torch.nn as nn
import random
import itertools
from DataLoader import * 
import numpy as np
SOS_token = 2
EOS_token = 3
MAX_LENGTH_VI_EN = 1310
# max_length_zh_en
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


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

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


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

def train(input_tensor, target_tensor, length1, length2, order_target_for_source, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length,attention=False):
    # this is only for one batch clump. 
    batch_size = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    criterion = nn.NLLLoss()

    encoder_outputs = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)
    # Here, teh outputs is hidden_Size x mmax_lengt
    # TODO: Add pad_apkced and pack_padded here. 
    #embed1 = torch.nn.utils.rnn.pack_padded_sequence(embed1, length1.numpy(), batch_first=True)
    loss = 0
    # the pack_padded_input goes into here. 
    for ei in range(input_length):
    	input_slice = torch.LongTensor([[x[ei]]for x in input_tensor])
    	encoder_output, encoder_hidden = encoder(input_slice, encoder_hidden, length1) # take the slice through time
    	encoder_output = encoder_output.view((batch_size, encoder.hidden_size))
    	for i in range(len(encoder_output)):
    		encoder_outputs[i, ei] = encoder_output[i]
    # here, encoder_output must be of the size [[1,], [2], [3] , ] 
    # add SOS before decoding. <eng> EOS SOS <fra>
    SOS_tokens = [[SOS_token] for i in range(len(input_tensor))] # 32 
    decoder_input = torch.tensor(SOS_tokens, device=device)
    # TODO Now we match from the order of order_1 to the order of order_2, trnsform the encoder hidden 
    encoder_hidden = encoder_hidden.view((batch_size, encoder.hidden_size))
    encoder_hidden_aligned = torch.index_select(encoder_hidden, 0, order_target_for_source)
    decoder_hidden = encoder_hidden_aligned.view((1, batch_size, encoder.hidden_size))

    # encoder_hidden is the last hidden state of the encoder. 
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            else:
            	decoder_output, decoder_hidden, = decoder(decoder_input, decoder_hidden, target_length)
            target_tensor_slice = torch.LongTensor([[x[di]] for x in target_tensor])
            loss += criterion(decoder_output, target_tensor[:,di])
            decoder_input = target_tensor_slice
            # we also slicethrough a through a batch.
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
            	decoder_output, decoder_hidden, = decoder(decoder_input, decoder_hidden, target_length)
            topv, topi = decoder_output.topk(1) # returns a 1 x 32 tensor. 
            decoder_input = topi.squeeze().detach()  # detach from history as input
            # for those that have already generated an EOS tag, just ignore
            loss += criterion(decoder_output,target_tensor[:,di])
            # TODO - Figure out mechanism to stop when there's an eOS tag generated
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_epochs, lang1, lang2,  print_every=1000, plot_every=100, learning_rate=0.001):
    # tHIS trains for various epochs, and then 
    # this basically 
    pairs = load_cpickle_gc("preprocessed_data_no_elmo/iwslt-"+lang1.name+"-"+lang2.name+"/preprocessed_no_indices_pairs_train")
    # just to test, let's try making the size 32 so that we know what to expect
    BATCH_SIZE = 32
    train_dataset = TranslationDataset(pairs, lang1, lang2)
    train_loader =  torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=translation_collate_func_concat,
                                           	 	shuffle=True)

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    for i in range(1,n_epochs+1):
        # so now, becuase we've already indeed, 
        criterion = nn.NLLLoss()
        # framing it as a categorical loss function. 
        for iter, (source, target, length1, length2, order_target_for_source) in enumerate(train_loader):
        	print(iter)
        	loss = train(source, target, length1, length2, order_target_for_source, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH_VI_EN)
        	print_loss_total += loss
        	plot_loss_total += loss
        	if iter % print_every == 0:
        		print_loss_avg = print_loss_total / print_every
        		print_loss_total = 0
        		print('TRAIN SCORE '+str(print_loss_avg) +'at iteration '+str(iter))
        	if iter % plot_every == 0:
        		plot_loss_avg = plot_loss_total / plot_every
        		plot_losses.append(plot_loss_avg)
        		plot_loss_total = 0
    pickle.dump(encoder, open("encoder_" +lang1+"_"+lang2, "wb"))
    pickle.dump(decoder, open("decoder_" +lang1+"_"+"lang2", "wb"))
    showPlot(plot_losses)

def trainNoAttention(lang1, lang2):
    hidden_size = 256
    batch_size = 32
    lang_object_input =  load_cpickle_gc("preprocessed_data_no_elmo/iwslt-"+lang1+"-"+lang2+"/preprocessed_no_elmo_"+lang1+"lang")
    lang_object_output =  load_cpickle_gc("preprocessed_data_no_elmo/iwslt-"+lang1+"-"+lang2+"/preprocessed_no_elmo_"+lang2+"lang")
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    encoder = EncoderRNN(lang_object_input.n_words,batch_size, hidden_size).to(device)
    decoder = DecoderRNN(lang_object_output.n_words, batch_size, hidden_size).to(device)
    trainIters(encoder, decoder, 10, lang_object_input, lang_object_output, print_every=5) #train on a small subset

def trainWithAttention(lang1, lang2):
    hidden_size = 1024
    train_data = pickle.load(open("preprocessed_data/iwslt-vi-en/train_indexed", "rb"))
    lang_object_input =  pickle.load(open("preprocessed_data/iwslt-"+lang1+"-"+lang2+"/train_input", "rb"))
    lang_object_output =  pickle.load(open("preprocessed_data/iwslt-"+lang1+"-"+lang2+"/train_output", "rb"))
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    weights_en_torch = torch.load("weights_en_torch")
    weights_vi_torch = torch.load("weights_vi_torch")
    encoder = EncoderRNN(lang_object_input.n_words, hidden_size, weights_vi_torch).to(device)
    decoder = DecoderRNN(hidden_size, lang_object_output.n_words, weights_en_torch).to(device)
    trainIters(encoder, decoder, 750, 10, lang_object_input, lang_object_output, print_every=5) #train on a small subset


trainNoAttention("vi", "eng")
