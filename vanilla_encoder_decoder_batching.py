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


BATCH_SIZE = 16
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
teacher_forcing_ratio = 0.8
max_length_chinese = 530 # for chinese
max_length_viet = 759
max_generation = 619 
MAX_LENGTH = 20 
#MAX_LENGTH = max_length_viet 
import numpy as np
import torch
from torch.utils.data import Dataset

class LanguagePairDataset(Dataset):
    
    def __init__(self, sent_pairs): 
        # this is a list of sentences 
        self.sent_pairs_list = sent_pairs

    def __len__(self):
        return len(self.sent_pairs_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        sent1 = self.sent_pairs_list[key][0][:MAX_LENGTH]
        sent2 = self.sent_pairs_list[key][1][:MAX_LENGTH]
        return [sent1, sent2, len(sent1), len(sent2)]

def language_pair_dataset_collate_function(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    sent1_list = []
    sent1_length_list = []
    sent2_list = []
    sent2_length_list = []
    # padding
    # NOW PAD WITH THE MAXIMUM LENGTH OF THE FIRST and second batches 
    max_length_1 = max([len(x[0]) for x in batch])
    max_length_2 = max([len(x[1]) for x in batch])
    for datum in batch:
        padded_vec_1 = np.pad(np.array(datum[0]).T.squeeze(), pad_width=((0,max_length_1-len(datum[0]))), 
                                mode="constant", constant_values=PAD_token)
        padded_vec_2 = np.pad(np.array(datum[1]).T.squeeze(), pad_width=((0,max_length_2-len(datum[1]))), 
                                mode="constant", constant_values=PAD_token)
        sent1_list.append(padded_vec_1)
        sent2_list.append(padded_vec_2)
        sent1_length_list.append(len(datum[0]))
        sent2_length_list.append(len(datum[1]))
    return [torch.from_numpy(np.array(sent1_list)), torch.cuda.LongTensor(sent1_length_list), 
            torch.from_numpy(np.array(sent2_list)), torch.cuda.LongTensor(sent2_length_list)]


def save_model(encoder, decoder, title):
    link = title.replace(" ", "")
    torch.save(encoder.state_dict(), "output/"+link + "encodermodel_states")
    torch.save(decoder.state_dict(), "output/"+link + "decodermodel_states")

def make_graph(encoder, decoder, val_accs, train_accs, title):
    print("SAVE")
    val_accs = np.array(val_accs) # this is the BLEU score. 
    max_val = val_accs.max() 
    train_accs = np.array(train_accs)
    link = title.replace(" ", "")
    pickle.dump(val_accs, open("output/"+link + "val_accuracies", "wb"))
    pickle.dump(train_accs, open("output/"+link + "train_accuracies", "wb"))
    pickle.dump(max_val, open("output/"+link + "maxvalaccis"+str(max_val), "wb"))
    # this is when you want to overlay
    num_in_epoch = np.shape(train_accs)[1]
    num_epochs = np.shape(train_accs)[0]
    x_vals_train = np.arange(0, num_epochs, 1.0/float(num_in_epoch))
    num_in_epoch = np.shape(val_accs)[1]
    num_epochs = np.shape(val_accs)[0]
    x_vals_val = np.arange(0, num_epochs, 1.0/float(num_in_epoch))
    fig = plt.figure()
    plt.title(title)
    # plot the title of this data. 
    plt.plot(x_vals_train, train_accs.flatten(), label="Training Loss (NLLoss)")
    plt.plot(x_vals_val, val_accs.flatten(), label="Validation Accuracy (BLEU score)")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy of Model")
    plt.xlabel("Epochs (Batch Size 32)")
    plt.ylim(0, 50) # for loss
    plt.xlim(0, num_epochs)
    plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.xticks(np.arange(num_epochs + 1))
    fig.savefig("output/"+link+"graph.png")


def train(sent1_batch,  sent1_length_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, sent2_batch, sent2_length_batch, criterion, count):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    outputs, encoder_hidden = encoder(sent1_batch, sent1_length_batch)
    # encoder outputs is currently size 696 x 256
    encoder_hidden = encoder_hidden[0]
    loss = 0
    output_translations = []
    for i in range(len(sent1_batch)):
        # going over each batch size
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden[i]
        target_length = sent2_length_batch[i] # get the length of the current sentence
        target_tensor = sent2_batch[i]
        output_translation = []
        # now this is 
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                count += 1
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden.view(1, 1, -1))
                output_translation.append(decoder_output)
                loss += criterion(decoder_output, torch.cuda.LongTensor([target_tensor[di]]) )# adding per each token. 
                decoder_input = target_tensor[di]  # Teacher forcing
                if decoder_input == PAD_token: # so that it learns to predict EOS. 
                    break # since we are batching here

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden.view(1, 1, -1))
                output_translation.append(decoder_output)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                count += 1
                loss += criterion(decoder_output, torch.cuda.LongTensor([target_tensor[di]]))
                if decoder_input.item() == EOS_token:
                    break
        output_translations.append(output_translation)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(),  output_translations, count

def trainIters(encoder, decoder, n_epochs, pairs, validation_pairs, lang1, lang2, search, title, max_length_generation,  print_every=1000, plot_every=1000, learning_rate=0.0001):
    """
    lang1 is the Lang object for language 1 
    Lang2 is the Lang object for language 2
    Max length generation is the max length generation you want 
    """
    start = time.time()
    plot_losses = []
    val_losses = [] 
    count = 0 
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    val_loss_total = 0
    plot_val_loss = 0
    encoder_optimizer = torch.optim.Adadelta(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adadelta(decoder.parameters(), lr=learning_rate)
    #encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode="min")
    #decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode="min")

    criterion = nn.NLLLoss(ignore_index=PAD_token) # this ignores the padded token. 
    plot_loss =[]
    val_loss = []
    for epoch in range(n_epochs):

        plot_loss = []
        val_loss = []
        for step, (sent1s, sent1_lengths, sent2s, sent2_lengths) in enumerate(train_loader):
            encoder.train() # what is this for?
            decoder.train()
            sent1_batch, sent2_batch = sent1s.to(device), sent2s.to(device) 
            sent1_length_batch, sent2_length_batch = sent1_lengths.to(device), sent2_lengths.to(device)
            loss, output_translations, count = train(sent1_batch, sent1_length_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, sent2_batch, sent2_length_batch, criterion, count) # Yikes, what is this. 
            i = 0  #look at the first output ranslation
            output = output_translations[i]
            translated = []
            answer = []
            for j in range(len(output)):
                token = torch.argmax(output[j][0])[0] # you get the index
                translated.append(lang2.index2word[token.squeeze().item()])
                answer.append(lang2.index2word[sent2_batch[i][j].squeeze().item()])
            print(answer)
            print("translated prediction")
            print(translated) 
            # lets output what it's actually getting as itsoutput of teh decoder here. 
            # check if there is an SOS here as well. 
            print_loss_total += loss
            plot_loss_total += loss
            # we also have tomaks when it's an eOS tag. 
            if (step+1) % print_every == 0:
                # lets train and polot at the same time. 
                print_loss_avg = print_loss_total / count
                count = 0
                print_loss_total = 0
                print('TRAIN SCORE %s (%d %d%%) %.4f' % (timeSince(start, step / n_epochs),
                                             step, step / n_epochs * 100, print_loss_avg))
                with torch.no_grad():
                    v_loss = test_model(encoder, decoder, search, validation_pairs, lang2, max_length=max_length_generation)
                # returns bleu score
                print("VALIDATION BLEU SCORE: "+str(v_loss))
                val_loss.append(v_loss)
                plot_loss.append(print_loss_avg)
                # save it every time it hits the step now. 
                save_model(encoder, decoder, title)
                sys.stdin.flush()
                plot_loss_total = 0

        plot_losses.append(plot_loss)
        val_losses.append(val_loss)
        print("AVERAGE PLOT LOSS")
        print(np.mean(plot_loss))
        sys.stdin.flush()
        #encoder_scheduler.step(np.mean(plot_loss)) # this isnt' really doing anything. 
        #decoder_scheduler.step(np.mean(plot_loss))
        save_model(encoder, decoder, title)
        make_graph(encoder, decoder, val_losses, plot_losses, title)
    assert len(val_losses) == len(plot_losses)
    save_model(encoder, decoder, title)
    make_graph(encoder, decoder, val_losses, plot_losses, title)

hidden_size = 256
print(BATCH_SIZE)
input_lang = pickle.load(open("preprocessed_data_no_elmo/iwslt-vi-eng/preprocessed_no_elmo_vilang", "rb"))
target_lang = pickle.load(open("preprocessed_data_no_elmo/iwslt-vi-eng/preprocessed_no_elmo_englang", "rb"))
train_idx_pairs = load_cpickle_gc("preprocessed_data_no_elmo/iwslt-vi-eng/preprocessed_no_indices_pairs_train_tokenized")
val_pairs = pickle.load(open("preprocessed_data_no_elmo/iwslt-vi-eng/preprocessed_no_indices_pairs_validation_tokenized", "rb"))
train_dataset = LanguagePairDataset(train_idx_pairs)
# is there anything in the train_idx_pairs that is only 0s right noww instea dof padding. 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=language_pair_dataset_collate_function,
                                          )

encoder1 = Encoder_Batch_RNN(input_lang.n_words, hidden_size).to(device)
decoder1 = Decoder_RNN(target_lang.n_words, hidden_size).to(device)

args = {
    'n_epochs': 10,
    'learning_rate': 0.001,
    'search': 'beam',
    'encoder': encoder1,
    'decoder': decoder1,
    'lang1': input_lang, 
    'lang2': target_lang,
    "pairs":train_idx_pairs[10000:], 
    "validation_pairs": val_pairs[:200], 
    "title": "Training Curve for Basic 1-Directional Encoder Decoder Model With LR = 0.001 no decoder batching",
    "max_length_generation": 2, 
    "plot_every": 500, 
    "print_every": 500
}

"""
We take the input sentence as the length of the maximum generating sentence 
We follow https://arxiv.org/pdf/1406.1078.pdf 
and use the Adadelta optimizer
Have max_length_generation

"""
print(BATCH_SIZE)

trainIters(**args)

