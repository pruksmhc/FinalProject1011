from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import pickle
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import pdb

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from model_architectures import Encoder_RNN, Decoder_RNN
from data_prep import prepareTrainData, tensorsFromPair, prepareNonTrainDataForLanguagePair, load_cpickle_gc
from inference import generate_translation
from misc import timeSince, load_cpickle_gc

device = "cpu"


BATCH_SIZE = 32
PAD_token = 0
PAD_TOKEN = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
teacher_forcing_ratio = 1.0


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
        sent1 = self.sent_pairs_list[key][0]
        sent2 = self.sent_pairs_list[key][1]
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



class Encoder_Batch_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder_Batch_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, sents, sent_lengths):
        '''
            sents is (batch_size by padded_length)
            when we evaluate sentence by sentence, you evaluate it with batch_size = 1, padded_length.
            [[1, 2, 3, 4]] etc. 
        '''
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # We sort and then do pad packed sequence here. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = torch.tensor(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = self.init_hidden(batch_size)
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # rnn_out is 32 by 72 by 256
        
        # change the order back
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device))  
        rnn_out = torch.index_select(rnn_out, 0, torch.LongTensor(change_it_back).to(device)) 
        
        return rnn_out, self.hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.softmax = torch.nn.Softmax(dim=1)
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):

        # Create variable to store attention energies
        # hidden is 32 by 256
        # encoder_outputs is 32 by 72 by 256
        hidden = hidden[0]
        batch_size = hidden.size()[0]
        attn_energies = []
        for i in range(batch_size):
            attn_energies.append(self.score(hidden[i], encoder_outputs[i]))
        
        # attn_energies is 32 by 72
        attn_energies = self.softmax(torch.stack(attn_energies))
        
        context_vectors = []
        for i in range(batch_size):
            context_vectors.append(torch.matmul(attn_energies[i], encoder_outputs[i]))
                
        context_vectors = torch.stack(context_vectors)
        
        return context_vectors
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':            
            # hidden is 1 by 256
            # encoder_output is 22 by 256
            encoder_output = torch.transpose(encoder_output, 0, 1)
            # encoder_output is 256 by 22
            energy = torch.matmul(hidden, encoder_output)
            return energy
        
        elif self.method == 'general':
            # hidden is 1 by 256
            # encoder_output is 256 by 22
            energy = torch.matmul(hidden, self.attn(encoder_output))
            return energy
        
        elif self.method == 'concat':
            len_encoder_output = encoder_output.size()[1]
            # hidden is 1 by 256
            # encoder_output is 256 by 22
            hidden = torch.transpose(hidden, 0, 1)
            # hidden is 256 by 1
            hidden = hidden.repeat(hidden_size, len_encoder_output)
            # hidden is 256 by 22
            concat = torch.cat((hidden, encoder_output), dim=0)
            # concat is 512 by 22
            # self.attn(concat) --> 256 by 22
            energy = torch.matmul(self.v, F.tanh(self.attn(concat)))
            return energy
        

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_TOKEN)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        context = self.attn(rnn_output, encoder_outputs)
        # context is 32 by 256

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        # rnn_output is 32 by 256
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # output is 32 by vocab_size
        output = self.LogSoftmax(output)

        # Return final output, hidden state
        return output, hidden
    

def trainIters(encoder, decoder, n_epochs, pairs, validation_pairs, lang1, lang2, search, title, max_length_generation,  print_every=1000, plot_every=1000, learning_rate=0.0001):
    """
    lang1 is the Lang object for language 1 
    Lang2 is the Lang object for language 2
    Max length generation is the max length generation you want 
    """
    start = time.time()
    plot_losses, val_losses = [], []
    val_losses = [] 
    count, print_loss_total, plot_loss_total, val_loss_total, plot_val_loss = 0, 0, 0, 0, 0 
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
            encoder.train()
            decoder.train()
            sent1_batch, sent2_batch = sent1s.to(device), sent2s.to(device) 
            sent1_length_batch, sent2_length_batch = sent1_lengths.to(device), sent2_lengths.to(device)
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            encoder_outputs, encoder_hidden = encoder(sent1_batch, sent1_length_batch)
            # outputs is 32 by 72 by 256
            # encoder_hidden is 1 by 32 by 256
            
            decoder_input = torch.LongTensor([SOS_token] * BATCH_SIZE).view(-1, 1).to(device)
            decoder_hidden = encoder_hidden
            # decoder_input is 32 by 1
            # decoder_hidden is 1 by 32 by 256
                        
            max_trg_len = max(sent2_lengths)
            loss = 0
            
            # Run through decoder one time step at a time using TEACHER FORCING=1.0
            for t in range(max_trg_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # decoder_output is 32 by vocab_size
                # sent2_batch is 32 by 46
                loss += criterion(decoder_output, sent2_batch[:, t])
                decoder_input = sent2_batch[:, t]
             
            loss = loss / max_trg_len.float()
            print_loss_total += loss
            count += 1
            loss.backward()
        
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            if (step+1) % print_every == 0:
                # lets train and plot at the same time. 
                print_loss_avg = print_loss_total / count
                count = 0
                print_loss_total = 0
                print('TRAIN SCORE %s (%d %d%%) %.4f' % (timeSince(start, step / n_epochs),
                                             step, step / n_epochs * 100, print_loss_avg))
                # 42s
#                 v_loss = test_model(encoder, decoder, search, validation_pairs, lang2, max_length=max_length_generation)
                # returns bleu score
#                 print("VALIDATION BLEU SCORE: "+str(v_loss))
#                 val_loss.append(v_loss)
                plot_loss.append(print_loss_avg)
                plot_loss_total = 0
                

    save_model(encoder, decoder, val_losses, plot_losses, title)
    

input_lang, target_lang, train_pairs = prepareTrainData(
    "iwslt-vi-en-processed/train.vi",
    "iwslt-vi-en-processed/train.tok.en",
    input_lang = 'vi',
    target_lang = 'en')

train_idx_pairs = []
for x in train_pairs:
    indexed = list(tensorsFromPair(x, input_lang, target_lang))
    train_idx_pairs.append(indexed)

pickle.dump(input_lang, open("input_lang_vi", "wb"))
pickle.dump(target_lang, open("target_lang_en", "wb"))
pickle.dump(train_idx_pairs, open("train_vi_en_idx_pairs", "wb"))

_, _, val_pairs = prepareTrainData(
    "iwslt-vi-en-processed/dev.vi",
    "iwslt-vi-en-processed/dev.en",
    input_lang = 'vi',
    target_lang = 'en')

val_idx_pairs = []
for x in val_pairs:
    indexed = list(tensorsFromPair(x, input_lang, target_lang))
    val_idx_pairs.append(indexed)

pickle.dump(val_pairs, open("val_pairs", "wb"))
pickle.dump(val_idx_pairs, open("val_idx_pairs", "wb"))


input_lang = load_cpickle_gc("input_lang_vi")
target_lang = load_cpickle_gc("target_lang_en")
train_idx_pairs = load_cpickle_gc("train_vi_en_idx_pairs")
val_idx_pairs = load_cpickle_gc("val_idx_pairs")
val_pairs = load_cpickle_gc("val_pairs")

train_dataset = LanguagePairDataset(train_idx_pairs)
# is there anything in the train_idx_pairs that is only 0s right now instead of padding. 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=language_pair_dataset_collate_function,
                                          )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hidden_size = 256
attn_model = 'dot'
encoder1 = Encoder_Batch_RNN(input_lang.n_words, hidden_size).to(device)
# decoder1 = Decoder_Batch_2RNN(target_lang.n_words, hidden_size).to(device)
decoder1 = LuongAttnDecoderRNN(attn_model, hidden_size, target_lang.n_words, 1).to(device)
args = {
    'n_epochs': 10,
    'learning_rate': 0.1,
    'search': 'beam',
    'encoder': encoder1,
    'decoder': decoder1,
    'lang1': input_lang, 
    'lang2': target_lang,
    "pairs":train_idx_pairs, 
    "validation_pairs": val_idx_pairs[:200], 
    "title": "Training Curve for Basic 1-Directional Encoder Decoder Model With LR = 0.0001",
    "max_length_generation": 20, 
    "plot_every": 10, 
    "print_every": 10
}

"""
We follow https://arxiv.org/pdf/1406.1078.pdf 
and use the Adadelta optimizer

"""
print(BATCH_SIZE)

trainIters(**args)