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
import pdb

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from model_architectures import Encoder_RNN, Decoder_RNN
from data_prep import prepareData, tensorsFromPair, prepareNonTrainDataForLanguagePair, load_cpickle_gc
from inference import generate_translation
from misc import timeSince, load_cpickle_gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 32
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
teacher_forcing_ratio = 1.0
max_length_chinese = 530 # for chinese
max_length_viet = 759
max_generation = 619 
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

class EncoderRNNBidirectionalBatch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNNBidirectionalBatch, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, dropout=0.1)

    def forward(self, sents, sent_lengths):
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # We sort and then do pad packed sequence here. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = np.array(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = self.init_hidden(batch_size)
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        # change the order back
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device)) 
        return rnn_out, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=device)

class DecoderRNNBidirectionalBatch(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderRNNBidirectionalBatch, self).__init__()
        self.hidden_size = hidden_size*2
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size*2)
        self.gru = nn.GRU(hidden_size*2,  hidden_size*2 , dropout=0.2)
        self.out = nn.Linear(hidden_size*2, hidden_size*4) # sincec output_size >> hidden_size, we increase 
        self.out2 = nn.Linear(hidden_size*4, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.leaky =  torch.nn.LeakyReLU()

    def forward(self, sents, sent_lengths, hidden):
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # sents is the original, and try  to see if self.hidden  is the same as sents. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = np.array(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = hidden
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device))
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        output = self.leaky(self.out(rnn_out))
        output = self.softmax(self.out2(output))
        return output, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=device)


class DecoderRNNBidirectionalBatchWithAttenntion(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderRNNBidirectionalBatch, self).__init__()
        self.hidden_size = hidden_size*2
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size*2)
        self.gru = nn.GRU(hidden_size*2,  hidden_size*2 , dropout=0.2)
        self.out = nn.Linear(hidden_size*2, hidden_size*4) # sincec output_size >> hidden_size, we increase 
        self.out2 = nn.Linear(hidden_size*4, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.leaky =  torch.nn.LeakyReLU()
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)


    def forward(self, sents, sent_lengths, hidden, encoder_ouputs):

        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # sents is the original, and try  to see if self.hidden  is the same as sents. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = np.array(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # here, we make the attention weights based on the encoder_outputs and the current 
        # inputs, which is equal to y1, y2,...yn in the  traget language because 
        # we pass in the whole batch. 
        attn_weights = F.softmax(self.attn(torch.cat((embedded, encoder_outputs), 1)), dim=1)
        # sum h_ialpha_i 
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # now this is the context vector, which takes the place of the hidden 
        # state in the no-attention case.
        # and then we go through the rest of this decoder like we do without attention.  
        self.hidden =attn_applied 
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device))
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        output = self.leaky(self.out(rnn_out))
        output = self.softmax(self.out2(output))
        return output, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=device)

#train_idx_pairs = load_cpickle_gc("train_vi_en_idx_pairs")
class EncoderRNNBidirectionalBatch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNNBidirectionalBatch, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, dropout=0.1)

    def forward(self, sents, sent_lengths):
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # We sort and then do pad packed sequence here. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = np.array(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = self.init_hidden(batch_size)
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        # change the order back
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device)) 
        return rnn_out, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=device)


class DecoderRNNBidirectionalBatch(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderRNNBidirectionalBatch, self).__init__()
        self.hidden_size = hidden_size*2
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size*2)
        self.gru = nn.GRU(hidden_size*2,  hidden_size*2 , dropout=0.2)
        self.out = nn.Linear(hidden_size*2, hidden_size*4) # sincec output_size >> hidden_size, we increase 
        self.out2 = nn.Linear(hidden_size*4, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.leaky =  torch.nn.LeakyReLU()

    def forward(self, sents, sent_lengths, hidden):
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # sents is the original, and try  to see if self.hidden  is the same as sents. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = np.array(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = hidden
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device))
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        output = self.leaky(self.out(rnn_out))
        output = self.softmax(self.out2(output))
        return output, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=device)

class EncoderRNNBidirectionalBatch(nn.Modeul):
    def __init__(self, input_size, hidden_size):

class EncoderRNNBidirectionalBatch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNNBidirectionalBatch, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, dropout=0.1)

    def forward(self, sents, sent_lengths):
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # We sort and then do pad packed sequence here. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = np.array(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = self.init_hidden(batch_size)
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        # change the order back
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device)) 
        return rnn_out, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=device)


class DecoderRNNBidirectionalBatch(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderRNNBidirectionalBatch, self).__init__()
        self.hidden_size = hidden_size*2
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size*2)
        self.gru = nn.GRU(hidden_size*2,  hidden_size*2 , dropout=0.2)
        self.out = nn.Linear(hidden_size*2, hidden_size*4) # sincec output_size >> hidden_size, we increase 
        self.out2 = nn.Linear(hidden_size*4, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.leaky =  torch.nn.LeakyReLU()

    def forward(self, sents, sent_lengths, hidden):
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # sents is the original, and try  to see if self.hidden  is the same as sents. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = np.array(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = hidden
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device))
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        output = self.leaky(self.out(rnn_out))
        output = self.softmax(self.out2(output))
        return output, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=device)

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
            sents is a tensor with the shape (batch_size, padded_length )
            when we evaluate sentence by sentence, you evaluate it with batch_size = 1, padded_length.
            [[1, 2, 3, 4]] etc. 
        '''
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # We sort and then do pad packed sequence here. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = np.array(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = self.init_hidden(batch_size)
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        # change the order back
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device)) 
        
        # **TODO**: What is rnn_out - for attention. 
        return rnn_out, self.hidden

class Decoder_Batch_RNN(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder_Batch_RNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, sents, sent_lengths, hidden):
        '''
        For evaluate, you compute [batch_size x ] [[1]]
        '''
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        # sents is the original, and try  to see if self.hidden  is the same as sents. 
        descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
        descending_lengths = np.array(descending_lengths)
        descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
        
        # get embedding
        embed = self.embedding(descending_sents)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        
        # fprop though RNN
        self.hidden = hidden
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        
        change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
        self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device))
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # rnn_out is batch_size x 28 x 256
        """      
        final_hidden = self.hidden
        final_hidden = final_hidden.view(final_hidden.size(1), final_hidden.size(0), -1)
        first_hidden = hidden
        first_hidden = first_hidden.view(first_hidden.size(1), first_hidden.size(0), -1)
        
        rnn_out = torch.cat((first_hidden, rnn_out, final_hidden), 1)        
        """
#         rnn_out = rnn_out.view(-1, rnn_out.size(2))
        output = self.softmax(self.out(rnn_out))
        # now output is the size 28 by 31257 (vocab size)
        return output, self.hidden

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


def beam_search(decoder, decoder_input, hidden, max_length, k, target_lang):
    candidates = [(decoder_input, 1, hidden)]
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
            if c_sequence[-1] == EOS_token:
                completed_translations.append((c_sequence, c_score))
                k = k - 1
            else:
                next_word_probs, hidden = decoder(torch.cuda.LongTensor([c_sequence[-1]]).view(1, 1), torch.cuda.LongTensor([1]),torch.cuda.FloatTensor(c_hidden)) 
                s = next_word_probs.size()
                # 1 x 3 x 256
                #pdb.set_trace()
                next_word_probs = next_word_probs[0, 2, :]
                # in the worst-case, one sequence will have the highest k probabilities
                # so to save computation, only grab the k highest_probability from each candidate sequence
                top_probs, top_idx = torch.topk(next_word_probs, k)
                # 1 x 1 x k right now. 
                # check that beam search actually works the way we think it is working. 
                for i in range(len(top_probs)):
                    word = torch.from_numpy(np.array(top_idx[0][0][i]).reshape(1, 1)).to(device)
                    new_score = c_score + top_probs[i]
                    potential_candidates.append((torch.cat((c_sequence, word)).to(device), new_score, hidden))
        candidates = sorted(potential_candidates, key= lambda x: x[1], reverse=True)[0:k] 
        potential_candidates = []

    completed = completed_translations + candidates
    completed = sorted(completed, key= lambda x: x[1], reverse=True)[0] 
    final_translation = []
    for x in completed[0]:
        final_translation.append(target_lang.index2word[x.squeeze().item()])
    return final_translation

def generate_translation(encoder, decoder, sentence, max_length, target_lang, search="greedy", k = None):
    """ 
    @param max_length: the max # of words that the decoder can return
    @returns decoded_words: a list of words in target language
    """    
    with torch.no_grad():
        input_tensor = sentence
        input_length = sentence.size()[0]
        
        # encode the source sentence
        encoder_hidden = encoder.init_hidden(1)
        encoder_output, encoder_hidden = encoder(input_tensor.view(1, -1),torch.tensor([input_length]))
        # start decoding
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        
        if search == 'greedy':
            decoded_words = greedy_search(decoder, decoder_input, decoder_hidden, max_length)
        elif search == 'beam':
            if k == None:
                k = 5 # since k = 2 preforms badly
            decoded_words = beam_search(decoder, decoder_input, decoder_hidden, max_length, k, target_lang)  

        return decoded_words

import sacrebleu
def calculate_bleu(predictions, labels):
    """
    Only pass a list of strings 
    """
    # tthis is ony with n_gram = 4

    bleu = sacrebleu.raw_corpus_bleu(predictions, [labels], .01).score
    return bleu


def test_model(encoder, decoder, search, test_pairs, lang2, max_length):
    # for test, you only need the lang1 words to be tokenized,
    # lang2 words is the true labels
    encoder_inputs = [pair[0] for pair in test_pairs]
    true_labels = [pair[1] for pair in test_pairs]
    translated_predictions = []
    for i in range(len(encoder_inputs)):
        e_input = encoder_inputs[i]
        decoded_words = generate_translation(encoder, decoder, e_input, max_length, lang2, search=search)
        translated_predictions.append(" ".join(decoded_words))
    start = time.time()
    print(translated_predictions[0])
    print(true_labels[0])
    bleurg = calculate_bleu(translated_predictions, true_labels)
    return bleurg

def save_model(encoder, decoder, val_accs, train_accs, title):
    val_accs = np.array(val_accs) # this is the BLEU score. 
    max_val = val_accs.max() 
    train_accs = np.array(train_accs)
    link = title.replace(" ", "")
    torch.save(encoder.state_dict(), "output/"+link + "encodermodel_states")
    torch.save(decoder.state_dict(), "output/"+link + "decodermodel_states")
    pickle.dump(val_accs, open("output/"+link + "val_accuracies", "wb"))
    pickle.dump(train_accs, open("output/"+link + "train_accuracies", "wb"))
    pickle.dump(max_val, open("output/"+link + "maxvalaccis"+str(max_val), "wb"))
    # this is when you want to overlay
    num_in_epoch = np.shape(train_accs)[1]
    num_epochs = np.shape(train_accs)[0]
    x_vals = np.arange(0, num_epochs, 1.0/float(num_in_epoch))
    fig = plt.figure()
    plt.title(title)
    # plot the title of this data. 
    plt.plot(x_vals, train_accs.flatten(), label="Training Accuracy (NLLoss)")
    plt.plot(x_vals, val_accs.flatten(), label="Validation Accuracy (BLEU score)")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy of Model")
    plt.xlabel("Epochs (Batch Size 32)")
    plt.ylim(0,100)
    plt.xlim(0, num_epochs)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xticks(np.arange(num_epochs + 1))
    fig.savefig("output/"+link+"graph.png")

    
def trainIters(encoder, decoder, n_epochs, pairs, validation_pairs, lang1, lang2, search, title, max_length_generation,  print_every=1000, plot_every=1000, learning_rate=0.0001):
    """
    lang1 is the Lang object for language 1 
    Lang2 is the Lang object for language 2
    Max length generation is the max length generation you want 
    """
    start = time.time()
    plot_losses = []
    val_losses = [] 
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
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            outputs, encoder_hidden = encoder(sent1_batch, sent1_length_batch)
            # encoder outputs is currently size 696 x 256
            encoder_hidden_batch = encoder_hidden 
            decoder_hidden = encoder_hidden_batch
            
            decoder_input = torch.tensor([[SOS_token]], device=device)
            use_teacher_forcing = True
            
            loss = 0
            count = 0
            outputs, decoder_hidden = decoder(sent2_batch, sent2_length_batch, decoder_hidden)
            
            # outputs should be batch_size x vocab ize 
            for i in range(len(sent2_batch)):
                l = sent2_length_batch[i]
                for j in range(l):
                    o = outputs[i][j].view(1, -1)
                    s = torch.tensor([sent2_batch[i][j]]).to(device)
                    loss += criterion(o, s) # this will ignore if s is "EOS"
                    count += 1
            print_loss_total += loss.item()
            plot_loss_total += loss.item()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            # we also have tomaks when it's an eOS tag. 
            if (step+1) % print_every == 0:
                # lets train and polot at the same time. 
                print_loss_avg = print_loss_total / count
                print_loss_total = 0
                print('TRAIN SCORE %s (%d %d%%) %.4f' % (timeSince(start, step / n_epochs),
                                             step, step / n_epochs * 100, print_loss_avg))
                #v_loss = test_model(encoder, decoder, search, validation_pairs, lang1, max_length=max_length_generation)
                # returns bleu score
                #print("VALIDATION BLEU SCORE: "+str(v_loss))
                #val_loss.append(v_loss)
                plot_loss_avg = plot_loss_total / plot_every
                plot_loss.append(plot_loss_avg)
                plot_loss_total = 0

        plot_losses.append(plot_loss)
        val_losses.append(val_loss)
        print("AVERAGE PLOT LOSS")
        print(np.mean(plot_loss))
        #encoder_scheduler.step(np.mean(plot_loss)) # this isnt' really doing anything. 
        #decoder_scheduler.step(np.mean(plot_loss))
        #save_model(encoder, decoder, val_losses, plot_losses, title)
    assert len(val_losses) == len(plot_losses)
    save_model(encoder, decoder, val_losses, plot_losses, title)

hidden_size = 256
print(BATCH_SIZE)
input_lang = pickle.load(open("preprocessed_data_no_elmo/iwslt-vi-eng/preprocessed_no_elmo_vilang", "rb"))
target_lang = pickle.load(open("preprocessed_data_no_elmo/iwslt-vi-eng/preprocessed_no_elmo_englang", "rb"))
train_idx_pairs = load_cpickle_gc("preprocessed_data_no_elmo/iwslt-vi-eng/preprocessed_no_indices_pairs_train_tokenized_sample")
val_pairs = load_cpickle_gc("preprocessed_data_no_elmo/iwslt-vi-eng/preprocessed_no_indices_pairs_validation_tokenized")
train_dataset = LanguagePairDataset(train_idx_pairs)
# is there anything in the train_idx_pairs that is only 0s right noww instea dof padding. 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=language_pair_dataset_collate_function,
                                          )

encoder1 = Encoder_Batch_RNN(input_lang.n_words, hidden_size).to(device)
decoder1 = Decoder_Batch_RNN(target_lang.n_words, hidden_size).to(device)
args = {
    'n_epochs': 10,
    'learning_rate': 0.001,
    'search': 'beam',
    'encoder': encoder1,
    'decoder': decoder1,
    'lang1': input_lang, 
    'lang2': target_lang,
    "pairs":train_idx_pairs, 
    "validation_pairs": val_pairs, 
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

