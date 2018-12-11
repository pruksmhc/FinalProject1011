import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embed = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embed, hidden)
        # output and hidden are the same vectors
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    

class Decoder_RNN(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder_RNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embed = self.embedding(input).view(1, 1, -1)
        embed = F.relu(embed)
        output, hidden = self.gru(embed, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

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
        self.softmax = nn.LogSoftmax(dim=2)
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, sents, sent_lengths, hidden, train=True):
        '''
        For evaluate, you compute [batch_size x ] [[1]]
        '''
        batch_size = sents.size()[0]
        sent_lengths = list(sent_lengths)
        if train:
            # sents is the original, and try  to see if self.hidden  is the same as sents. 
            descending_lengths = [x for x, _ in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
            descending_indices = [x for _, x in sorted(zip(sent_lengths, range(len(sent_lengths))), reverse=True)]
            descending_lengths = np.array(descending_lengths)
            descending_sents = torch.index_select(sents, 0, torch.tensor(descending_indices).to(device))
            sents = descending_sents  
        # get embedding
        embed = self.embedding(sents)
        # pack padded sequence
        if train:
            embed = torch.nn.utils.rnn.pack_padded_sequence(embed, descending_lengths, batch_first=True)
        # fprop though RNN
        self.hidden = hidden
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        if train:
            change_it_back = [x for _, x in sorted(zip(descending_indices, range(len(descending_indices))))]
            self.hidden = torch.index_select(self.hidden, 1, torch.LongTensor(change_it_back).to(device))
            rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
            # this basically pads it back, and then we have to actually get to the right order. 
            # 0th dimension since rnn_out is of size 32. 
            rnn_out = torch.index_select(rnn_out, 0, torch.LongTensor(change_it_back).to(device))
        output = self.softmax(self.out(rnn_out))
        # now output is the size 28 by 31257 (vocab size)
        return output, self.hidden
