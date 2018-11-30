import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F
import numpy as np

SOS_token = 2
EOS_token = 3
MAX_LENGTH_VI_EN = 1310
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size) # if you pad, you NEED to do batch first

    def forward(self, input, hidden, input_length):
    	embedded = self.embedding(input).view((1, 32, 1024))
    	output, hidden = self.gru(embedded, hidden)
    	return output, hidden

    def initHidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(1, batch_size, self.hidden_size)

        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, target_length):
    	output = self.embedding(input).view((1, 32, 1024))
    	output = F.relu(output)
    	output, hidden = self.gru(output, hidden)
    	output = self.softmax(self.out(output[0]))
    	return output, hidden

    def initHidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(1, batch_size, self.hidden_size)

        return hidden