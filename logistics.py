
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