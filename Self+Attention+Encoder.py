
# coding: utf-8

# In[1]:


import gc
gc.collect()


# In[2]:


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import pickle
import time
import math, copy
from random import randint
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import pdb
import tensorflow as tf

import seaborn
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# from model_architectures import Encoder_RNN, Decoder_RNN
from data_prep import prepareData, tensorsFromPair, prepareNonTrainDataForLanguagePair, load_cpickle_gc
from misc import timeSince, load_cpickle_gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: PAD_token, 1: SOS_token, 2: EOS_token, 3:UNK_token}
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


# In[4]:


BATCH_SIZE = 16
PAD_token = 0
PAD_TOKEN = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
teacher_forcing_ratio = 1.0
attn_model = 'dot'


# In[5]:


train_idx_pairs = load_cpickle_gc("./iwslt-vi-eng/preprocessed_no_indices_pairs_train_tokenized")


# In[37]:


input_lang = load_cpickle_gc("iwslt-vi-eng/preprocessed_no_elmo_vilang")


# In[38]:


target_lang = load_cpickle_gc("iwslt-vi-eng/preprocessed_no_elmo_englang")


# In[8]:


val_idx_pairs =  pickle.load(open("iwslt-vi-eng/preprocessed_no_indices_pairs_validation_tokenized", 'rb'))


# In[9]:


train_idx_pairs =  pickle.load(open("iwslt-vi-eng/preprocessed_no_indices_pairs_train_tokenized", 'rb'))


# In[10]:


print(len(train_idx_pairs))


# In[11]:


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


# In[12]:


train_dataset = LanguagePairDataset(train_idx_pairs)
# is there anything in the train_idx_pairs that is only 0s right noww instea dof padding. 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=language_pair_dataset_collate_function,
                                          )


# Decoder

# In[13]:


class Decoder_RNN(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size).cuda()
        self.gru = nn.GRU(hidden_size, hidden_size).cuda()
        self.out = nn.Linear(hidden_size, output_size).cuda()
        self.softmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, input, hidden):
        embed = self.embedding(input).view(1, 1, -1)
        embed = F.relu(embed).cuda()
        output, hidden = self.gru(embed, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ### Encoder

# In[42]:


class SupEncoder(nn.Module):

    """
    A super class of Encoder that learns embeddings
    """
    def __init__(self, encoder, src_embed):
        super(SupEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed 
        
    def forward(self, src, src_mask):
        "Take in and process masked src sequences."
        a = self.encode(src, src_mask)
#         a = tf.math.reduce_mean(a, axis=1,keepdims=True)

        a = a[:,0,:]
        a = a.unsqueeze(1)
        a = a.cuda()
#         pdb.set_trace()
#         print(np.shape(a))
        return a
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    


# In[43]:


def clones(module, N):
    "Produce N identical layers."
#     module = module.cpu()
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        x = x.cuda()
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# In[44]:


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        features = features
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        x = x.cuda()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (self.a_2 * (x - mean) / (std + self.eps) + self.b_2).cuda()


# In[45]:


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x = x.cuda()
        return (x + self.dropout(sublayer(self.norm(x))))


# In[46]:


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = x.cuda()
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# In[47]:


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))              / math.sqrt(d_k)
    if mask is not None:
        mask = mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# In[48]:


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value =             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.cuda()
        x = x.transpose(1, 2).contiguous()              .view(nbatches, -1, self.h * self.d_k)
        
        return self.linears[-1](x)


# In[49]:


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff).cuda()
        self.w_2 = nn.Linear(d_ff, d_model).cuda()
        self.dropout = nn.Dropout(dropout).cuda()

    def forward(self, x):
        x = x.cuda()
        return self.w_2(self.dropout(F.relu(self.w_1(x)).cuda()))


# In[50]:


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model).cuda()
        self.d_model = d_model

    def forward(self, x):
        # x is the weights here
        x = x.cuda()
        return self.lut(x) * math.sqrt(self.d_model)


# In[51]:


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).cuda()
        position = torch.arange(0., max_len).unsqueeze(1).cuda()
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        div_term = div_term.cuda()
#         pdb.set_trace()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
#         pdb.set_trace()
        x = x.cuda()
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)

        return self.dropout(x)


# ### Training

# In[52]:


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, count):
#     encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    err_real = 0
    loss = 0
#     encoder_outputs = torch.zeros(max_length, decoder.hidden_size, device=device)

    # iterate GRU over words --> final hidden state is representation of source sentence. 
    for ei in range(input_length):
#         encoder_output = encoder(input_tensor[ei], encoder_hidden)
        encoder_output = encoder(input_tensor[ei],None)
#         encoder_outputs[ei] = encoder_output[0,0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_output
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(1, target_length-1):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
#             err_real += loss.data[0]
#             err_real += loss.item()
            count += 1
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(1, target_length-1):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
#             err_real += loss.data[0]
#             err_real += loss.item()
            count += 1
            if decoder_input.item() == EOS_token:
                break
                
    if type(loss) != torch.Tensor:
#         pdb.set_trace()
        loss = torch.Tensor(loss)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, count


# In[53]:


def trainIters(encoder, decoder,n_epochs, validation_pairs, pairs, lang1, lang2, max_length, max_length_generation, title, print_every=5000, plot_every=5000, learning_rate=3e-4, search="beam"):
    """
    lang1 is the Lang o|bject for language 1 
    Lang2 is the Lang object for language 2
    n_iters is the number of training pairs per epoch you want to train on
    """
    
    start = time.time()
    training_pairs = pairs
    n_iters = len(pairs)
    plot_losses, val_losses = [], []
    val_losses = [] 
    count, print_loss_total, plot_loss_total, val_loss_total, plot_val_loss = 0, 0, 0, 0, 0 
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss(ignore_index=PAD_token)
    plot_loss =[]
    val_loss = []
    
    for i in range(n_epochs):
        plot_loss =[]
        val_loss = []
        # framing it as a categorical loss function. 
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1] 
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            input_length = input_tensor.size(0)
            if target_tensor.size(0) < 3:
                continue
            loss_value, count = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, count)
            print_loss_total += loss_value 
            plot_loss_total += loss_value
            
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / count
                count = 0
                print_loss_total = 0
                print('TRAIN SCORE %s (%d %d%%) %.4f' % (timeSince(start, iter / n_epochs),
                                             iter, iter / n_epochs * 100, print_loss_avg))
                plot_loss.append(print_loss_avg)
                plot_loss_total = 0
                with torch.no_grad():
                    v_loss = test_model(encoder, decoder, search, validation_pairs, lang2, max_length=None)
                # returns bleu score
                print("VALIDATION BLEU SCORE: "+str(v_loss))
                val_loss.append(v_loss)
                save_model(encoder,decoder, title)
        plot_losses.append(plot_loss)
        val_losses.append(val_loss)
        save_model(encoder,decoder, title)
        make_graph(encoder, decoder, val_losses, plot_losses, title)

   
    


# In[54]:


def save_model(encoder, decoder, title):
    link = title.replace(" ", "")
    torch.save(encoder.state_dict(), "output/"+link + "encodermodel_states")
    torch.save(decoder.state_dict(), "output/"+link + "decodermodel_states")


# In[55]:


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


# In[56]:


hidden_size = 256
# number of duplicate layers in encoder
N = 1
# number of heads
h=8
dropout=0.1
"Helper: Construct a model from hyperparameters."
attn = MultiHeadedAttention(h, hidden_size).cuda()
ff = PositionwiseFeedForward(hidden_size,input_lang.n_words, dropout).cuda()
position = PositionalEncoding(hidden_size, dropout).cuda()
src_embed = nn.Sequential(Embeddings(hidden_size, input_lang.n_words), position).cuda()
encoder1 = SupEncoder(Encoder(EncoderLayer(hidden_size, attn, ff, dropout), N),src_embed).cuda()


# In[57]:


import sacrebleu
def calculate_bleu(predictions, labels):
    """
    Only pass a list of strings 
    """
    # tthis is ony with n_gram = 4

    bleu = sacrebleu.raw_corpus_bleu(predictions, [labels], .01).score
    return bleu


# In[58]:


def test_model(encoder, decoder, search, test_pairs, lang2, max_length=None):
    # for test, you only need the lang1 words to be tokenized,
    # lang2 words is the true labels
    encoder_inputs = [pair[0] for pair in test_pairs]
    true_labels = [pair[1] for pair in test_pairs]
    translated_predictions = []
    for i in range(len(encoder_inputs)):
        e_input = encoder_inputs[i]
        if max_length is None:
            max_length = len(e_input)
        decoded_words = generate_translation(encoder, decoder, e_input, max_length, lang2, search=search)
        translated_predictions.append(" ".join(decoded_words))
    rand = randint(0, 1)
    print(translated_predictions[rand])
    print(true_labels[rand])
    bleurg = calculate_bleu(translated_predictions, true_labels)
    return bleurg


# In[59]:


def generate_translation(encoder, decoder, sentence, max_length, target_lang, search="greedy", k = None):
    """ 
    @param max_length: the max # of words that the decoder can return
    @returns decoded_words: a list of words in target language
    """    
    with torch.no_grad():
        input_tensor = sentence
        input_length = sentence.size()[0]
        
        # encode the source sentence
        
        encoder_output = encoder(input_tensor.view(1, -1),torch.tensor([input_length]))
        # start decoding
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_output
        decoded_words = []
        
        if search == 'beam':
            if k == None:
                k = 5 # since k = 2 preforms badly
            decoded_words = beam_search(decoder, decoder_input, decoder_hidden, max_length, k, target_lang)  
        elif search == 'greedy':
            decoded_words = greedy_search(decoder, decoder_input, decoder_hidden, max_length)
        return decoded_words


# In[60]:


def beam_search(decoder, decoder_input, hidden, max_length, k, target_lang):
    
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
            if c_sequence[-1] == EOS_token:
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
    completed = sorted(completed, key= lambda x: x[1], reverse=True)[0] 
    #it's quite weird that it's not learning  what to do without the. . . 
    final_translation = []
    for x in completed[0]:
        final_translation.append(target_lang.index2word[x.squeeze().item()])
    return final_translation


# In[63]:


decoder1 = Decoder_RNN(target_lang.n_words,hidden_size).cuda()
args = {
    'n_epochs': 10,
    'learning_rate': 0.001,
    'search': 'beam',
    'encoder': encoder1,
    'decoder': decoder1,
    'lang1': input_lang, 
    'lang2': target_lang,
    "pairs":train_idx_pairs, 
    "validation_pairs": val_idx_pairs[:200], 
    "title": "Training Curve for Basic Self Encoder With LR = 0.0001",
    "max_length": 100,
    "max_length_generation": 20, 
    "plot_every": 500, 
    "print_every": 500
}

"""
We follow https://arxiv.org/pdf/1406.1078.pdf 
and use the Adadelta optimizer

"""
print(BATCH_SIZE)

trainIters(**args)

