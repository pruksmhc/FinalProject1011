import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data_prep import tensorFromSentence
import numpy as np
import pickle
from random import randint

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def evaluate(encoder, decoder, sentence,max_length,  max_length_generation, search="greedy"):
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
        input_tensor = sentence # this is already tokenized to a pair so it doens't 
        # take as long to run. 
        input_length = input_tensor.size()[0]
        # encode the source lanugage
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        # decode the context vector
        decoder_hidden = encoder_hidden # decoder starts from the last encoding sentence
        # output of this function
        decoder_attentions = torch.zeros(max_length, max_length)
        
        if search == 'greedy':
            decoded_words = greedy_search(decoder, decoder_input, decoder_hidden, max_length_generation)
        elif search == 'beam':
            decoded_words = beam_search(decoder, decoder_input, decoder_hidden, max_length_generation)  
        return decoded_words


import sacrebleu
def calculate_bleu(predictions, labels):
    """
    Only pass a list of strings 
    """
    # tthis is ony with n_gram = 4

    bleu = sacrebleu.raw_corpus_bleu(predictions, [labels], .01).score
    return bleu

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
    rand = randint(0, 100)
    print(translated_predictions[rand])
    print(true_labels[rand])
    bleurg = calculate_bleu(translated_predictions, true_labels)
    return bleurg