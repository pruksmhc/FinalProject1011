
# now, we tokenize our current dataset
import pickle
import pdb 
import numpy as np
import pandas as pd
import pprint
import numpy as np
from operator import itemgetter
import numpy as np
import torch
from torch.utils.data import Dataset
import _pickle as cPickle
import gc
from sklearn.preprocessing import OneHotEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SENTENCE_LENGTH_FIRST = 50 
MAX_SENTENCE_LENGTH_SECOND = 28 
EMBED_DIM = 300
SOS_token = 2
EOS_token = 3

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

class TranslationDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, lang1, lang2):
        """
        @param data_list: list of the preprocessed tokens, so fro
        Inspired by https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649
        Pass in lang1 and lang2 
        """

        self.datasets = self.tensorsFromPairs(data_list, lang1, lang2)
        lengths_source = [len(x) for x in self.datasets[0]]
        self.max_sourcelength = max(lengths_source)
        lengths_target = [len(x) for x in self.datasets[1]]
        self.max_targetlength = max(lengths_target)

    def __len__(self):
        return len(self.datasets[0])

    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return indexes

    def tensorsFromPairs(self, pairs, lang1, lang2):
        # at this point, dont' hav eto make it a Tensor yet. 
        source_tensors = []
        target_tensors = []
        for pair in pairs:
            source_tensor = self.tensorFromSentence(lang1, pair[0])
            target_tensor = self.tensorFromSentence(lang2, pair[1])
            source_tensors.append(source_tensor)
            target_tensors.append(target_tensor)
        # outputs the batc
        return [source_tensors, target_tensors]

    def __getitem__(self, key):
        sentences = tuple(d[key] for d in self.datasets)
        lengths = tuple(len(d[key]) for d in self.datasets)
        return [sentences, lengths, self.max_sourcelength, self.max_targetlength]

def get_order(sorted_list, to_construct):
    order = []
    for elt in to_construct:
        index = []
        for i in range(len(sorted_list)):
            s_elt = sorted_list[i]
            if s_elt == elt:
                index = i
        order.append(index)
    return order

def get_index(query_list, in_list):
    # this get sindex by hecking if the sum of non 0 is equal. 
    order = []
    for i in range(len(in_list)):
        elt = in_list[i]
        if np.array(query_list).sum() == np.array(elt).sum():
            return i
    return -1

def translation_collate_func_concat(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    Here, similar to teh way we used index_select to re-arrange the two sentence sin hw2, 
    we want to rearrange the encoder outputs to match the traget, so rearragne bseed on teh 
    target's indices hwer the target is the "true" order. 
    """
    first_data_list = []
    second_data_list = []
    length_list_first = []
    length_list_second = []
    data_list_first = []
    data_list_second = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    #pdb.set_trace()
    for datum in batch:
        first_data_list.append(datum[0][0])
        second_data_list.append(datum[0][1])
        length_list_first.append(datum[1][0])
        length_list_second.append(datum[1][1])
    sorted_first = sorted(first_data_list, key=lambda e: len(e), reverse=True)
    # this is the sorted data list. 
    sorted_second = sorted(second_data_list, key=lambda e: len(e), reverse=True)
    helper_sorted = [x for _,x in sorted(zip(length_list_second, first_data_list),  key=lambda e: e[0], reverse=True )]
    # so now this should be the "true" ones, that you want to reconstruct
    # which is acutally kin the order of  the sorted_second
    order_target_for_source=  get_order(sorted_first, helper_sorted)

    length_first =  sorted(length_list_first, reverse=True)
    length_second = sorted(length_list_second, reverse=True)

    # Assert tthat the indexing is the same 
    # lopo through the original array, and find the indices of the first in the sorted_first and seocnd in 
    # sorted second, and then the order_target_for_source
    #pdb.set_trace()
    for i in range(len(batch)):
        # padding
        first_sentence = sorted_first[i]
        second_sentence = sorted_second[i]
        first_sentence.extend([0]*(batch[0][2]- len(first_sentence)))
        second_sentence.extend([0]*(batch[0][3]-len(second_sentence)))
        data_list_first.append(first_sentence)
        data_list_second.append(second_sentence)
    # checked that the index_selected in line with the sorted_second
    check_list = torch.index_select(torch.LongTensor(data_list_first), 0, torch.LongTensor(order_target_for_source))
    for i in range(len(check_list)):
        target_qn = sorted_second[i]
        source_qn = check_list[i]
        index_target = get_index(target_qn, second_data_list)
        index_source = get_index(source_qn, first_data_list)
        assert index_target == index_source

    return [torch.LongTensor(data_list_first), torch.LongTensor(data_list_second), torch.LongTensor(length_first),  torch.LongTensor(length_second), torch.LongTensor( order_target_for_source)]

