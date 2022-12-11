from sklearn.model_selection import train_test_split
import torch
import copy
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler
from imblearn.under_sampling import RandomUnderSampler
import unicodedata
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from os import listdir
from os.path import isfile, join
import emoji
import itertools
import math
import scipy.stats
dev = "cuda:0"
device = torch.device(dev) if torch.cuda.is_available() else None

rus = RandomUnderSampler()

def check_early(vals_early, flag=8):
    for i in range(len(vals_early)):
        count = 0
        for j in range(i+1, min(len(vals_early), i+flag+1)):
            if vals_early[i] < vals_early[j]:
                count += 1
        if count == flag:
            return i
    return None

def get_remain_handles(path="/home/yaguang/new_nonstop_onefeaturesword1.csv", sep = "\x1b"):
    f = open(path)
    f.readline()
    handles = set()
    for line in f:
        line = line.strip()
        handle = line.split(sep)[0]
        handles.add(handle)
    return handles

class MyMLP(nn.Module):
    def __init__(self, D_in, H, D_out, bin_label=False):
        super(MyMLP, self).__init__()
        if H ==0:
            layers = [torch.nn.Linear(D_in, D_out)]
        else:
            layers = [torch.nn.Linear(D_in, H), torch.nn.ReLU(),  torch.nn.Linear(H, D_out)]
        if bin_label:
            layers.append(torch.nn.Sigmoid())
        self.classifier = torch.nn.Sequential(*layers)
    def forward(self, x):
        x = self.classifier(x)
        return x

class Attention(nn.Module):
    def __init__(self, D_in, H, D_out, hidden_size, bin_label, use_lstm=False, use_fc=True):
        super(Attention, self).__init__()
        layers = [
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    ]
        #print ("attention detail")
        #print ( D_in, H, D_out, hidden_size)
        if bin_label:
            layers.append(torch.nn.Sigmoid())
        #else:
         #   layers.append(torch.nn.Softmax(dim=1))
        self.use_lstm = use_lstm
        self.use_fc = use_fc
        self.classifier = torch.nn.Sequential(*layers)
        self.weight = nn.Parameter(torch.Tensor(2*hidden_size, 2*hidden_size))
        self.bias = nn.Parameter(torch.Tensor(1, 2*hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2*hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.bias.data.normal_(mean, std)

    def forward(self, numerical, embedding):
        #print (embedding.shape, embedding.permute(1, 0, 2).shape, self.weight.shape, self.bias.shape)
        # embedding shape is batch_size*seq_length*embedding_dim
        output = matrix_mul(embedding.permute(1, 0, 2), self.weight, self.bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, 1)
        weight = output
        output = element_wise_mul(embedding.permute(1, 0, 2), output.permute(1, 0)).squeeze(0)
        if self.use_fc:
            output = self.classifier(output)
        #print (output.shape)
        #print ("~~~~")
        #print (1/0)
        return output

class SeqAttention(nn.Module):
    def __init__(self, D_in, H, D_out, hidden_size, binary_classification, use_attention=True):
        super(SeqAttention, self).__init__()
        input_shape = D_in if use_attention else hidden_size
        media_shape = hidden_size if use_attention else int(hidden_size/2)

        layers = [
        torch.nn.Linear(input_shape, media_shape),
        torch.nn.ReLU(),
        torch.nn.Linear(media_shape, D_out),
    ]
        if binary_classification:
            layers.append(torch.nn.Sigmoid())
        self.lstm = nn.LSTM(D_in, hidden_size, bidirectional=True)
        self.gru = nn.GRU(D_in, hidden_size, bidirectional=True)
        self.classifier = torch.nn.Sequential(*layers)
        self.weight = nn.Parameter(torch.Tensor(2*hidden_size, 2*hidden_size))
        self.bias = nn.Parameter(torch.Tensor(1, 2*hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2*hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)
        self.use_attention = use_attention

    def _create_weights(self, mean=0.0, std=0.05):
        self.weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.bias.data.normal_(mean, std)

    def forward(self, numerical, embedding, seq_lens=None, use_fc=True):
        weight = None
        #print ("input shape", embedding.shape)
        input = pack_padded_sequence(embedding, seq_lens, enforce_sorted=False) if seq_lens!=None else embedding
        #packed_output, (final_hidden_state, final_cell_state) = self.lstm(input)
        packed_output, final_hidden_state = self.gru(input)
        if self.use_attention:
            if seq_lens!=None:
                f_output, input_sizes = pad_packed_sequence(packed_output)
                #print ("f_output", f_output.shape)
            else:
                f_output = packed_output
            output = matrix_mul(f_output, self.weight, self.bias)
            output = matrix_mul(output, self.context_weight).permute(1, 0)
            output = F.softmax(output, 1)
            #print ("med att", output.shape)
            weight = output
            #print (embedding.shape, output.shape, f_output.shape)
            output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
            #print ("end att", output.shape)
            #print ("~~~~")
            output1 = self.classifier(output) if use_fc else output
            return output1
        else:
            output = final_hidden_state[-1]
            if use_fc:
                output = self.classifier(output)
        return output


def matrix_mul_without_batch(feature, weight, bias=False):
    feature_weight = torch.mm(feature, weight)
    if isinstance(bias, torch.nn.parameter.Parameter):
        feature_weight = feature_weight + bias.expand(feature_weight.size()[0], bias.size()[1])
    return feature_weight

def matrix_mul(input, weight, bias=False):
    #input = input.permute(1, 0, 2)
    feature_list = []
    for feature in input:
        feature_weight = matrix_mul_without_batch(feature, weight, bias)
        feature_weight = torch.tanh(feature_weight).unsqueeze(0)
        feature_list.append(feature_weight)
    output = torch.cat(feature_list, 0)
    return torch.squeeze(output, len(output.shape)-1)

def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def under_sample(X, y):
    X, y = rus.fit_resample(X, y)
    return X, y

def to_cuda(d, device):
    if device:
        return d.to(device)
    return d

def to_self_cuda(d, device):
    return torch.tensor(d).to(device)

def to_float_cuda(d, device):
    return torch.tensor(d).float().to(device)

def map_imdb_handle_gt():
    ages = {}
    genders = {}
    f = open("/home/yaguang/imdb/age_ground_truth.csv")
    f.readline()
    for line in f:
        info = line.strip().split(",")
        name,age,gender,handle,verified = info
        handle = handle.lower()
        age = int(age)
        ages[handle] = age
        genders[handle] = gender
    f.close()
    return genders, ages

# wiki data from https://portals.mdi.georgetown.edu/public/demographic-inference
def map_handle_gt(filename="/home/yaguang/query_race_attributes.csv"):
    genders = {}
    ages = {}
    handle2year = get_handle2year("/home/yaguang/handle2year.csv")
    f = open(filename)
    for line in f:
        info = line.strip().split("\x1b")
        handle, gender, age, race = info[1].lower(), info[2].lower(), info[3].lower(), info[-1].lower()
        if gender:
            genders[handle] = gender
        if age:
            ages[handle] = 2022-int(age.split("-")[0])
            if handle in handle2year:
                ages[handle] = handle2year[handle]-int(age.split("-")[0])
    f.close()
    f = open("/home/yaguang/query_attributes.csv")
    for line in f:
        info = line.strip().split("\x1b")
        handle, gender, age, race = info[1].lower(), info[2].lower(), info[3].lower(), info[-1].lower()
        if age and (handle not in ages):
            ages[handle] = 2020-int(age.split("-")[0])
            if handle in handle2year:
                ages[handle] = handle2year[handle]-int(age.split("-")[0])
        if gender and handle not in genders:
            genders[handle] = gender
    f.close()

    return genders, ages

def get_handle2year(filename):
    handle2year = {}
    f = open(filename)
    for line in f:
        handle, year = line.strip().split(",")
        handle2year[handle] = int(year)
    f.close()
    return handle2year

def get_files_under_dir(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

