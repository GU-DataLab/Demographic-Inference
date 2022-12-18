import sys
sys.path.append("../")
import random
import time
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,  f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from os import listdir
from os.path import isfile, join
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from helper import SeqAttention, check_early
from helper import to_cuda, to_float_cuda, to_self_cuda, under_sample, map_handle_gt, get_files_under_dir
import os
import argparse, sys

parser=argparse.ArgumentParser()

parser.add_argument('--inference_type')
parser.add_argument('--bin_type')
parser.add_argument('--epochs')
parser.add_argument('--fix_seq_len')

args=parser.parse_args()
inference_type = args.inference_type
bin_type = args.bin_type
epochs = int(args.epochs)
fix_seq_len = int(args.fix_seq_len)

dev = "cuda"
device = torch.device(dev) if torch.cuda.is_available() else torch.device("cpu")

class DataProcessType:
    more_20_tweets = "more_20_tweets"
    all_tweets = "all_tweets"

class InferenceType:
    age = "age"
    gender = "gender"

# for age only
class BinType:
    two = "two"
    three = "three"
    four = "four"

def map_race_to_label(handle):
    if handle not in races:
        return -1
    val = 1 if races[handle] == "black" else 0
    return val

def map_gender_to_label(handle):
    if handle not in genders:
        return -1
    if genders[handle] == "female":
        val = 1
    elif genders[handle] == "male":
        val = 0
    else:
        val = -1
    return val

def map_num_to_label(num, category):
    for i in range(len(category)-1):
        if category[i]<=num<category[i+1]:
            return i
    return len(category)-1

def map_age_to_label(handle):
    if handle not in ages:
        return -1
    val = map_num_to_label(ages[handle], bins)
    return val

# create an index to associate user handle with a counter for 10-fold cross validation prupose
def map_index_to_file_label(onlyfiles):
    index_file = {}
    count = 0
    for filename in onlyfiles:
        handle = filename[:-4].lower()
        val = map_attribute(handle)
        if val == -1:
            continue
        index_file[count] = (filename, val)
        count += 1
    return index_file

def get_index_label(index_to_file_label):
    index = []
    labels = []
    for idx in index_to_file_label:
        index.append(idx)
        labels.append(index_to_file_label[idx][1])
    index, labels = np.array(index), np.array(labels)
    return index, labels

def train_model(model, train_names, test_names, val_names):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss() if inference_type == InferenceType.age else nn.BCELoss()
    vals_early = []
    for epoch in range(epochs):
        total = 0
        idx = 0
        counter = [0, 0, 0 ,0 ,0]
        #model.train()
        train_names = random.sample(train_names, len(train_names))
        while idx < len(train_names):
            model.train()
            batch_train_names = train_names[idx:idx+batch_size]
            X_batch_train = []
            y_batch_train = []
            seq_lens = []
            end_time = time.time()
            for i in range(len(batch_train_names)):
                handle = batch_train_names[i][:-4].lower()
                pick_emd = txn[handle]
                tweet_emb = pick_emd[:fix_seq_len]
                seq_lens.append(len(tweet_emb))
                while len(tweet_emb) < fix_seq_len:
                    tweet_emb.append([0 for i in range(input_dim)])
                X_batch_train.append(tweet_emb)
                val = map_attribute(handle)
                y_batch_train.append(val)
                counter[map_attribute(handle)] += 1
            X_batch_train = to_float_cuda(X_batch_train, device)
            X_batch_train = X_batch_train.permute(1, 0, 2)
            y_batch_train = to_float_cuda(y_batch_train, device).reshape(-1, 1) if inference_type != InferenceType.age else to_self_cuda(y_batch_train, device)
            y_pred = model(None, X_batch_train, seq_lens)
            loss = loss_fn(y_pred, y_batch_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            idx += batch_size
            total += loss.item()
        f1, loss_val = eval_model(model, val_names)
        vals_early.append(loss_val)
        f1, loss_val = eval_model(model, test_names)
        print("epoch "+str(epoch))
        print ("test f1 is "+str(f1))
        #torch.save(model, "model"+"/"+str(total))
        print ("-------------")
        print ()
        if check_early(vals_early):
            break
    return loss.item()

def eval_model(model, test_names):
    test_names = random.sample(test_names, len(test_names))
    idx = 0
    y_hat_test_class = []
    y_test = []
    loss_fn = nn.CrossEntropyLoss() if inference_type == InferenceType.age else nn.BCELoss()
    model.eval()
    texts = []
    with torch.no_grad():
        total = 0
        while idx < len(test_names):
            y_eval_test = []
            batch_test_names = test_names[idx:idx+batch_size]
            X_batch_test = []
            seq_lens = []
            for i in range(len(batch_test_names)):
                handle = batch_test_names[i][:-4].lower()
                pick_emd = txn[handle]
                tweet_emb = pick_emd[:fix_seq_len]
                seq_lens.append(len(tweet_emb))
                while len(tweet_emb) < fix_seq_len:
                    tweet_emb.append([0 for i in range(input_dim)])
                X_batch_test.append(tweet_emb)
                val = map_attribute(handle)
                y_test.append(val)
                y_eval_test.append(map_attribute(handle))
            X_batch_test = to_float_cuda(X_batch_test, device)
            X_batch_test = X_batch_test.permute(1, 0, 2)
            #y_hat_test, weights = model(None, X_batch_test, seq_lens)
            y_hat_test = model(None, X_batch_test, seq_lens)
            #print (weights.shape)
            y_eval_test = to_float_cuda(y_eval_test, device).reshape(-1, 1) if inference_type != InferenceType.age else to_self_cuda(y_eval_test, device)
            loss = loss_fn(y_hat_test, y_eval_test)
            #print ("eval loss is "+str(loss.item()))
            total += loss.item()
            if inference_type == InferenceType.age:
                target = np.argmax(y_hat_test.cpu().detach().numpy(), axis=1)
                for i in range(len(target)):
                    y_hat_test_class.append(target[i])
            else:
                target = np.where(y_hat_test.cpu().detach().numpy()<0.5, 0, 1)
                for i in range(len(target)):
                    y_hat_test_class.append(target[i][0])
            idx += batch_size
    f1 = f1_score(y_test, y_hat_test_class, average='macro')
    return f1, total

# load embeddings
def load_data(foldername):
    txn = {}
    filenames = get_files_under_dir(foldername)
    for filename in filenames:
        f = open(foldername+filename)
        emds = []
        handle = filename.split(".")[0]
        for line in f:
            info = line.strip().split(",")
            emds.append([float(val) for val in info[1:]])
        txn[handle] = emds
        f.close()
    return txn

txn = load_data("wiki_embeddings/")

if inference_type == InferenceType.gender:
    map_attribute = map_gender_to_label
elif inference_type == InferenceType.age:
    map_attribute = map_age_to_label
else:
    print ("------------ wrong type ------------")

if bin_type == BinType.two:
    bins = [0, 45]
elif bin_type == BinType.three:
    bins = [0, 35, 55]
elif bin_type == BinType.four:
    bins = [0, 30, 40, 50]
else:
    print ("------------ wrong type ------------")

D_out = len(bins) if inference_type == InferenceType.age else 1
bin_classification = True if inference_type == InferenceType.gender else False

genders, ages = map_handle_gt()
# load handles from wiki data
onlyfiles = [handle+".csv" for handle in genders]
input_dim = 512
batch_size = 32

index_to_file_label = map_index_to_file_label(onlyfiles)
counter = [0, 0, 0, 0]
index, labels = get_index_label(index_to_file_label)
for label in labels:
    counter[label] += 1

usecols = list(np.arange(1,769))
counter = [0, 0, 0, 0]
# undersample data
index, labels = under_sample(index.reshape(-1, 1), labels)
index = index.ravel()
for label in labels:
    counter[label] += 1


skf = StratifiedKFold(n_splits=5, shuffle=True)
learning_rate = 0.0001

fold = 0
for train_index, test_index in skf.split(index, labels):
    X_train_index, X_test_index = index[train_index], index[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    X_train_index, X_val_index, y_train, y_val = train_test_split(X_train_index, y_train, test_size=0.2, stratify=y_train, shuffle=True)
    y_train = to_float_cuda(y_train, device)
    y_test = to_float_cuda(y_test, device)
    y_val = to_float_cuda(y_val, device)
    X_train_names = []
    X_test_names = []
    X_val_names = []
    for idx in X_train_index:
        X_train_names.append(index_to_file_label[idx][0])
    for idx in X_test_index:
        X_test_names.append(index_to_file_label[idx][0])
    for idx in X_val_index:
        X_val_names.append(index_to_file_label[idx][0])
    model = SeqAttention(input_dim, int(input_dim/2/2), D_out, int(input_dim/2), bin_classification, True)
    to_cuda(model, device)
    train_model(model, X_train_names, X_test_names, X_val_names)
    fold += 1
    break
