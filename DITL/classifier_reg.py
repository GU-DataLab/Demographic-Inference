import random
import time
import pickle
import torch
import collections
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
from helper import map_handle_gt
from helper import to_cuda, to_float_cuda, to_self_cuda, under_sample, get_files_under_dir
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


class InferenceType:
    age = "age"
    gender = "gender"

# for age only
class BinType:
    two = "two"
    three = "three"
    four = "four"

def map_gender_to_label(handle, genders):
    if handle not in genders:
        return -1
    if genders[handle] in ["Female", "female"]:
        val = 1
    elif genders[handle] in ["Male", "male"]:
        val = 0
    else:
        val = -1
    return val

def map_num_to_label(num, category):
    for i in range(len(category)-1):
        if category[i]<=num<category[i+1]:
            return i
    return len(category)-1

def map_age_to_label(handle, ages):
    if handle not in ages:
        return -1
    val = map_num_to_label(ages[handle], bins)
    return val

# create an index to associate user handle with a counter for 10-fold cross validation prupose
def map_index_to_file_label(onlyfiles, att):
    index_file = {}
    count = 0
    for filename in onlyfiles:
        handle = filename[:-4].lower()
        val = map_attribute(handle, att)
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

# get training data and label information
def get_train_label(batch_train_names, txn, att):
    X_batch_train = []
    y_batch_train = []
    y_gt = []
    seq_lens = []
    counter = [0, 0, 0 ,0]
    for i in range(len(batch_train_names)):
        handle = batch_train_names[i][:-4].lower()
        tweet_emb = pick_emd = txn[handle]
        seq_lens.append(len(tweet_emb))
        while len(tweet_emb) < fix_seq_len:
            tweet_emb.append([0 for i in range(input_dim)])
        X_batch_train.append(tweet_emb)
        val = map_attribute(handle, att)
        y_batch_train.append(val)
        y_gt.append(val)
        counter[map_attribute(handle, att)] += 1
    #print (counter)
    X_batch_train = to_float_cuda(X_batch_train, device)
    X_batch_train = X_batch_train.permute(1, 0, 2)
    y_batch_train = to_float_cuda(y_batch_train, device).reshape(-1, 1) if inference_type != InferenceType.age else to_self_cuda(y_batch_train, device)
    return X_batch_train, y_batch_train, y_gt, seq_lens

def tensor2list(vals):
    y_hat_test_class = []
    if inference_type == InferenceType.age:
        target = np.argmax(vals.cpu().detach().numpy(), axis=1)
        for i in range(len(target)):
            y_hat_test_class.append(target[i])
    else:
        target = np.where(vals.cpu().detach().numpy()<0.5, 0, 1)
        for i in range(len(target)):
            y_hat_test_class.append(target[i][0])
    return y_hat_test_class

def train_model(model, imdb_train_names, imdb_test_names, imdb_val_names, wiki_train_names, wiki_test_names, wiki_val_names):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss() if inference_type == InferenceType.age else nn.BCELoss()
    vals_early = []
    for epoch in range(epochs):
        y_hat_test = []
        y_test = []
        total = 0
        idx = 0
        counter = [0, 0, 0 ,0 ,0]
        imdb_train_names = random.sample(imdb_train_names, len(imdb_train_names))
        wiki_train_names = random.sample(wiki_train_names, len(wiki_train_names))
        while idx < len(imdb_train_names):
            model.train()

            imdb_batch_train_names = imdb_train_names[idx:idx+batch_size]
            imdb_X_batch_train, imdb_y_batch_train, imdb_y_gt, imdb_seq_lens = get_train_label(imdb_batch_train_names, txn_imdb, gt_imdb)

            wiki_batch_train_names = wiki_train_names[idx:idx+batch_size]
            wiki_X_batch_train, wiki_y_batch_train, wiki_y_gt, wiki_seq_lens = get_train_label(wiki_batch_train_names, txn_wiki, gt_wiki)

            y_pred = model(None, imdb_X_batch_train, imdb_seq_lens)
            imdb_loss = loss_fn(y_pred, imdb_y_batch_train)
            y_hat_test += tensor2list(y_pred)
            y_test += imdb_y_gt

            y_pred = model(None, wiki_X_batch_train, wiki_seq_lens)
            wiki_loss = loss_fn(y_pred, wiki_y_batch_train)

            loss = sum([imdb_loss, lam*wiki_loss])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            idx += batch_size
            total += loss.item()
        print ("total train loss is "+str(total))
        f1 = f1_score(y_test, y_hat_test, average='macro')
        print ("train f1 is "+str(f1))
        f1, loss_val = eval_model(model, imdb_val_names, wiki_val_names, True)
        vals_early.append(loss_val)
        f1, loss_val = eval_model(model, imdb_test_names, wiki_test_names)
        print ("-------------")
        print ()
        # early stopping
        if check_early(vals_early) is not None:
            break
    return loss.item()

def eval_model(model, imdb_test_names, wiki_test_names, is_eval=False):
    imdb_test_names = random.sample(imdb_test_names, len(imdb_test_names))
    wiki_test_names = random.sample(wiki_test_names, len(wiki_test_names))
    idx = 0
    y_hat_test_class = []
    y_test = []
    loss_fn = nn.CrossEntropyLoss() if inference_type == InferenceType.age else nn.BCELoss()
    model.eval()
    test_handles = []
    texts = []
    with torch.no_grad():
        total = 0
        while idx < len(imdb_test_names):
            imdb_batch_test_names = imdb_test_names[idx:idx+batch_size]
            imdb_X_batch_test, imdb_y_eval_test, imdb_y_gt, imdb_seq_lens = get_train_label(imdb_batch_test_names, txn_imdb, gt_imdb)
            y_test += imdb_y_gt
            imdb_y_hat_test = model(None, imdb_X_batch_test, imdb_seq_lens)
            imdb_loss = loss_fn(imdb_y_hat_test, imdb_y_eval_test)

            if is_eval:
                wiki_batch_test_names = wiki_test_names[idx:idx+batch_size]
                wiki_X_batch_test, wiki_y_eval_test, wiki_y_gt, wiki_seq_lens = get_train_label(wiki_batch_test_names, txn_wiki, gt_wiki)
                wiki_y_hat_test = model(None, wiki_X_batch_test, wiki_seq_lens)
                wiki_loss = loss_fn(wiki_y_hat_test, wiki_y_eval_test)
                loss = sum([imdb_loss, lam*wiki_loss])
            else:
                loss = imdb_loss
            total += loss.item()
            y_hat_test_class += tensor2list(imdb_y_hat_test)
            idx += batch_size
    f1 = f1_score(y_test, y_hat_test_class, average='macro')
    if not is_eval:
        print ("total test loss is "+str(total))
        print ("test f1 is "+str(f1))
        print (",".join([str(val) for val in y_test]))
        print (",".join([str(val) for val in y_hat_test_class]))
    else:
        print ("total eval loss is "+str(total))
        print ("eval f1 is "+str(f1))
    #auc = roc_auc_score(y_test, y_hat_test_class )
    return f1, total

# imdb ground truth
def imdb_map_handle_gt():
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

def get_filenames(path):
    filenames = []
    f = open(path)
    for line in f:
        handle = line.strip().split(",")[0]
        filenames.append(handle+".csv")
    return filenames

# get index for cross validation
def get_index_labels(handles, att):
    onlyfiles = [handle+".csv" for handle in handles]
    index_to_file_label = map_index_to_file_label(onlyfiles, att)
    counter = [0, 0, 0, 0]
    index, labels = get_index_label(index_to_file_label)
    for label in labels:
        counter[label] += 1
    counter = [0, 0, 0, 0]
    index, labels = under_sample(index.reshape(-1, 1), labels)
    index = index.ravel()
    for label in labels:
        counter[label] += 1
    
    return index, labels, index_to_file_label

# split files into train, val and test
def get_train_test_val_names(index, labels, index_to_file_label, train_index, test_index):
    X_train_index, X_test_index = index[train_index], index[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    X_train_index, X_val_index, y_train, y_val = train_test_split(X_train_index, y_train, test_size=0.2, stratify=y_train, shuffle=True,random_state=0)
    X_train_names = []
    X_test_names = []
    X_val_names = []
    for idx in X_train_index:
        X_train_names.append(index_to_file_label[idx][0])
    for idx in X_test_index:
        X_test_names.append(index_to_file_label[idx][0])
    for idx in X_val_index:
        X_val_names.append(index_to_file_label[idx][0])
    return X_train_names, X_test_names, X_val_names

# load embeddings
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

txn_wiki = load_data("embeddings/")
txn_imdb = load_data("imdb_embeddings/")

if inference_type == InferenceType.gender:
    map_attribute = map_gender_to_label
elif inference_type == InferenceType.age:
    map_attribute = map_age_to_label
else:
    print ("error")

if bin_type == BinType.two:
    bins = [0, 45]
elif bin_type == BinType.three:
    bins = [0, 35, 55]
elif bin_type == BinType.four:
    bins = [0, 30, 40, 50]
else:
    print (1/0)

D_out = len(bins) if inference_type == InferenceType.age else 1
bin_classification = True if inference_type == InferenceType.gender else False
input_dim = 512
batch_size = 32

# regularization term
lam = 0.1

# imdb
genders_imdb, ages_imdb = map_handle_gt("imdb_gt.csv")
genders_wiki, ages_wiki = map_handle_gt("gt.csv")

# define the demographic to infer and the sample data for regularization
if inference_type == InferenceType.age:
    gt_imdb = ages_imdb
    gt_wiki = ages_wiki
    if bin_type == BinType.two:
        pretrained = ""
        tl_file = ""
    elif bin_type == BinType.three:
        pretrained = ""
        tl_file = ""
    elif bin_type == BinType.four:
        pretrained = ""
        tl_file = ""
else:
    gt_imdb = genders_imdb
    gt_wiki = genders_wiki
    pretrained = "model/model"
    tl_file = "gender_tl_handles.csv"


index_imdb, labels_imdb, index_to_file_label_imdb = get_index_labels(list(genders_imdb.keys()), gt_imdb)
index_wiki, labels_wiki, index_to_file_label_wiki = get_index_labels(list(genders_wiki.keys()), gt_wiki)

skf = StratifiedKFold(n_splits=10, shuffle=True)
learning_rate = 0.0001


skf_imdb = skf.split(index_imdb, labels_imdb)
skf_wiki = skf.split(index_wiki, labels_wiki)
fold = 0
while fold < 10:
    train_index, test_index = next(skf_imdb)
    imdb_train_names, imdb_test_names, imdb_val_names = get_train_test_val_names(index_imdb, labels_imdb, index_to_file_label_imdb, train_index, test_index)
    train_index, test_index = next(skf_wiki)
    wiki_train_names, wiki_test_names, wiki_val_names = get_train_test_val_names(index_wiki, labels_wiki, index_to_file_label_wiki, train_index, test_index)
    model = torch.load(pretrained, device)
    to_cuda(model, device)
    train_model(model, imdb_train_names, imdb_test_names, imdb_val_names, wiki_train_names, wiki_test_names, wiki_val_names)
    fold += 1
    break
