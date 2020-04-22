# -*- coding: utf-8 -*-
import pdb
import uuid
import numpy as np
import datetime
import os
# import tqdm
import sys
import collections
import matplotlib.image as img

from PIL import Image
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader,Subset
from torch.utils.data import Sampler,SubsetRandomSampler
import torch.nn.init as init
from models.resnet import i3_res50_nl,i3_res50_nl_new,i3_res50_nl_new_test,i3_res50_nl_new_test_1block,Bottleneck_test,NonLocalBlock_test_Conv1
from torchvision import models, transforms
import time
# from train_singlenet_phase_addnonlocal_AL import SeqSampler,resnet_lstm_nonlocal
#for diversity
# import pandas as pd
# from LocalitySensitiveHashing import *
# from pandas import read_csv
# from sklearn.decomposition import PCA
from PIL import Image
import csv
import numpy as np
# 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_num = 7
crop_type = 1
sequence_length = 10
def load_select_data(filename):
    import json
    with open(filename) as f:
        data = json.load(f)
        assert list(data.keys()) == ['selected', 'unselected', 'mask']
    return data['selected'],data['unselected'],data['mask']

def save_select_data(save_select_txt_path,selected,unselected,mask,time_cur):
    import json
    dictObj = {'selected':selected,'unselected':unselected,'mask':mask}
    jsObj = json.dumps(dictObj)
    save_name = os.path.join(save_select_txt_path,str(len(selected))+ '_' + str(time_cur) + '.json')
    fileObject = open(save_name, 'w')
    fileObject.write(jsObj)
    fileObject.close()

def save_indices(time_cur,indices,epoch,save_select_txt_path):
    import json
    dictObj = {'indices':indices,'round':epoch}
    jsObj = json.dumps(dictObj)
    save_name = os.path.join(save_select_txt_path,str(time_cur) + str(len(indices))+ '_' + str(epoch) + '.json')
    print (save_name)
    if os.path.exists(save_select_txt_path) is False:
        os.makedirs(save_select_txt_path)
    fileObject = open(save_name, 'w')
    fileObject.write(jsObj)
    fileObject.close()

def save_select_data_bylabel(save_select_txt_path,selected,unselected,mask,label):
    import json
    dictObj = {'selected':selected,'unselected':unselected,'mask':mask}
    jsObj = json.dumps(dictObj)
    save_name = os.path.join(save_select_txt_path,str(len(selected))+ '_' + str(label) + '_' + currtime + '.json')
    fileObject = open(save_name, 'w')
    fileObject.write(jsObj)
    fileObject.close()
    
def random_select_data(X,N,selected,unselected,mask):
    '''
    parameter
    X:input sorted index list[0,1,2,3,4,5,6]
    N: how many samples to select
    selected:list that are aready selected in X
    unselected:list that are not selected in X
    mask:list using 0/1 to represent each position is selected/unavaliable(0) or unselected/avaliable(1)
    write by @michelle
    '''
    # pdb.set_trace()
    new_samples = random.sample(unselected,N)
    for i in range(len(X)):
        if X[i] in new_samples:
            mask[i] = 0
    selected.extend(new_samples)
    selected.sort()
    # unselected = [j for j in X if j not in selected]
    unselected = [X[i] for i in range(len(X)) if X[i] not in selected]
    print ("selected:",selected)
    print ("unselected:",unselected)
    print ("mask:",mask)
    return selected,unselected,mask

class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

class resnet_lstm_dropout(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm_dropout, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True,dropout=0.2)
        self.fcDropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, class_num)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.fcDropout(y)
        y = self.fc(y)
        return y
        
class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        # self.fcDropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, class_num)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        # y = self.fcDropout(y)
        y = self.fc(y)
        return y

class resnet_lstm_feature(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm_feature, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        # self.fcDropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, class_num)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = x.view(-1, sequence_length, 2048)
        # feature = x.clone().detach()
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        feature = y.clone().detach()
        # y = self.fcDropout(y)
        y = self.fc(y)
        return y,feature

class resnet_lstm_nonlocal(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm_nonlocal, self).__init__()
        resnet_lstm_base = resnet_lstm()
        chkPath = '../AL_Res_LSTM/results/1568095370.2707942/checkpoint_best-25.pt'
        print("Restoring: ",chkPath)
        # Load
        state = torch.load(chkPath)
        resnet_lstm_base.load_state_dict(state['state_dict'])  
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet_lstm_base.share.conv1)
        self.share.add_module("bn1", resnet_lstm_base.share.bn1)
        self.share.add_module("relu", resnet_lstm_base.share.relu)
        self.share.add_module("maxpool", resnet_lstm_base.share.maxpool)
        self.share.add_module("layer1", resnet_lstm_base.share.layer1)
        self.share.add_module("layer2", resnet_lstm_base.share.layer2)
        self.share.add_module("layer3", resnet_lstm_base.share.layer3)
        self.share.add_module("layer4", resnet_lstm_base.share.layer4)
        self.share.add_module("avgpool", resnet_lstm_base.share.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        # self.share.add_module("lstm", resnet_lstm_base.lstm)
        # planes = 256
        # block.expansion = 4
        # inplanes = planes * block.expansion
        
        outplanes = 512
        self.nl = NonLocalBlock_test_Conv1(outplanes, outplanes, outplanes//2)
        # self.nonlocal = Bottleneck_test(inplanes, planes, 1, None, temp_conv[i], temp_stride[i], use_nl=True)
        # self.nonlocal = Bottleneck_test()
        for m in self.nl.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.fcDropout = nn.Dropout(0.5)
        # self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, class_num)
        # init.xavier_normal_(self.lstm.all_weights[0][0])
        # init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        # y = y.contiguous().view(-1, 512) # b x T x C
        y = y.contiguous().view(-1, sequence_length, 512) # b x T x C
        y_p = y.permute(0, 2, 1) # b x T x C -> b x C x T (b x 512 x 10)
        y_p = self.nl(y_p)
        # out = self.relu(out)
        y_p = self.fcDropout(y_p)
        y_p_p = y_p.permute(0, 2, 1)#(b x 512 x 10) - > b x 10 x 512
        y_p_p = y_p_p.contiguous().view(-1,y_p_p.shape[-1]) # 160 x 512
        y_p_p = self.fc(y_p_p)
        return y_p_p

    def nonlocal_features(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        # y = y.contiguous().view(-1, 512) # b x T x C
        y = y.contiguous().view(-1, sequence_length, 512) # b x T x C
        y_p = y.permute(0, 2, 1) # b x T x C -> b x C x T (b x 512 x 10)
        # y_p = self.nl(y_p)
        # # out = self.relu(out)
        # y_p = self.fcDropout(y_p)
        # y_p_p = y_p.permute(0, 2, 1)#(b x 512 x 10) - > b x 10 x 512
        # y_p_p = y_p_p.contiguous().view(-1,y_p_p.shape[-1]) # 160 x 512
        # return y_p_p
        return y_p

class resnet_lstm_dropout(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm_dropout, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True,dropout=0.2)
        self.fcDropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, class_num)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.fcDropout(y)
        y = self.fc(y)
        return y

def val_for_selection(model_path, dataset, sequence_length, unselected):
    num_test = len(unselected)
    # test_useful_start_idx = get_useful_start_idx(sequence_length, test_num_each)

    # num_test_we_use = len(test_useful_start_idx)

    # test_we_use_start_idx = test_useful_start_idx[0:num_test_we_use]

    # test_idx = []
    # for i in range(num_test_we_use):
    #     for j in range(sequence_length):
    #         test_idx.append(test_we_use_start_idx[i] + j)

    # num_test_all = len(test_idx)

    # print('num test start idx : {:6d}'.format(len(test_useful_start_idx)))
    # print('last idx test start: {:6d}'.format(test_useful_start_idx[-1]))
    # print('num of test dataset: {:6d}'.format(num_test))
    # print('num of test we use : {:6d}'.format(num_test_we_use))
    # print('num of all test use: {:6d}'.format(num_test_all))
    # test_loader = DataLoader(train_dataset,
    #                          batch_size=test_batch_size,
    #                          sampler=SeqSampler(test_dataset, test_idx),
    #                          num_workers=workers)
    pred_val_dicts = {}
    test_batch_size = sequence_length
    workers = 2
    sequence_length = 10
    test_idx = []
    for i in range(len(unselected)):
        for j in range(sequence_length):
            test_idx.append(unselected[i] + j)
    num_test_all = len(test_idx)

    # test_loader = DataLoader(
    #     dataset,
    #     batch_size=test_batch_size,
    #     sampler=SeqSampler(dataset, test_idx),
    #     num_workers=workers,
    #     pin_memory=True
    # )
    test_loader = DataLoader(
        dataset,
        batch_size=test_batch_size,
        sampler=SeqSampler(dataset,test_idx),
        num_workers=workers,
        pin_memory=True
    )

    # model = i3_res50_nl_new_test(400)
    # model = i3_res50_nl_new_test_1block(400)
    model = resnet_lstm_nonlocal()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, class_num) 
    # print(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!") 
    #consider multi gpu formatted at module.
    state = torch.load(model_path)    
    newdict = {}    
    for k,v in state['state_dict'].items():
        if k[0:7] == 'module.': 
            name = k[7:]
            newdict[name] = v
        else:
            newdict[k] = v
    model.load_state_dict(newdict) 
    model = DataParallel(model)
    model.to(device)

    # criterion = nn.CrossEntropyLoss(size_average=False)

    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_start_time = time.time()

    all_preds = []
    pth_blobs = {}
    # f = open('./possibility.txt', 'a')
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # torch.cuda.empty_cache()            
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels[(sequence_length - 1)::sequence_length]

            if crop_type == 0 or crop_type == 1:
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs = model.module.nonlocal_features(inputs)
                relation_matrix = model.module.nl.softmax_results(outputs)
            # outputs = outputs[sequence_length - 1::sequence_length]
            topN = 5
            clip_relation_score = torch.topk(relation_matrix.view(-1),topN)[0].mean()
            pred_val_dicts[str(unselected[i])] = clip_relation_score
            select_oneclip_time = time.time() - start_time
            print("select:%d, select_oneclip_time:%.2f" % (i,select_oneclip_time))
            pdb.set_trace()

    return pred_val_dicts

def val_for_CNNembselection(model_path, dataset, sequence_length, unselected,sim_metric):
    num_test = len(unselected)
    pred_val_dicts = {}
    test_batch_size = sequence_length
    workers = 2
    sequence_length = 10
    test_idx = []
    for i in range(len(unselected)):
        for j in range(sequence_length):
            test_idx.append(unselected[i] + j)
    num_test_all = len(test_idx)

    test_loader = DataLoader(
        dataset,
        batch_size=test_batch_size,
        sampler=SeqSampler(dataset,test_idx),
        num_workers=workers,
        pin_memory=True
    )
    model = resnet_lstm_feature()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!") 
    #consider multi gpu formatted at module.
    state = torch.load(model_path)    
    newdict = {}    
    for k,v in state['state_dict'].items():
        if k[0:7] == 'module.': 
            name = k[7:]
            newdict[name] = v
        else:
            newdict[k] = v
    model.load_state_dict(newdict) 
    model = DataParallel(model)
    model.to(device)

    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_start_time = time.time()

    all_preds = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):           
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels[(sequence_length - 1)::sequence_length]

            if crop_type == 0 or crop_type == 1:
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs,cnn_emb_fea = model.module.forward(inputs)
            if sim_metric == "dot": # dot product as a similarity metric.
                # relation_matrix = torch.bmm(cnn_emb_fea.view(-1, sequence_length, 2048), cnn_emb_fea.view(-1, sequence_length, 2048).permute(0, 2, 1))
                relation_matrix = torch.bmm(cnn_emb_fea.view(-1, sequence_length, cnn_emb_fea.shape[-1]), cnn_emb_fea.view(-1, sequence_length, cnn_emb_fea.shape[-1]).permute(0, 2, 1))
            topN = 5
            clip_relation_score = torch.topk(relation_matrix.view(-1),topN)[0].mean()
            pred_val_dicts[str(unselected[i])] = clip_relation_score
            print("select:",i)

    return pred_val_dicts

def val_for_DBNselection(model_path, dataset, sequence_length, unselected):
    num_test = len(unselected)
    pred_val_dicts = {}
    test_batch_size = sequence_length
    workers = 2
    sequence_length = 10
    test_idx = []
    for i in range(len(unselected)):
        for j in range(sequence_length):
            test_idx.append(unselected[i] + j)
    num_test_all = len(test_idx)

    test_loader = DataLoader(
        dataset,
        batch_size=test_batch_size,
        sampler=SeqSampler(dataset,test_idx),
        num_workers=workers,
        pin_memory=True
    )

    model = resnet_lstm_dropout()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, class_num) 
    # print(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!") 
    #consider multi gpu formatted at module.
    state = torch.load(model_path)    
    newdict = {}    
    for k,v in state['state_dict'].items():
        if k[0:7] == 'module.': 
            name = k[7:]
            newdict[name] = v
        else:
            newdict[k] = v
    model.load_state_dict(newdict) 
    model = DataParallel(model)
    model.to(device)

    # criterion = nn.CrossEntropyLoss(size_average=False)

    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_start_time = time.time()

    all_preds = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # torch.cuda.empty_cache()            
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels[(sequence_length - 1)::sequence_length]

            if crop_type == 0 or crop_type == 1:
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs = model.module.forward(inputs)
       
            outputs = outputs[sequence_length - 1::sequence_length]
            Sm = nn.Softmax()
            outputs = Sm(outputs)
            pdb.set_trace()
            Class_Probability, _ = torch.max(outputs.data, 1)
            Class_Log_Probability = np.log2(Class_Probability.data.cpu())
            Entropy_Each_Cell = - np.multiply(Class_Probability.data.cpu(), Class_Log_Probability)
            pred_val_dicts[str(unselected[i])] = Entropy_Each_Cell
            print("select:",i)

    return pred_val_dicts

def non_local_select(val_model_path, pool_dataset, sequence_length, X, select_num,selected,unselected,mask):
    pred_val_dicts = val_for_selection(val_model_path, pool_dataset, sequence_length, unselected)
    from operator import itemgetter
    keys = []
    values = []
    #ranking probobility, low score in the front, high in the back
    for key, value in sorted(pred_val_dicts.items(), key = itemgetter(1), reverse = False):
        keys.append(int(key))
        values.append(float(value))
    # pick
    new_samples = keys[0:select_num-1]#pick the low probobility
    # update selected/unselected/mask
    selected.extend(new_samples)
    selected.sort()
    unselected = [j for j in X if j not in selected]
    for i in range(len(X)):
        if i in new_samples:
            mask[i] = 0

    return selected,unselected,mask

def CNN_emb_select(val_model_path, pool_dataset, sequence_length, X, select_num,selected,unselected,mask):
    pred_val_dicts = val_for_CNNembselection(val_model_path, pool_dataset, sequence_length, unselected,'dot')
    from operator import itemgetter
    keys = []
    values = []
    #ranking probobility, low score in the front, high in the back
    for key, value in sorted(pred_val_dicts.items(), key = itemgetter(1), reverse = False):
        keys.append(int(key))
        values.append(float(value))
    # pick
    new_samples = keys[0:select_num-1]#pick the low probobility
    # update selected/unselected/mask
    selected.extend(new_samples)
    selected.sort()
    unselected = [j for j in X if j not in selected]
    for i in range(len(X)):
        if i in new_samples:
            mask[i] = 0

    return selected,unselected,mask

def DBN_select(val_model_path, pool_dataset, sequence_length, X, select_num,selected,unselected,mask):
    pred_val_dicts = val_for_DBNselection(val_model_path, pool_dataset, sequence_length, unselected)
    from operator import itemgetter
    keys = []
    values = []
    #ranking probobility, low score in the front, high in the back
    for key, value in sorted(pred_val_dicts.items(), key = itemgetter(1), reverse = False):
        keys.append(int(key))
        values.append(float(value))
    # pick
    #invere to decending order
    keys_dc = keys[::-1] #high score in the front, low in the back
    #pick the large entropy
    new_samples = keys_dc[0:select_num-1]
    # update selected/unselected/mask
    selected.extend(new_samples)
    selected.sort()
    unselected = [j for j in X if j not in selected]
    for i in range(len(X)):
        if i in new_samples:
            mask[i] = 0

    return selected,unselected,mask

def p_value(result1,result2):
    #calculate p-value
    from scipy import stats
    import pickle
    '''
    path10 = 'results_ResLSTM_Nolocal/RESLSTM_NOLOCAL/1572444732.135057txtname8602_1571811265.416326.json_0.0005_tbs400_seq10_opt1_crop0_adamgamma0.1_adamstep3_adamweightdecay0.0001_block_num1/checkpoint_best-13_test_8075_crop_1.pkl'
    path20 = 'results_ResLSTM_Nolocal/RESLSTM_NOLOCAL/1572599460.4324906txtname17195_1572568659.835147.json_0.0005_tbs400_seq10_opt1_crop0_adamgamma0.1_adamstep3_adamweightdecay0.0001_block_num1/checkpoint_best-23_test_8237_crop_1.pkl'
    path30 = 'results_ResLSTM_Nolocal/RESLSTM_NOLOCAL/1572651009.994679txtname25788_1572620340.5691814.json_0.0005_tbs400_seq10_opt1_crop0_adamgamma0.1_adamstep3_adamweightdecay0.0001_block_num1/checkpoint_best-24_test_8238_crop_1.pkl'
    path40 = 'results_ResLSTM_Nolocal/RESLSTM_NOLOCAL/1572741346.4730155txtname34381_1572702907.6751492.json_0.0005_tbs400_seq10_opt1_crop0_adamgamma0.1_adamstep3_adamweightdecay0.0001_block_num1/checkpoint_best-16_test_8253_crop_1.pkl'
    path50 = 'results_ResLSTM_Nolocal/RESLSTM_NOLOCAL/1572847215.642195txtname42974_1572767025.1601517.json_0.0005_tbs400_seq10_opt1_crop0_adamgamma0.1_adamstep3_adamweightdecay0.0001_block_num1/checkpoint_best-23_test_8414_crop_1.pkl'
    '''
    
    with open(result1, 'rb') as f1:
        result_A = pickle.load(f1)
    with open(result2, 'rb') as f2:
        result_B = pickle.load(f2)

    # result_A = np.array([0.9,0.8,0.7,0.6,0.4])
    # result_B = np.array([0.9,0.8,0.7,0.6,0.5])
    
    # t_value,p_value = stats.ttest_ind(result_A,result_B)
    
    resultA_numpy = np.zeros((len(result_A),1))
    resultB_numpy = np.zeros((len(result_B),1))
    # prosB = val('../srccheckpoint/org/cxuecode/task2/full/2018_10_27_15_01_15/clean_2018_10_27_15_01_15_baseline_weightsS.hdf5')
    # t_value,p_value = stats.ttest_1samp(rvs,0.0)
    # pros = np.array([0.9,0.8,0.7,0.6,0.4])
    # pros_full = np.array([0.9,0.8,0.7,0.6,0.5])
    # t_value,p_value = stats.ttest_ind(pros[483:599],pros_full[483:599])
    
    for i in range(len(result_A)):
        resultA_numpy[i] = result_A[i].cpu().numpy()
        resultB_numpy[i] = result_B[i].cpu().numpy()
    t_value,p_value = stats.ttest_ind(resultA_numpy,resultB_numpy, equal_var = False)
    print (t_value,p_value)
    pdb.set_trace()