import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader,Subset
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
import os
import datetime
from tensorboardX import SummaryWriter
import utils 
from glob import glob
import re
import pdb
import math
from models.resnet import i3_res50_nl,i3_res50_nl_new,i3_res50_nl_new_test,i3_res50_nl_new_test_1block,Bottleneck_test,NonLocalBlock_test_Conv1
from SAASmodule import save_select_data,load_select_data,random_select_data,non_local_select,DBN_select

from torch.nn import functional as F
import inspect

# from Pytorch_Memory_Utils.gpu_mem_track import  MemTracker
# from Pytorch_Memory_Utils.modelsize_estimate import modelsize

# CUDA_LAUNCH_BLOCKING=1 
# frame = inspect.currentframe() 
# gpu_tracker = MemTracker(frame)

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-dir', '--savedir', default='results', type=str, help='savefolder')
parser.add_argument('-old', '--ifloadold', default=False, type=bool, help='if to load the previous model')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=400, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=10, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=45, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=8, type=int, help='num of workers to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=5e-4, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('--adamstep', default=25, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--adamgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('--adamweightdecay',default=1e-4,type=float)
parser.add_argument('--block_num', default=1, type=int)
parser.add_argument("--is_first_selection",type=bool,default=False)
parser.add_argument("--quary_portion",default=0.1, type=float)
parser.add_argument("--save_select_txt_path",type=str)
parser.add_argument("--json_name",type=str)
parser.add_argument("--output_fold",type=str)
parser.add_argument("--val_model_path",type=str)
parser.add_argument("--is_save_json",type=bool,default=False)
parser.add_argument("--select_chose",type=str,default='no')
parser.add_argument("--summary_dir",type=str,default='no')
parser.add_argument('--class_num', default=7, type=int)
parser.add_argument('--train_mode', default='RESLSTM', type=str)
parser.add_argument('--FT_checkpoint',type=str)
parser.add_argument('--sv_init_model', type=str)

args = parser.parse_args()

save_dir_base = args.savedir
if_load_old = args.ifloadold
gpu_usg = args.gpu
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov
summary_dir = args.summary_dir

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep 
sgd_gamma = args.sgdgamma
adam_step = args.adamstep
adamgamma = args.adamgamma
adamweightdecay = args.adamweightdecay
block_num =  args.block_num
class_num = args.class_num
train_mode = args.train_mode
FT_checkpoint = args.FT_checkpoint

is_first_selection = args.is_first_selection
quary_portion = args.quary_portion
save_select_txt_path = args.save_select_txt_path
json_name = args.json_name
output_fold = args.output_fold
val_model_path = args.val_model_path
is_save_json = args.is_save_json
select_chose = args.select_chose
sv_init_model = args.sv_init_model

num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")
# os.environ["TORCH_HOME"] = "/research/dept5/xyshi/sxy/workspace1"

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('train batch size: {:6d}'.format(train_batch_size))
print('valid batch size: {:6d}'.format(val_batch_size))
print('optimizer choice: {:6d}'.format(optimizer_choice))
print('multiple optim  : {:6d}'.format(multi_optim))
print('num of epochs   : {:6d}'.format(epochs))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))
print('whether to flip : {:6d}'.format(use_flip))
print('learning rate   : {:.4f}'.format(learning_rate))
print('momentum for sgd: {:.4f}'.format(momentum))
print('weight decay    : {:.4f}'.format(weight_decay))
print('dampening       : {:.4f}'.format(dampening))
print('use nesterov    : {:6d}'.format(use_nesterov))
print('method for sgd  : {:6d}'.format(sgd_adjust_lr))
print('step for sgd    : {:6d}'.format(sgd_step))
print('gamma for sgd   : {:.4f}'.format(sgd_gamma))

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class RandomCrop(object):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomRotation(object):
    def __init__(self,degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        self.count += 1
        random.seed(seed)
        angle = random.randint(-self.degrees,self.degrees)
        return TF.rotate(img, angle)

class ColorJitter(object):
    def __init__(self,brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        self.count += 1
        random.seed(seed)
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img,brightness_factor)
        img_ = TF.adjust_contrast(img_,contrast_factor)
        img_ = TF.adjust_saturation(img_,saturation_factor)
        img_ = TF.adjust_hue(img_,hue_factor)
        
        return img_

class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels = file_labels[:, -1]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels = self.file_labels[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels

    def __len__(self):
        return len(self.file_paths)

class CholecDataset1(Dataset):
    def __init__(self, file_paths, file_labels, clip_len, num_each, split, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels = file_labels[:, -1]
        self.transform = transform
        self.loader = loader
        self.clip_len = clip_len
        self.split = split
        useful_start_idx = get_useful_start_idx(32, num_each)

        num_we_use = len(useful_start_idx) // num_gpu * num_gpu

        we_use_start_idx = useful_start_idx[0:num_we_use]

        self.idx_start = []
        for i in range(num_we_use):
            for j in range(self.clip_len):
                self.idx_start.append(we_use_start_idx[i] + j)
        # num_train_all = len(train_idx)
        # num_val_all = len(val_idx)

    def __getitem__(self, index):
        # entry = self.data[index]
        imgs = [self.loader(img) for img in self.file_paths[self.idx_start[index]:self.idx_start[index]+self.clip_len]]
        labels = [label for label in self.file_labels[self.idx_start[index]:self.idx_start[index]+self.clip_len]]
        # frames = self.transform(frame for frame in frames) # (T, 3, 224, 224)
        frames = []
        for i in range(len(imgs)):
            frames.append( self.transform(imgs[i]) ) # (T, 3, 224, 224)
        frames = frames.permute(1, 0, 2, 3) # (3, T, 224, 224)
        instance = {'frames':frames, 'label':labels}

        return instance

    def __len__(self):
        return len(self.idx_start)

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

class resnet_lstm_nonlocal(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm_nonlocal, self).__init__()
        resnet_lstm_base = resnet_lstm()
        chkPath = FT_checkpoint
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
        self.fcDropout = nn.Dropout(0.2)
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
    '''
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
        y_p = self.nl(y_p)
        # out = self.relu(out)
        y_p = self.fcDropout(y_p)
        y_p_p = y_p.permute(0, 2, 1)#(b x 512 x 10) - > b x 10 x 512
        y_p_p = y_p_p.contiguous().view(-1,y_p_p.shape[-1]) # 160 x 512
        return y_p_p'''

def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]
#    test_paths = train_test_paths_labels[2]
    train_labels = train_test_paths_labels[3]
    val_labels = train_test_paths_labels[4]
#    test_labels = train_test_paths_labels[5]
    train_num_each = train_test_paths_labels[6]
    val_num_each = train_test_paths_labels[7]
#    test_num_each = train_test_paths_labels[8]

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))
#    print('test_paths   : {:6d}'.format(len(test_paths)))
#    print('test_labels  : {:6d}'.format(len(test_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
#    test_labels = np.asarray(test_labels, dtype=np.int64)

    train_transforms = None
    test_transforms = None

    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.40577063,0.27282622,0.28533617],[0.24071056,0.19952665,0.20165241])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.40577063,0.27282622,0.28533617],[0.24071056,0.19952665,0.20165241])
        ])

    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.40577063,0.27282622,0.28533617],[0.24071056,0.19952665,0.20165241])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.40577063,0.27282622,0.28533617],[0.24071056,0.19952665,0.20165241])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.40577063,0.27282622,0.28533617],[0.24071056,0.19952665,0.20165241])(crop) for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.40577063,0.27282622,0.28533617],[0.24071056,0.19952665,0.20165241])(crop) for crop in crops]))
        ])

    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    val_dataset = CholecDataset(val_paths, val_labels, test_transforms)

    # print('num of train dataset: {:6d}'.format(num_train))
    # print('num train start idx : {:6d}'.format(len(train_useful_start_idx)))
    # print('last idx train start: {:6d}'.format(train_useful_start_idx[-1]))
    # print('num of train we use : {:6d}'.format(num_train_we_use))
    # print('num of all train use: {:6d}'.format(num_train_all))
    # print('num of valid dataset: {:6d}'.format(num_val))
    # print('num valid start idx : {:6d}'.format(len(val_useful_start_idx)))
    # print('last idx valid start: {:6d}'.format(val_useful_start_idx[-1]))
    # print('num of valid we use : {:6d}'.format(num_val_we_use))
    # print('num of all valid use: {:6d}'.format(num_val_all))
    # train_dataset = CholecDataset1(train_paths, train_labels, 32, train_num_each, 'train', train_transforms)
    # val_dataset = CholecDataset1(val_paths, val_labels, 32, val_num_each, 'val', test_transforms)
    # pdb.set_trace()
#    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each, val_dataset, val_num_each


class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    if if_load_old == True:
        pdb.set_trace()
        print ("please choose the previous one")
        time_cur = '1586310709.4848218'
    else:
        time_cur = time.time()

    writer = SummaryWriter(summary_dir + str(time_cur))
    logger = utils.get_log('log/'  + str(time_cur) + '.txt')

    # num_train = len(train_dataset)
    # num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)
    # train_idx = []
    # for i in range(num_train_we_use):
    #     for j in range(sequence_length):
    #         train_idx.append(train_we_use_start_idx[i] + j)

    val_idx = []
    for i in range(len(val_useful_start_idx)):
        for j in range(sequence_length):
            val_idx.append(val_useful_start_idx[i] + j)

    # num_train_all = len(train_idx)
    num_val_all = len(val_idx)
    # print('num of train dataset: {:6d}'.format(num_train))
    # print('num train start idx : {:6d}'.format(len(train_useful_start_idx)))
    # print('last idx train start: {:6d}'.format(train_useful_start_idx[-1]))
    # print('num of train we use : {:6d}'.format(num_train_we_use))
    # print('num of all train use: {:6d}'.format(num_train_all))
    # print('num of valid dataset: {:6d}'.format(num_val))
    # print('num valid start idx : {:6d}'.format(len(val_useful_start_idx)))
    # print('last idx valid start: {:6d}'.format(val_useful_start_idx[-1]))
    # print('num of valid we use : {:6d}'.format(num_val_we_use))
    # print('num of all valid use: {:6d}'.format(num_val_all))

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=workers,
        pin_memory=True
    )

    #select data to train
    X = train_useful_start_idx
    select_num = math.floor(len(X)*quary_portion) #every time choose 10% 
    if is_first_selection is True:
        pdb.set_trace()
        print ("this is first selectin!!!! please check your parameter in .sh")
        import random
        mask = [1 for n in range(0, len(X))]
        selected = random.sample(X,select_num)
        for i in range(len(X)):
            if X[i] in selected:
                mask[i] = 0
        unselected = [X[i] for i in range(len(X)) if X[i] not in selected]
        save_select_data(save_select_txt_path, selected,unselected,mask,time_cur)        
    else:
        # load_select_data return: data['selected'],data['unselected'],data['mask']
        selected, unselected, mask = load_select_data(os.path.join(save_select_txt_path,json_name))
        if select_chose == 'non_local':
            print ("this is non_local select")
            test_idx = []
            for i in range(len(unselected)):
                for j in range(sequence_length):
                    test_idx.append(unselected[i] + j)
            num_test_all = len(test_idx)
            subset = Subset(train_dataset,test_idx)
            selected,unselected,mask = non_local_select(val_model_path, subset, sequence_length, X, select_num,selected,unselected,mask)
        elif select_chose == 'DBN' :
            print ("this is DBN select")
            test_idx = []
            for i in range(len(unselected)):
                for j in range(sequence_length):
                    test_idx.append(unselected[i] + j)
            num_test_all = len(test_idx)
            subset = Subset(train_dataset,test_idx)
            selected,unselected,mask = DBN_select(val_model_path, subset, sequence_length, X, select_num,selected,unselected,mask)
        elif select_chose == 'random' :
            print ("this is random select")
            test_idx = []
            for i in range(len(unselected)):
                for j in range(sequence_length):
                    test_idx.append(unselected[i] + j)
            num_test_all = len(test_idx)
            selected,unselected,mask = random_select_data(X, select_num,selected,unselected,mask)
            pdb.set_trace()
            selected = [selected[i] for i in range(len(selected)) if selected[i] in test_idx]
        else :
            print ("just using old load select data to train without select new data")
            # pdb.set_trace()
        if is_save_json is True:
            save_select_data(save_select_txt_path, selected, unselected, mask,time_cur)
    pdb.set_trace()
    # save_dir = save_dir_base + '/' + str(time_cur) + '_' + str(learning_rate) + '_tbs' + str(train_batch_size) \
    # + '_seq' + str(sequence_length) + '_opt' + str(optimizer_choice) + '_crop' + str(crop_type) + '_adjlr' \
    #  + '_adamgamma' + str(adamgamma) + '_adamstep' + str(adam_step) + '_weight_decay' + str(adamweightdecay) + '_block_num' + str(block_num)
    if train_mode == 'RESLSTM' or train_mode == 'RESLSTM_DBN':
        save_dir = save_dir_base + '/' + str(train_mode) + '/' + str(time_cur) + 'txtname' + json_name + '_' + str(learning_rate) + '_tbs' + str(train_batch_size) \
        + '_seq' + str(sequence_length) + '_opt' + str(optimizer_choice) + '_crop' + str(crop_type)  \
        + '_sgdstep' + str(sgd_step) + '_sgd_gamma' + str(sgd_gamma) + '_sgd_adjust_lr' + str(sgd_adjust_lr)+ '_weight_decay' + str(weight_decay)
    elif train_mode == 'RESLSTM_NOLOCAL' or train_mode == 'RESLSTM_NOLOCAL_dropout0.2':
        save_dir = save_dir_base + '/' + str(train_mode) + '/' + str(time_cur) + 'txtname' + json_name + '_' + str(learning_rate) + '_tbs' + str(train_batch_size) \
        + '_seq' + str(sequence_length) + '_opt' + str(optimizer_choice) + '_crop' + str(crop_type)  \
         + '_adamgamma' + str(adamgamma) + '_adamstep' + str(adam_step) + '_adamweightdecay' + str(adamweightdecay) + '_block_num' + str(block_num)

    if if_load_old == True:
        # Check if a checkpoint is in there
        if len([name for name in os.listdir(save_dir)]) > 0:
            print("Loading old model")
        else:
            print ("nothing to load")
            pdb.set_trace()
            
    else:
        os.makedirs(save_dir)

    if train_mode == 'RESLSTM':
        model = resnet_lstm()
    elif train_mode == 'RESLSTM_NOLOCAL' :
        model = resnet_lstm_nonlocal()
    elif train_mode == 'RESLSTM_NOLOCAL_dropout0.2':
        model = resnet_lstm_nonlocal()
        chk = 'results_ResLSTM_Nolocal/RESLSTM_NOLOCAL/1572847215.642195txtname42974_1572767025.1601517.json_0.0005_tbs400_seq10_opt1_crop0_adamgamma0.1_adamstep3_adamweightdecay0.0001_block_num1/checkpoint_best-23.pt'
        print("Restoring: ",chk)
        # Load
        state = torch.load(chk)
        # newdict = {}    
        # for k,v in state['state_dict'].items():
        #     if k[0:7] != 'module.': 
        #         name = 'module.' + k
        #         newdict[name] = v
        #     else:
        #         newdict[k] = v
        
        model.load_state_dict(state['state_dict'])  
    elif train_mode == 'RESLSTM_DBN':
        model = resnet_lstm_dropout()
    else:
        print ("not implemented")
        pdb.set_trace()
    # print (model)
    # pdb.set_trace()
    if use_gpu:
        model = DataParallel(model)
        model.to(device)

    criterion = nn.CrossEntropyLoss(size_average=False)

    optimizer = None
    exp_lr_scheduler = None

    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_step, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_step, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            # optimizer = optim.Adam([
            #     {'params': model.module.share.parameters()},
            #     {'params': model.module.lstm.parameters(), 'lr': learning_rate},
            #     {'params': model.module.fc.parameters(), 'lr': learning_rate},
            # ], lr=learning_rate / 10)
            optim_params = list(filter(lambda p: p.requires_grad, model.parameters()))
            print ('Optimizing %d paramters'%len(optim_params))
            optimizer = optim.Adam(optim_params, lr=learning_rate,weight_decay=adamweightdecay)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=adam_step, gamma=adamgamma)

    #check if need load old weigth, optimizer
    if if_load_old:
        # Find last, not last best checkpoint
        files = glob(save_dir+'/*')
        global_steps = np.zeros([len(files)])
        for i in range(len(files)):
            # Use meta files to find the highest index
            if 'best' in files[i]:
                continue
            if 'checkpoint-' not in files[i]:
                continue                
            # Extract global step
            nums = [int(s) for s in re.findall(r'\d+',files[i])]
            global_steps[i] = nums[-1]
        # Create path with maximum global step found
        chkPath = save_dir + '/checkpoint-' + str(int(np.max(global_steps))) + '.pt'
        print("Restoring: ",chkPath)
        # Load
        state = torch.load(chkPath)
        # Initialize model and optimizer
        newdict = {}    
        for k,v in state['state_dict'].items():
            if k[0:7] != 'module.': 
                name = 'module.' + k
                newdict[name] = v
            else:
                newdict[k] = v
        model.load_state_dict(newdict) 
        # model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])  
        # pdb.set_trace()   
        start_epoch = state['epoch']
        best_epoch = int(np.max(global_steps))
        best_val_accuracy = state['best_val_accuracy']
        correspond_train_acc = state['correspond_train_acc']
    else:
        start_epoch = 1
        best_epoch = -1
        best_val_accuracy = 0.0
        correspond_train_acc = 0.0
        if sv_init_model is not None:
            print("Restoring supervised model: ",sv_init_model)
            # Load
            state = torch.load(sv_init_model)
            # Initialize model and optimizer
            newdict = {}    
            for k,v in state['state_dict'].items():
                if k[0:7] != 'module.': 
                    name = 'module.' + k
                    newdict[name] = v
                else:
                    newdict[k] = v
            model.load_state_dict(newdict) 

    best_model_wts = copy.deepcopy(model.module.state_dict())

    for epoch in range(start_epoch, epochs+1):
        np.random.shuffle(selected)
        train_idx = []
        for i in range(len(selected)):
            for j in range(sequence_length):
                train_idx.append(selected[i] + j)
        num_train_all = len(train_idx)
        # subset = Subset(train_dataset,train_idx)
        # train_loader = DataLoader(
        #     subset,
        #     batch_size=train_batch_size,
        #     sampler=SeqSampler(subset, train_idx),
        #     num_workers=workers,
        #     pin_memory=True
        # )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_dataset, train_idx),
            num_workers=workers,
            pin_memory=True
        )
        # pdb.set_trace()
        # Sets the module in training mode.
        model.train()
        train_loss = 0.0
        train_corrects = 0
        batch_progress = 0.0
        train_start_time = time.time()
        for data in train_loader:           
            optimizer.zero_grad()
            # torch.cuda.empty_cache()
            with torch.set_grad_enabled(True): 
                if use_gpu:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    labels = labels[(sequence_length - 1)::sequence_length]
                else:
                    inputs, labels = data[0], data[1]
                    labels = labels[(sequence_length - 1)::sequence_length]
                # pdb.set_trace()
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                # pdb.set_trace()
                outputs = model.forward(inputs)
                # pdb.set_trace()
                outputs = outputs[sequence_length - 1::sequence_length]
                _, preds = torch.max(outputs.data, 1)            
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.data.item()
                batch_corrects = torch.sum(preds == labels.data)
                train_corrects += batch_corrects

                batch_acc=float(batch_corrects)/train_batch_size*sequence_length

                batch_progress += 1
                if batch_progress*train_batch_size >= num_train_all:
                    percent = 100.0
                    print('Batch progress: %s [%d/%d] Batch acc:%.2f' % (str(percent) + '%', num_train_all, num_train_all, batch_acc), end='\n')
                else:
                    percent = round(batch_progress*train_batch_size / num_train_all * 100, 2)
                    print('Batch progress: %s [%d/%d] Batch acc:%.2f' % (str(percent) + '%', batch_progress*train_batch_size, num_train_all, batch_acc), end='\r')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy = float(train_corrects) / float(num_train_all)*sequence_length
        train_average_loss = train_loss / num_train_all*sequence_length

        # Sets the module in evaluation mode.
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_start_time = time.time()
        val_progress = 0

        with torch.no_grad():
            for data in val_loader:
                # torch.cuda.empty_cache()
                if use_gpu:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    labels = labels[(sequence_length - 1)::sequence_length]
                else:
                    inputs, labels = data[0], data[1]
                    labels = labels[(sequence_length - 1)::sequence_length]

                if crop_type == 0 or crop_type == 1:
                    inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                    outputs = model.forward(inputs)

                elif crop_type == 5:
                    inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                    inputs = inputs.view(-1, 3, 224, 224)
                    outputs = model.forward(inputs)
                    outputs = outputs.view(5, -1, 7)
                    outputs = torch.mean(outputs, 0)
                elif crop_type == 10:
                    inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                    inputs = inputs.view(-1, 3, 224, 224)
                    outputs = model.forward(inputs)
                    outputs = outputs.view(10, -1, 7)
                    outputs = torch.mean(outputs, 0)

                outputs = outputs[sequence_length - 1::sequence_length]

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data)

                val_progress += 1
                if val_progress*val_batch_size >= num_val_all:
                    percent = 100.0
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', num_val_all, num_val_all), end='\n')
                else:
                    percent = round(val_progress*val_batch_size / num_val_all * 100, 2)
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress*val_batch_size, num_val_all), end='\r')

        val_elapsed_time = time.time() - val_start_time
        val_accuracy = float(val_corrects) / float(num_val_all)*sequence_length
        val_average_loss = val_loss / num_val_all*sequence_length
        write_dict = {"train_loss":train_average_loss,"val_loss":val_average_loss,"train_accuracy":train_accuracy,"val_accuracy":val_accuracy}
        writer.add_scalars('scalar', write_dict, epoch)  

        if optimizer_choice == 0:
            if sgd_adjust_lr == 0:
                exp_lr_scheduler.step()
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler.step(val_average_loss)
        
        if optimizer_choice == 1:
            exp_lr_scheduler.step()

        if val_accuracy >= best_val_accuracy:
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                correspond_train_acc = train_accuracy
                best_model_wts = copy.deepcopy(model.module.state_dict())
                oldBestInd = best_epoch
                best_epoch = epoch
            if val_accuracy == best_val_accuracy:
                if train_accuracy > correspond_train_acc:
                    correspond_train_acc = train_accuracy
                    best_model_wts = copy.deepcopy(model.module.state_dict())
                    oldBestInd = best_epoch
                    best_epoch = epoch
            # Delte previously best model
            if os.path.isfile(save_dir + '/checkpoint_best-' + str(oldBestInd) + '.pt'):
                os.remove(save_dir + '/checkpoint_best-' + str(oldBestInd) + '.pt')
            # Save currently best model
            state = {'epoch': epoch,'state_dict': best_model_wts,'optimizer': optimizer.state_dict(),
                    'best_val_accuracy':best_val_accuracy,'correspond_train_acc':correspond_train_acc}
            torch.save(state, save_dir + '/checkpoint_best-' + str(epoch) + '.pt')       

        # If its not better, just save it delete the last checkpoint if it is not current best one
        # Save current model
        state = {'epoch': epoch,'state_dict': model.module.state_dict(),'optimizer': optimizer.state_dict(),
                'best_val_accuracy':best_val_accuracy,'correspond_train_acc':correspond_train_acc}
        torch.save(state, save_dir + '/checkpoint-' + str(epoch) + '.pt')                           
        # Delete last one
        if os.path.isfile(save_dir + '/checkpoint-' + str(epoch-1) + '.pt'):
            os.remove(save_dir + '/checkpoint-' + str(epoch-1) + '.pt') 

        logger.info("\n")
        logger.info('Epoch: %d/%d (%d h %d m %d s)' % (epoch, epochs, int(train_elapsed_time/3600), int(np.mod(train_elapsed_time,3600)/60), int(np.mod(np.mod(train_elapsed_time,3600),60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
        logger.info('validation time: %d h %d m %d s' % (int(val_elapsed_time/3600), int(np.mod(val_elapsed_time,3600)/60), int(np.mod(np.mod(val_elapsed_time,3600),60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
        logger.info("training loss: %6f" % train_average_loss)
        logger.info("validation loss: %6f" % val_average_loss)
        logger.info("train accu: %6f" % train_accuracy)
        logger.info("validation accu: %6f" % val_accuracy)
        logger.info("best val accu: %6f at Epoch %d" % (best_val_accuracy,best_epoch))
        logger.info("best corresponding train accu: %6f" % correspond_train_acc)
    writer.close()
    # print('best accuracy: {:.4f} cor train accu: {:.4f}'.format(best_val_accuracy, correspond_train_acc))

def main():
    # pkl_path = 'train32_val8_test40_paths_labels.pkl'
    # pkl_path = 'train40_val8_test32_paths_labels.pkl'
    # pkl_path = 'train40_val32-40_test40_paths_labels.pkl'
    # pkl_path = 'train40_val8_test32_paths_labels.pkl'
    pkl_path = 'train40_val8_test41-80_paths_labels_server.pkl'
    train_dataset, train_num_each, val_dataset, val_num_each = get_data(pkl_path)
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
