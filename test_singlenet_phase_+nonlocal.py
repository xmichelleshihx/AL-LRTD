import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn import DataParallel
import os
from PIL import Image
import time
import pickle
import numpy as np
import argparse
from torchvision.transforms import Lambda
import pdb
from models.resnet import i3_res50_nl,i3_res50_nl_new,i3_res50_nl_new_test,i3_res50_nl_new_test_1block, NonLocalBlock_test_Conv1,MyModel
from utils1.visualize import visualize_mini_batch

parser = argparse.ArgumentParser(description='lstm testing')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='use gpu, default True')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 8')
parser.add_argument('-t', '--test', default=10, type=int, help='test batch size, default 8')
parser.add_argument('-w', '--work', default=2, type=int, help='num of workers to use, default 4')
parser.add_argument('-n', '--name', type=str, help='name of model')
parser.add_argument(
    '-c', '--crop', default=0, type=int, help='0 rand, 1 cent, 2 resize, 5 five_crop, 10 ten_crop, default 2')
parser.add_argument('--class_num', default=7, type=int)
args = parser.parse_args()

'''
gpu_usg = ",".join(list(map(str, args.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
'''
sequence_length = args.seq
test_batch_size = args.test
workers = args.work
model_name = args.name
crop_type = args.crop
use_gpu = args.gpu
class_num = args.class_num

pkl_path = 'train40_val8_test41-80_paths_labels_server.pkl'
model_pure_name, _ = os.path.splitext(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('gpu             : ', device)
print('sequence length : {:6d}'.format(sequence_length))
print('test batch size : {:6d}'.format(test_batch_size))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))
print('name of this model: {:s}'.format(model_name))  # so we can store all result in the same file
print('Result store path: {:s}'.format(model_pure_name))

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


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
        self.fc = nn.Linear(512, 7)

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
        y = self.fc(y)
        return y

class resnet_lstm_nonlocal(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm_nonlocal, self).__init__()
        resnet_lstm_base = resnet_lstm()
        chkPath = '../AL_Res_LSTM/results/1568095370.2707942/checkpoint_best-25.pt'
        print("Restoring: ",chkPath)
        # Load
        state = torch.load(chkPath)
        # pdb.set_trace()
        # Initialize model and optimizer
        # newdict = {}    
        # for k,v in state['state_dict'].items():
        #     if k[0:5] == 'share': 
        #         name = 'module.share.' + k
        #         newdict[name] = v
        #     else:
        #         name = 'module.' + k
        #         newdict[name] = v
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

def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_test_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)

    test_paths = train_test_paths_labels[2]
    test_labels = train_test_paths_labels[5]
    test_num_each = train_test_paths_labels[8]

    print('test_paths   : {:6d}'.format(len(test_paths)))
    print('test_labels  : {:6d}'.format(len(test_labels)))

    test_labels = np.asarray(test_labels, dtype=np.int64)

    test_transforms = None
    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4310, 0.2971, 0.3126], [0.2405, 0.1863, 0.1935])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4310, 0.2971, 0.3126], [0.2405, 0.1863, 0.1935])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.4310, 0.2971, 0.3126], [0.2405, 0.1863, 0.1935])(crop) for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.4310, 0.2971, 0.3126], [0.2405, 0.1863, 0.1935])(crop) for crop in crops]))
        ])
    elif crop_type == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4310, 0.2971, 0.3126], [0.2405, 0.1863, 0.1935])
        ])
    elif crop_type == 3:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4310, 0.2971, 0.3126], [0.2405, 0.1863, 0.1935])
        ])

    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)
    return test_dataset, test_num_each

class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

def test_model(test_dataset, test_num_each):
    num_test = len(test_dataset)
    test_useful_start_idx = get_useful_start_idx(sequence_length, test_num_each)

    num_test_we_use = len(test_useful_start_idx)

    test_we_use_start_idx = test_useful_start_idx[0:num_test_we_use]

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)

    num_test_all = len(test_idx)

    print('num test start idx : {:6d}'.format(len(test_useful_start_idx)))
    print('last idx test start: {:6d}'.format(test_useful_start_idx[-1]))
    print('num of test dataset: {:6d}'.format(num_test))
    print('num of test we use : {:6d}'.format(num_test_we_use))
    print('num of all test use: {:6d}'.format(num_test_all))


    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             sampler=SeqSampler(test_dataset, test_idx),
                             num_workers=workers)

    # model = i3_res50_nl_new_test(400)
    # model = i3_res50_nl_new_test_1block(400)
    model = resnet_lstm_nonlocal()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, class_num) 
    print(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!") 
    #consider multi gpu formatted at module.
    state = torch.load(model_name)    
    newdict = {}    
    for k,v in state['state_dict'].items():
        if k[0:7] == 'module.': 
            name = k[7:]
            newdict[name] = v
        else:
            newdict[k] = v
    model.load_state_dict(newdict) 
    model = DataParallel(model)

    if use_gpu:
        model.to(device)

    criterion = nn.CrossEntropyLoss(size_average=False)

    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_start_time = time.time()

    all_preds = []
    pth_blobs = {}
    # f = open('./possibility.txt', 'a')

    with torch.no_grad():

        for data in test_loader:
            
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
       
            
            outputs = outputs[sequence_length - 1::sequence_length]

            _, preds = torch.max(outputs.data, 1)

            for i in range(len(preds)):
                all_preds.append(preds[i])
            print("all_preds length:",len(all_preds))
            loss = criterion(outputs, labels)
            test_loss += loss.data.item()
            test_corrects += torch.sum(preds == labels.data)

            print("preds:",preds)
            print("labels:",labels.data)
            # pdb.set_trace()
            test_loss += loss.data.item()
            print("test_corrects:",test_corrects)
            # f.write("preds:"+str(preds.cpu().numpy()))
            # f.write('\t')
            # f.write("labels:" + str(labels.data.cpu().numpy()))
            # f.write('\t')
            # f.write("possibility:" + str(possibility.cpu().numpy()))
            # f.write('\n')

    # f.close()

    test_elapsed_time = time.time() - test_start_time
    test_accuracy = float(test_corrects) / float(num_test_we_use)
    test_average_loss = test_loss / num_test_we_use

    # print('type of all_preds:', type(all_preds))
    # print('leng of all preds:', len(all_preds))
    save_test = int("{:4.0f}".format(test_accuracy * 10000))
    pred_name = model_pure_name + '_test_' + str(save_test) + '_crop_' + str(crop_type) + '.pkl'

    with open(pred_name, 'wb') as f:
        pickle.dump(all_preds, f)
    print('test elapsed: {:2.0f}m{:2.0f}s'
          ' test loss: {:4.4f}'
          ' test accu: {:.4f}'
          .format(test_elapsed_time // 60,
                  test_elapsed_time % 60,
                  test_average_loss, test_accuracy))


def main():
    # test_dataset, test_num_each = get_test_data('train40_val32-40_test40_paths_labels.pkl')
    test_dataset, test_num_each = get_test_data(pkl_path)
    
    # train40_val8_test32_paths_labels
    # train40_val32-40_test60-80_paths_labels.pkl
    test_model(test_dataset, test_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
