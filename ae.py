# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import  shutil
import  numpy as np

import pandas as pd
import os
from copy import deepcopy
import math
from scipy.io import mmread, mmwrite
from scipy import  linalg
import numpy as np
import time
from scipy.io import mmread
import pandas as pd
import argparse
import pickle
import os
import importlib.util


import rec9 as rec
import pathlib
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import subprocess as sbp

parser = argparse.ArgumentParser(description='PyTorch AE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden-size', type=int, default=100, metavar='HS',
                    help='how many hidden units to put in bottleneck')
parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='WD',
                    help='weight-decay value, default 0.01')
# parser.add_argument('--weight', type=float, default=1, metavar='WD',
#                     help='weight-decay value, default 1')
# parser.add_argument('--noise', type=float, default=0.1, metavar='WD',
#                     help='weight-decay value, default 0.1')
parser.add_argument('--dropout', type=float, default=0.0, metavar='WD',
                    help='weight-decay value, default 0.1')
parser.add_argument('--thread-num', type=int, default=16, metavar='WD',
                    help='weight-decay value, default 16')
parser.add_argument('--log-dir', type=str, default="./", metavar='WD',
                    help='weight-decay value, default ./')
parser.add_argument('--data', type=str)
parser.add_argument('--split', type=str)
parser.add_argument('--cv', type=int)

args = parser.parse_args()

rec.THREAD_NUM = 16


param = dict(vars(args))
param.pop('data')
param.pop('split')
param.pop('momentum')
param.pop('log_interval')
param.pop('log_dir')
param.pop('no_cuda')
param.pop('epochs')
param.pop('thread_num')
param_str = ("%s" % param).strip("{}").replace("'", "").replace(":", "_").replace(",", "-").replace(" ", "")
model_dir_path = os.path.join(args.log_dir, "ae-%s"%param_str)
pathlib.Path(model_dir_path).mkdir(parents=True, exist_ok=True)


class MyDataset(Dataset):

    def __init__(self, path,flip_prob=0):
        super(MyDataset, self).__init__()
        train_data = sio.mmread(path)
        self.train_data = train_data.A.astype(np.float32)
        self.user_num, self.item_num = self.train_data.shape
        self.flip_prob = flip_prob

    def __getitem__(self, index):
        item_profile = self.train_data[:,index]

        return (item_profile, item_profile)


    def __len__(self):
        return self.item_num



args.cuda = not args.no_cuda and torch.cuda.is_available()

def get_path(data,split,cv):
    path = "/dir_to_data_folder"
    return os.path.join(path, "profile.%s" % cv), os.path.join(path, "input.%s" % cv), os.path.join(path, "train.%s" % cv), os.path.join( path, "test.%s" % cv)


profile_path, input_path,train_path, test_path = get_path(args.data, args.split, args.cv)

train_data = mmread(train_path).A
input_data = mmread(input_path).A
profile_data = mmread(profile_path).A
test_data = mmread(test_path).A
targets = np.unique(test_data.nonzero()[0])

dataset = MyDataset(train_path)
dataset.test_data = sio.mmread(test_path).A.astype(np.float32)
dataset.targets = np.unique(dataset.test_data.nonzero()[0])
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)



max_pred = {'prec@5':0, 'pairwise_auc':0,'prec@1':0}
# max_cf = {'prec@5':0, 'pairwise_auc':0}



class AENet(nn.Module):
    def __init__(self, input_size, hidden_size,dropout=0 ):
        super(AENet, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,input_size)
        self.rep = None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / (input_size + hidden_size)))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.fc1(x)
        # out = F.relu(out)
        if self.training == False:
            self.rep = out.cpu().data.numpy().copy()
        out = self.fc2(out)
        return out



model = AENet(dataset.user_num, args.hidden_size,args.dropout)
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
def evaluate_pred(pred):
    r = rec.Rec()
    perf = r.evaluate2(dataset.train_data,dataset.test_data,dataset.targets,pred, [1,5])

    return perf


def test_on_pred(epoch,batch_idx, pred):
 
    pred_ret = evaluate_pred(pred)
    pred_ret.update(param)
    pred_ret['loss']=loss.data[0]
    df = pd.DataFrame([pred_ret])
    df['batch_idx'] = batch_idx
    df['epoch'] = epoch
    df['model'] = 'ae-pred'
    df['idx'] = batch_idx+(epoch+1)*args.batch_size
    with open(os.path.join(model_dir_path,"ae-linear-pred-results-%s-epoch_%s-iteration_%s.csv" % (param_str,epoch, batch_idx)), 'a') as f:

        df.to_csv(f, index=False)
    print("epoch: %s, batch : %s, prec@5: %s, reca@5: %s"%(epoch,batch_idx+1, df['prec@5'].data[0],df['reca@5'].data[0]))
    
    return pred_ret



def test(epoch, batch_idx):

    model.eval()
    data = Variable(torch.from_numpy(dataset.train_data.T))
    if args.cuda:
        data = data.cuda()
    output = model(data)
    hidden = model.rep
    mmwrite(os.path.join(model_dir_path,'ae-linear-rep-%s-epoch_%s-batch_%s'%(param_str,epoch,batch_idx)), hidden)
    perf_pred = test_on_pred(epoch,batch_idx,output.data.cpu().numpy().T)
    print("epoch %s batch_idx %s: %s"%(epoch,batch_idx,perf_pred))


for epoch in range(1, args.epochs + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            with open(os.path.join(model_dir_path,'loss-%s.csv'%param_str), 'a') as f:
                res = {"param_str": param_str, "epoch": epoch, "batch+1": batch_idx + 1, 'loss': loss.data[0],'model':'ae-linear-pred'}
                res.update(param)
                pd.DataFrame([res]).to_csv(f, header=False, index=False)

            print('Train Epoch {} Batch {} Loss: {:.6f}'.format( epoch, batch_idx+1 , loss.data[0]))
            test(epoch,batch_idx+1)
            print()
# learning rate schedual depending on dataset
        if epoch == 3 and batch_idx + 1 == 1:
            optimizer = optim.Adam(model.parameters(), lr=args.lr / 5, weight_decay=args.weight_decay)
        if epoch % 3 == 0 and batch_idx+1 == 1:
            optimizer = optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr']*0.9,weight_decay=args.weight_decay)
       

