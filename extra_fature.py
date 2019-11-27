'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import logging
import argparse
import torchvision
#from models import *
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from itertools import combinations, permutations
#from utils import progress_bar
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='PyTorch UIUC Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--new_net', default=True, type=bool, help='W_linear')
parser.add_argument('--my_loss', default=True, type=bool, help='my_loss')
args = parser.parse_args()
logging.info(args)

store_name = "UIUC-train-features"

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset    = torchvision.datasets.ImageFolder(root='LabelMe/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=2)

testset    = torchvision.datasets.ImageFolder(root='LabelMe/test', transform=transform_train)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


queryset    = torchvision.datasets.ImageFolder(root='LabelMe/query', transform=transform_train)
queryloader = torch.utils.data.DataLoader(queryset, batch_size=1, shuffle=False, num_workers=2)

from scipy import stack

import torchvision.models as models

net =models.vgg16(pretrained=True)
#net =models.densenet121(pretrained=True)
net = net.features
for param in net.parameters():
    param.requires_grad = False

if use_cuda:
    net.cuda()
    cudnn.benchmark = True


x = torch.randn(2,3,224,224)
print(net(Variable(x.cuda())).size())

features_out = []
targets_out=[]
for batch_idx, (inputs, targets) in enumerate(trainloader):
    pass
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    outputs = net(inputs)
    features_out+=outputs.cpu().data.numpy().tolist()
    targets_out+=targets.cpu().data.numpy().tolist()

features_out = np.array(features_out).reshape(800,512,7,7)
np.save("LM-train-features-vgg16",features_out)



features_out = []
targets_out=[]
for batch_idx, (inputs, targets) in enumerate(testloader):
    pass
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    outputs = net(inputs)
    features_out+=outputs.cpu().data.numpy().tolist()
    targets_out+=targets.cpu().data.numpy().tolist()

features_out = np.array(features_out).reshape(800,512,7,7)
np.save("LM-test-features-vgg16",features_out)


features_out = []
targets_out=[]
for batch_idx, (inputs, targets) in enumerate(queryloader):
    pass
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    outputs = net(inputs)
    features_out+=outputs.cpu().data.numpy().tolist()
    targets_out+=targets.cpu().data.numpy().tolist()

features_out = np.array(features_out).reshape(8,512,7,7)
np.save("LM-query-features-vgg16",features_out)