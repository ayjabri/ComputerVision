#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 19:57:27 2020

@author: aymanjabri
Classify one of the MNIST datasets using adaptive filter number, run multiple 
batch sizes,learning rates and other hyper parameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms,datasets,models
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from collections import Counter,OrderedDict,namedtuple
from itertools import product

#Setup the GUP
if torch.cuda.is_available(): device=torch.device('cuda:0')
else: device=torch.device('cpu')

#Tensorboard
# tb = SummaryWriter(log_dir='runs',comment='Network finder')

##Prepare the data for training,validation
    #Transforms
tfms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.5,),(0.5,))])
    #Download the data and define the datasets
trns = datasets.MNIST('/Users/aymanjabri/notebooks/MNIST',train=True,transform=
                    tfms,download=True)
tsts = datasets.MNIST('/Users/aymanjabri/notebooks/MNIST',train=False,transform=
                    tfms,download=True)
    #Datasets imbalance:
print(sorted(Counter(trns.targets.numpy()).items()))
print(sorted(Counter(tsts.targets.numpy()).items()))

    #Combine test and training datasets:This step is not necessary if you are happy with current distribution
ds = torch.utils.data.ConcatDataset((trns,tsts))

        #Split training and validation sets
trainset,validset = torch.utils.data.random_split(ds,[50000,20000])

train = torch.utils.data.DataLoader(trainset,batch_size=100,shuffle=True)
valid = torch.utils.data.DataLoader(validset,batch_size=500)

    #View data
img,label = next(iter(train))
imgs = tv.utils.make_grid(img,nrow=10,padding=1,normalize=True)
# tb.add_image('Snapshot from Training Set',imgs)
plt.figure(figsize=(12,12))
plt.imshow(imgs.permute(1,2,0))

'''Create a small dataloader to overfit the model with,
before introducing the full dataset'''
    #Define sampling method

weights= 100/(torch.bincount(trns.targets).double())
weighted = torch.utils.data.WeightedRandomSampler(weights,num_samples=100
                                               ,replacement=True)
random = torch.utils.data.RandomSampler(trns,replacement=True,num_samples=100)

    #After trying so many different sampling methods i went with random,
    #because it gave the most balanced results.
dl_small= torch.utils.data.DataLoader(trns,batch_size=100,sampler=random)

xs,ys = next(iter(dl_small))
# print(sorted(Counter(ys.numpy()).items()))

##Build The CNN
class Net(nn.Module):
    def __init__(self,out,kernel):
        super().__init__()
        self.conv1 = nn.Conv2d(1,out,kernel,padding=1)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out,out,kernel,padding=1)
        self.bn2 = nn.BatchNorm2d(out)
        self.pool1 = nn.AdaptiveMaxPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(out*2,out)
        self.bn = nn.BatchNorm1d(out)
        self.lin2 = nn.Linear(out,10)
    def forward(self,x):
        x = self.bn1(F.relu_(self.conv1(x)))
        x = self.bn2(F.relu_(self.conv2(x)))
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        x = torch.cat((p1,p2),dim=1)
        x = x.view(x.size(0),-1)
        # x = self.convs(x)
        x = self.bn(F.relu(self.lin1(x)))
        x = self.lin2(x)
        return x

def get_correct_num(predict,label):
    correct = torch.argmax(predict.softmax(dim=1),dim=1).eq(label)
    return correct.sum().item()

def learn(net,data,epochs,tb,lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    total_losses = []
    total_accuracy = []
    for epoch in range(1,epochs+1):
        e_loss = 0
        correct = 0
        for batch in data:
            optimizer.zero_grad()
            img,label = batch
            output = net(img)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            e_loss += loss.item()
            correct += get_correct_num(output,label)
        e_acc = round(correct/(len(data)*data.batch_size)*100,5)
        if epoch <=10:
            print('''Epoch:{} Training Loss {} Training Accuracy {}%
              '''.format(epoch,e_loss,e_acc))
        else:
            if epoch%(epochs/20)==0:
                print('''Epoch:{} Training Loss {} Training Accuracy {}%
              '''.format(epoch,e_loss,e_acc))
        total_losses.append(e_loss)
        total_accuracy.append(e_acc)
        
        tb.add_scalar('Losses',e_loss,epoch)
        tb.add_scalar('Accuracy',e_acc,epoch)
        for name,param in net.named_parameters():
            tb.add_histogram('{}'.format(name), param)
        
    tb.close()
    return total_losses,total_accuracy

def predict_all(net,loader):
    with torch.no_grad():
        predict = torch.tensor([])
        labels = torch.tensor([]).int()
        for img,label in loader:
            if torch.cuda.is_available(): img,label=img.to(device),label.to(device)
            p=net(img)
            predict = torch.cat((predict,p),dim=0)
            labels = torch.cat((labels,label.int()),dim=0)
        return predict,labels
        
class runner():
    def __init__(self,out_channels,kernel_size,data,epochs,lr=1e-3):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.data = data
        self.epochs = epochs
        self.lr = lr
    def run(self):
        net = Net(self.out_channels,self.kernel_size)
        summary = SummaryWriter(comment='filters:{} kernel:{} lr:{}'.format(
            self.out_channels,self.kernel_size,self.lr))
        summary.add_graph(net,img)
        l,a = learn(net,self.data,self.epochs,summary,self.lr)
        
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('lr={} out_channels={} kernel={}'.format(
            self.lr,self.out_channels,self.kernel_size))
        ax1.set_ylabel('losses', color=color)
        ax1.plot(l, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
        ax2.plot(a, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        return net

summary = SummaryWriter()
summary.add_scalars(m)