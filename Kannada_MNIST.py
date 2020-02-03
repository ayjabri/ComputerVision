# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import pandas as pd

# %%

train_df = pd.read_csv('/Users/aymanjabri/notebooks/Kannada/train.csv')
test_df = pd.read_csv('/Users/aymanjabri/notebooks/Kannada/test.csv')
dig_mnist = pd.read_csv('/Users/aymanjabri/notebooks/Kannada/Dig-MNIST.csv')

img_train = torch.from_numpy(train_df.drop(['label'],axis=1).values).float()
img_dig_mnist = torch.from_numpy(dig_mnist.drop(['label'],axis=1).values).float()
img = torch.cat((img_train,img_dig_mnist),dim=0)
img = img.view(img.size(0),-1,28,28)

label_train = torch.from_numpy(train_df.label.values).int()
label_dig_mnist = torch.from_numpy(dig_mnist.label.values).int()
label = torch.cat((label_train,label_dig_mnist),dim=0).long()

# %%
class MyDataset(Dataset):
    def __init__(self,x,y,transform=None):
        super(MyDataset,self).__init__()
        self.data = x
        self.targets = y
        self.transform = transform
    def __len__(self):
        return len(self.targets)
    def __getitem__(self,idx):
        img = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img,label


class Net1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(28,100,7,padding=3,bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.conv2 = nn.Conv1d(100,200,7,padding=3,bias=False)
        self.bn2 = nn.BatchNorm1d(200)
        self.pool1 = nn.AdaptiveMaxPool1d(2)
        self.pool2 = nn.AdaptiveAvgPool1d(2)
        self.bn3 = nn.BatchNorm1d(400)
        self.fc1 = nn.Linear(400*2,200)
        self.bn4 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200,10)

    def forward(self,x):
        x = x.squeeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        a = self.pool1(x)
        b = self.pool2(x)
        x = torch.cat((a,b),dim=1)
        x = self.bn3(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x
    

@torch.no_grad()
def predict_all(net,dl):
    predicts = torch.tensor([])
    labels = torch.tensor([]).int()
    for batch in dl:
        img,label = batch
        pred = net(img)
        predicts = torch.cat((predicts,pred),dim=0)
        labels = torch.cat((labels,label),dim=0)
    return predicts,labels

def get_correct_num(predicts,labels):
    correct = predicts.softmax(dim=1).argmax(dim=1).eq(labels).sum()
    accuracy = round(correct.item()/len(labels)*100,3)
    return correct.item(),accuracy

def learn(net,epochs,dl,lr=1e-3):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9)
    for epoch in range(1,epochs+1):
        losses = 0
        correct = 0
        for batch in dl:
            optimizer.zero_grad()
            img,label = batch
            output = net(img)
            loss = loss_fn(output,label)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            c,_ = get_correct_num(output,label)
            correct += c
        accuracy = round(correct/len(dl.sampler)*100,3)
        if epoch <=5:    
            print('{}: loss:{} accuracy:{}%'.format(epoch,losses,accuracy))
        else:
            if epoch%10 ==0:
                print('{}: loss:{} accuracy:{}%'.format(epoch,losses,accuracy))
    return losses,correct

# %%
from torchvision import transforms    

tfms = transforms.Compose([transforms.Normalize((0.5,),(0.5,))])
ds = MyDataset(img,label,transform=tfms)
dl = DataLoader(ds,batch_size=100,shuffle=True)

sampler = torch.utils.data.RandomSampler(ds,replacement=True,num_samples=500)
dls = DataLoader(ds,batch_size=100,sampler=sampler)
