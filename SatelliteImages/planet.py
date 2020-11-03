#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:42:53 2020

@author: aymanjabri
reference:
https://github.com/mratsim/Amazon-Forest-Computer-Vision/blob/master/main_pytorch.py#L131-L140

"""
# %%
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from glob import glob
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
# import kornia.augmentation as k

# %%

# GPU
if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')
print(device)

#Prepare the data
path = '/Users/aymanjabri/notebooks/Planet/data'
tags = pd.read_csv(path+'/train_v2.csv/train_v2.csv')
x = glob(path+'/train-jpg/*')

classes = ['clear', 'cloudy', 'haze','partly_cloudy',
    'agriculture','artisinal_mine','bare_ground','blooming',
    'blow_down','conventional_mine','cultivation','habitation',
    'primary','road','selective_logging','slash_burn','water']

class_to_idx=dict(zip(classes,range(len(classes))))
a = lambda x: x.split(' ')
t = tags.tags.apply(a)

y = torch.zeros(len(t),len(classes))
for row in range(len(t)):
    for tag in t[row]:
        y[row,class_to_idx[tag]]+=1

targets = pd.DataFrame(y.numpy(),dtype=int,columns=classes)
weights = torch.from_numpy(targets.sum().values/len(t))

#np.nonzero()

########## Write a custom Datasets   ###########
class PlanetDataset(Dataset):
    def __init__(self,x,y,classes,class_to_idx,transform=None):
        self.pathes = x
        self.targets = y
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
    def __len__(self):
        return len(self.pathes)
    def __getitem__(self,idx):
        img,label = self.pathes[idx], self.targets[idx]
        img = Image.open(img)
        img = img.convert('RGB')
        if self.transform is not None:
            img=self.transform(img)
        return img,label

# %%
########## Prepare Datasets and Dataloaders  ###########
tfms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
ds = PlanetDataset(x,y,classes,class_to_idx,transform=tfms)
dl = DataLoader(ds,batch_size=17,shuffle=True,drop_last=True,num_workers=4,
                pin_memory=True)
# %%
########  Create a small dl to pre-train the network ###########
tfmss = transforms.Compose([transforms.Resize((50,50)),
        transforms.RandomApply([transforms.RandomRotation(degrees=25),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip()]),               
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
dss = PlanetDataset(x,y,classes,class_to_idx,transform=tfmss)
sample = torch.utils.data.RandomSampler(dss,replacement=True,num_samples=100)
dls = DataLoader(dss,batch_size=25,shuffle=False,sampler=sample)

# %%
########## View the Data  ###########
img,label = next(iter(dls))
imgs = tv.utils.make_grid(img,nrow=(5),normalize=True,padding=1)
plt.figure(figsize=(14,14))
plt.imshow(imgs.permute(1,2,0))

# %%

###CNN
class Net(nn.Module):
    def __init__(self,out,kernel):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3,out,kernel_size=kernel,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out,out,kernel_size=kernel,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(1)
        self.lin1 = nn.Linear(out*2,out)
        self.bnl1 = nn.BatchNorm1d(out)
        self.lin2 = nn.Linear(out,out)
        self.bnl2 = nn.BatchNorm1d(out)
        self.lin3 = nn.Linear(out,17)
    def forward(self,x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        a = self.pool1(x)
        b = self.pool2(x)
        x = torch.cat((a,b),dim=1)
        x = x.view(x.size(0),-1)
        x = self.bnl1(F.relu(self.lin1(x)))
        x = self.bnl2(F.relu(self.lin2(x)))
        x = self.lin3(x)
        return x

# %%
########## Predict All ###########
def predict_all(net,dataloader):
    with torch.no_grad():
        predict = torch.tensor([])
        targets = torch.tensor([])
        for batch in dataloader:
            x,y = batch
            if torch.cuda.is_available():
                x,y = x.to(device),y.to(device)
            predict = torch.cat((predict,net(x)),dim=0)
            targets = torch.cat((targets,y),dim=0)
    return predict,targets
            
########## Get Correct Number ###########
def get_correct_num(predict,target):
    z = torch.tensor([0.])
    o = torch.tensor([1.])
    p = torch.where(predict>=0.5,o,z)
    c = p.eq(target).sum().float()
    return c

# %% 8
########## Learner ###########

def learn(net,dataloader,epochs,lr):
    loss_fn = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(filter(
        lambda a:a.requires_grad,net.parameters()),lr=lr,momentum=0.9)
    for epoch in range(1,epochs+1):
        eloss = 0
        ecorrect = 0
        etargets = 0
        for batch in dataloader:
            optimizer.zero_grad()
            img,label = batch
            if torch.cuda.is_available():
                img,label = img.to(device),label.to(device)
            output = net(img)
            loss = loss_fn(output,label)
            loss.backward()
            optimizer.step()
            eloss +=loss.item()
            ecorrect += get_correct_num(output,label).item()
            etargets +=label.numel()
        accuracy = round(ecorrect/etargets*100,2)
        if epoch<=5:
            print('Epoch:{} Loss:{} Accuracy:{}'.format(epoch,eloss,accuracy))
        else:
            if epoch%(epochs/5)==0:
                print('Epoch:{} Loss:{} Accuracy:{}%'.format(epoch,eloss,accuracy))

# %%
                
pred = [' '.join(labels[y_pred_row > 0.21]) for y_pred_row in y_pred]
