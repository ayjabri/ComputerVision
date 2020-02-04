# %% Import_libraries
"""
Kandana MNIST challange on kaggle.com
try with one dimension CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# %% GPU/CPU
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# %% Load Data

train_df = pd.read_csv('/Users/aymanjabri/notebooks/Kannada/train.csv')
test_df = pd.read_csv('/Users/aymanjabri/notebooks/Kannada/test.csv')
dig_mnist = pd.read_csv('/Users/aymanjabri/notebooks/Kannada/Dig-MNIST.csv')

img_train = torch.from_numpy(train_df.drop(['label'],axis=1).values).float()
img_dig_mnist = torch.from_numpy(dig_mnist.drop(['label'],axis=1).values
                                 ).float()

#Concatenate images and labels from both csv files to create master X,Y
img = torch.cat((img_train,img_dig_mnist),dim=0)
img = img.view(img.size(0),-1,28,28)

label_train = torch.from_numpy(train_df.label.values).int()
label_dig_mnist = torch.from_numpy(dig_mnist.label.values).int()
label = torch.cat((label_train,label_dig_mnist),dim=0).long()

# %% Network
class MyDataset(Dataset):
    def __init__(self,x,y,transform=None):
        super(MyDataset,self).__init__()
        self.data = x
        self.targets = y
        self.transform = transform
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# %% Import_resnet
from torchvision import models

resnet = models.resnet152(pretrained=True)
for node in resnet.modules():
    if type(node)==nn.BatchNorm2d:
        node.weight.requires_grad=True
    else: # another way is to run: resnet.requiers_grad_(False)
        for param in node.parameters():
            param.requires_grad=False

resnet.fc = nn.Sequential(nn.Linear(2048,1024),
nn.BatchNorm1d(1024),
nn.ReLU(),
nn.Dropout(0.25),
nn.Linear(1024,10))   
resnet.conv1 = nn.Conv2d(1,64,7,stride=1,padding=3,bias=False)         

class Net1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(nn.Conv1d(28,84,7,padding=3,bias=False),
                                   nn.BatchNorm1d(84),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv1 = nn.Conv1d(84,100,3,padding=3,bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.conv2 = nn.Conv1d(100,200,3,padding=3,bias=False)
        self.bn2 = nn.BatchNorm1d(200)
        self.pool1 = nn.AdaptiveMaxPool1d(2)
        self.pool2 = nn.AdaptiveAvgPool1d(2)
        self.bn3 = nn.BatchNorm1d(400)
        self.fc1 = nn.Linear(400*2,200)
        self.bn4 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200,10)

    def forward(self,x):
        x = x.squeeze(1)
        x = self.input(x)
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
    
    def init_weights(self):
        for param in self.children():
            if isinstance(param,(nn.Conv1d,nn.Conv2d)):
                nn.init.kaiming_uniform_(param.weight)
                if param.bias is not None: nn.init.xavier_uniform(param.bias)
            elif isinstance(param,nn.Linear):
                param.reset_parameters() #resets them using kaiming
 #https://github.com/pytorch/pytorch/
 # blob/3b1c3996e1c82ca8f43af9efa196b33e36efee37/torch/nn/modules/linear.py
                nn.init.kaiming_uniform_(param.weight)


net = Net1D().to(device)

# %% Supporting_Modules

def GPU(img,label):
    if torch.cuda.is_available(): img,label = img.to(device),label.to(device)
    return img,label

@torch.no_grad()
def predict_all(net,dl):
    predicts = torch.tensor([],device=device)
    labels = torch.tensor([],device=device).int()
    for batch in dl:
        img,label = batch
        img,label = GPU(img,label)
        pred = net(img)
        predicts = torch.cat((predicts,pred),dim=0)
        labels = torch.cat((labels,label),dim=0)
    return predicts,labels

def get_correct_num(predicts,labels):
    correct = predicts.softmax(dim=1).argmax(dim=1).eq(labels).sum()
    accuracy = round(correct.item()/len(labels)*100,3)
    return correct.item(),accuracy

def learn(net,epochs,dl,dv=None,lr=1e-3):
    '''
    Train the network:
        net: CNN
        epochs: number of epochs
        dl: Training DataLoader
        dv: Validation DataLoader
        lr: Learning rate (defualt=1e-3)
    '''
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9)
    for epoch in range(1,epochs+1):
        losses = 0
        correct = 0
        for batch in dl:
            optimizer.zero_grad()
            img,label = batch
            GPU(img,label)
            output = net(img)
            loss = loss_fn(output,label)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            c,_ = get_correct_num(output,label)
            correct += c
        accuracy = round(correct/len(dl.sampler)*100,3)
        if dv is not None:
            lossv = 0
            correctv = 0
            for valid in dv:
                imv,labelv = valid
                GPU(imv,labelv)
                outputv = net(imv)
                lossv += loss_fn(outputv,labelv)
                cv,_=get_correct_num(outputv,labelv)
                correctv += cv
            accuracyv = round(correctv/len(dv.sampler)*100,3)

            if epoch <=5:    
                print('''{:<3}|Loss: train {:10.12f}, valid {:10.12f} |Accuracy: train {:3}%, valid:{:3}%
                      '''.format(epoch,losses,lossv,accuracy,accuracyv))
            else:
                if epoch%10 ==0:
                    print('''{:<3}|Loss: train {:10.12f}, valid {:10.12f} |Accuracy: train {:3}%, valid:{:3}%
                      '''.format(epoch,losses,lossv,accuracy,accuracyv))
        else:
            if epoch <=5:    
                print('''{:<3}|Loss: train {:10.12f} | Accuracy: train {:3}%
                      '''.format(epoch,losses,accuracy))
            else:
                if epoch%10 ==0:
                    print('''{:<3}|Loss: train {:10.12f} | Accuracy: train {:3}%
                      '''.format(epoch,losses,accuracy))
    return losses,correct


# %% DataLoaders
from torchvision import transforms    

tfms = transforms.Compose([transforms.Normalize((0.5,),(0.5,))])

#create a master dataset then split it into train and validation
dsm = MyDataset(img,label,transform=tfms) # Master Dataset
ds,dv = torch.utils.data.random_split(dsm,[60000,10240])

dl = DataLoader(ds,batch_size=100,shuffle=True) # training dataloader
dlv = DataLoader(dv,batch_size=100)             # validation dataloaderr

sampler = torch.utils.data.RandomSampler(ds,replacement=True,num_samples=500)
dls = DataLoader(ds,batch_size=100,sampler=sampler)

# %% Submit_Results
'''
Run this cell only after you train the network 'net' and happy with the 
results
'''
xt = torch.from_numpy(test_df.drop(labels='id',axis=1).values).float()
xt = xt.view(xt.size(0),-1,28,28)
dlt = DataLoader(xt,batch_size=100)

predictions = torch.tensor([])
for imgt in dlt:
    predictions = torch.cat((predictions,net(imgt)))

# %%
# torch.save(resnet.state_dict,'/Users/aymanjabri/notebooks/Kannada/ResNet152_Pre.pt')
resnet.load_state_dict(torch.load('/Users/aymanjabri/notebooks/Kannada/ResNet152_Pre.pt'))
    
    
    
    
    
    
    
    
    
    
    
    