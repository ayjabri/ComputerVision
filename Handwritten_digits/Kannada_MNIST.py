# %% Import_libraries
"""
Kandana MNIST challange on kaggle.com
try with one dimension CNN
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt

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

# %% Custom Dataset
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

# %% DataLoaders
from torchvision import transforms    
import kornia.augmentation as k

tfms = transforms.Compose([transforms.RandomApply([k.RandomRotation(45.)],p=0.7),
                           transforms.Normalize((0.5,),(0.5,))])

#create a master dataset then split it into train and validation
dsm = MyDataset(img,label,transform=tfms) # Master Dataset
ds,dv = torch.utils.data.random_split(dsm,[60000,10240])

dl = DataLoader(ds,batch_size=100,shuffle=True) # training dataloader
dlv = DataLoader(dv,batch_size=100)             # validation dataloaderr

sampler = torch.utils.data.RandomSampler(ds,replacement=True,num_samples=500)
dls = DataLoader(ds,batch_size=100,sampler=sampler)

# %% Mixed1D2D Network
class Net(nn.Module):
    def __init__(self,out,kernel):
        super(Net,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(28,out,kernel,padding=3,bias=False),
                                   nn.BatchNorm1d(out),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv1d(out,out,kernel-2,padding=3,bias=False),
                                   nn.BatchNorm1d(out),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv1d(out,out,kernel-2,padding=3,bias=False),
                                   nn.BatchNorm1d(out),
                                   nn.LeakyReLU(inplace=True)
                                   )
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.drop1 = nn.Dropout(p=0.25)
        
        self.conv2 = nn.Sequential(nn.Conv2d(1,out,kernel,padding=3,bias=False),
                                   nn.BatchNorm2d(out),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(out,out,kernel-2,padding=3,bias=False),
                                   nn.BatchNorm2d(out),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(out,out,kernel-2,padding=3,bias=False),
                                   nn.BatchNorm2d(out),
                                   nn.ReLU(inplace=True))
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveMaxPool2d(1)
        self.drop2 = nn.Dropout(p=0.25)

        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(out*4,out*2),
                                nn.BatchNorm1d(int(out*2)),
                                nn.LeakyReLU(inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(out*2,10))
    def forward(self,x):
        x1 = x.squeeze(1)
        x1 = self.conv1(x1)
        p1 = self.pool1(x1)
        p2 = self.pool2(x1)
        x1 = torch.cat((p1,p2),dim=1)
        x1 = self.drop1(x1)
        x2 = self.conv2(x)
        p3 = self.pool3(x2)
        p4 = self.pool4(x2)
        x2 = torch.cat((p3,p4),dim=1)
        x2 = self.drop2(x2)
        x = torch.cat((x1,x2.squeeze(2)),dim=1)
        x = self.fc(x)
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


# %% Deep Network
    
class AdaptiveMixedPool1d(nn.Module):
    def __init__(self,sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.mp = nn.AdaptiveAvgPool1d(sz)
        self.ap = nn.AdaptiveMaxPool1d(sz)
    def forward(self,x): return torch.cat((self.mp(x),self.ap(x)),dim=1)

class AdaptiveMixedPool2d(nn.Module):
    def __init__(self,sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.mp = nn.AdaptiveAvgPool2d(sz)
        self.ap = nn.AdaptiveMaxPool2d(sz)
    def forward(self,x): return torch.cat((self.mp(x),self.ap(x)),dim=1)


class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
                
        self.conv1 = nn.Sequential(nn.Conv1d(28,64,7,padding=3,bias=False),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv1d(64,64,5,padding=3,bias=False),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv1d(64,64,5,padding=3,bias=False),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.MaxPool1d(2,stride=2),
                                    nn.Dropout(p=0.25),
                                    
                                    nn.Conv1d(64,128,3,padding=2,bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv1d(128,128,3,padding=2,bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv1d(128,128,3,padding=2,bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.MaxPool1d(2,stride=2),
                                    nn.Dropout(p=0.25),
                                    
                                    nn.Conv1d(128,256,3,padding=2,bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv1d(256,256,3,padding=2,bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv1d(256,256,3,padding=2,bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(inplace=True),
                                    AdaptiveMixedPool1d(1),
                                    nn.Dropout(p=0.25)
                                    )
        
        
        self.conv2 = nn.Sequential(nn.Conv2d(1,64,7,padding=3,bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(64,64,5,padding=3,bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(64,64,5,padding=3,bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.MaxPool2d(2,stride=2),
                                    nn.Dropout2d(p=0.25),
                                    
                                    nn.Conv2d(64,128,3,padding=2,bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(128,128,3,padding=2,bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(128,128,3,padding=2,bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.MaxPool2d(2,stride=2),
                                    nn.Dropout2d(p=0.25),
                                    
                                    nn.Conv2d(128,256,3,padding=2,bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(256,256,3,padding=2,bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(256,256,3,padding=2,bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(inplace=True),
                                    AdaptiveMixedPool2d(1),
                                    nn.Dropout2d(p=0.25)
                                    )
        
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.BatchNorm1d(1024),
                                nn.Linear(1024,256),
                                nn.BatchNorm1d(256),
                                nn.LeakyReLU(),
                                nn.Dropout(p=0.25),
                                nn.Linear(256,10)
                                )
    def forward(self,x):
        x1 = self.conv1(x.squeeze(1))
        x2 = self.conv2(x)
        x = torch.cat((x1,x2.squeeze(2)),dim=1)
        x = self.fc(x)
        return x

# %% Supporting_Modules

def GPU(img,label):
    if torch.cuda.is_available(): img,label = img.to(device),label.to(device)
    return img,label

@torch.no_grad()
def predict_all(net,dl):
    predicts = torch.tensor([])
    labels = torch.LongTensor([])
    for batch in dl:
        img,label = batch
        net.cpu()
        pred = net(img)
        predicts = torch.cat((predicts,pred),dim=0)
        labels = torch.cat((labels,label),dim=0)
        del img,label # To free memory.very effective when using GPUs
    return predicts,labels


def get_correct_num(predicts,labels):
    correct = predicts.softmax(dim=1).argmax(dim=1).eq(labels).sum().item()
    accuracy = round(correct/len(labels)*100,3)
    del predicts,labels
    return correct,accuracy


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
    total_losses = []
    total_correct= []
    for epoch in range(epochs+1):
        losses = 0
        correct = 0
        for batch in dl:
            optimizer.zero_grad()
            img,label = batch
            img,label = GPU(img,label)
            output = net(img)
            loss = loss_fn(output,label)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            c,_ = get_correct_num(output,label)
            correct += c
            del img,label
        accuracy = round(correct/len(dl.sampler)*100,3)
        total_correct.append(correct)
        total_losses.append(losses)
        if dv is not None:
            lossv = 0
            correctv = 0
            for valid in dv:
                imv,labelv = valid
                imv,labelv = GPU(imv,labelv)
                outputv = net(imv)
                lossv += float(loss_fn(outputv,labelv))
                cv,_= get_correct_num(outputv,labelv)
                correctv += cv
                del imv,labelv
            accuracyv = round(correctv/len(dv.sampler)*100,3)

            if epoch <=5:    
                print('''{:<3}| Loss: train {:10.12f}, valid {:10.12f} | Accuracy: train {:3.3f}%, valid:{:3}%
                      '''.format(epoch,losses,lossv,accuracy,accuracyv))
            else:
                if epoch%10 ==0:
                    print('''{:<3}| Loss: train {:10.12f}, valid {:10.12f} |Accuracy: train {:3.3f}%, valid:{:3}%
                      '''.format(epoch,losses,lossv,accuracy,accuracyv))
        else:
            if epoch <=5:    
                print('''{:<3}| Loss: train {:10.12f} | Accuracy: train {:3.3f}%
                      '''.format(epoch,losses,accuracy))
            else:
                if epoch%10 ==0:
                    print('''{:<3}| Loss: train {:10.12f} | Accuracy: train {:3.3f}%
                      '''.format(epoch,losses,accuracy))
    return total_losses,total_correct


class Runner():
    def __init__(self,out_channels,kernel_size,data,epochs,lr=1e-3):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.data = data
        self.epochs = epochs
        self.lr = lr
    def run(self):
        net = Net(self.out_channels,self.kernel_size).to(device)
#         summary = SummaryWriter(comment='filters:{} kernel:{} lr:{}'.format(
#             self.out_channels,self.kernel_size,self.lr))
#         summary.add_graph(net,img)
        l,a = learn(net,self.epochs,self.data,None,self.lr)
        
        plot_run(l,a,self.lr,self.out_channels,self.kernel_size)
        return net

def plot_run(losses,accuracy,lr=None,out_channels=None,kernel_size=None):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('lr={} out_channels={} kernel={}'.format(
        lr,out_channels,kernel_size))
    ax1.set_ylabel('losses', color=color)
    ax1.plot(losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(accuracy, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.show()

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