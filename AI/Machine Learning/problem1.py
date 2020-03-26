import pandas as pd
import torch
import torch.nn as nn
import sys
import numpy as np

def pre_process(path):
    data = pd.read_csv(path,sep=',',header=None,names=['x1','x2','y'])
    y = torch.from_numpy(data.y.values).view(-1,1)
    x = torch.from_numpy(data.drop(['y'],axis=1).values).float()
    return x,y

def activate(t):
    t[t>0]=1
    t[t<0]=-1
    return t

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptron = nn.Linear(2,1)
    def forward(self,x):
        x = self.perceptron(x)
        return x
    
def train(net,x,y,epochs,lr=0.1):
    loss_fn = nn.SoftMarginLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    weights = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = net(x)
        loss = loss_fn(output,y.view(-1,1).float())
        loss.backward()
        optimizer.step()
        weights.append([net.perceptron.weight[0][0].item(),
                        net.perceptron.weight[0][1].item(),
                        net.perceptron.bias[0].item()])
        #if epoch%(epochs/20) ==0: print(loss.item())
        activate(output)
        if torch.all(output==y.view(-1,1)):
            return weights

def Perceptron(x,y,lr):
    n = len(x)
    dim = x.dim()
    x = torch.cat([torch.ones(n).view(-1,1),x],dim=1)
    w = torch.randn((dim+1),dtype=torch.float)
    no_match = True
    while no_match:
        predict = activate(x@w)
        if torch.all(predict!=y):
            no_match=True
            w += y*lr*x
    return w

# x,y = data_process('./input1.csv')


#$ python3 problem1.py input1.csv output1.csv


def main():

    in_file = sys.argv[1].lower()

    out_file = sys.argv[2].lower()

    x,y = pre_process(in_file)
    
    net = Net()
    
    weights = train(net,x,y,10000)
    
    np.savetxt(out_file,weights, delimiter=',')
    
    
if __name__ == '__main__':

    main()