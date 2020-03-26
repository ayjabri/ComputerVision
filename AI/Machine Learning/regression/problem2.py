#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:31:24 2020

@author: aymanjabri
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys

# =============================================================================
# Pre-Processing the Data
# =============================================================================

def Pre_Process(path):
    data = pd.read_csv(path,names=['age', 'weight','height'])
    x = torch.from_numpy(data.drop(['height'],axis=1).values).float()
    y = torch.from_numpy(data.height.values)
    x = torch.cat((torch.ones(len(x),1),x),dim=1)
    return x,y

def Normalize(x):
    mean,std = x.mean(dim=0),x.std(dim=0)    
    for i in range(1,3):
        x[:,i] = (x[:,i]-mean[i])/std[i]

# =============================================================================
# Loss Function
# =============================================================================

def MSE(y_hat,y):
    return ((y_hat-y)**2).mean()/2

def learn(x,y,epochs,lr):
    g = []
    w = nn.Parameter(torch.zeros(x.dim()+1))
    for epoch in range(epochs):
        y_hat = x@w
        loss = MSE(y_hat,y)
        loss.backward()
        # print(loss.item(),w.grad)
        with torch.no_grad():
            w.sub_(lr * w.grad)
            if epoch==epochs-1: 
                g = w.tolist()
                g.insert(0,epochs)
                g.insert(0,lr)
            w.grad.zero_()
    return g
        

def main():
    
    α = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10,3e-1]
    
    epochs = 100

    in_file = sys.argv[1].lower()

    out_file = sys.argv[2].lower()

    x,y = Pre_Process(in_file)

    Normalize(x)
    
    np.savetxt(out_file,[learn(x,y,epochs,lr) for lr in α],delimiter=',')
        
    
if __name__ == '__main__':

    main()