# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:46:44 2020
@author: Ayman Al Jabri

Run this script to train your model 
This script will detect faces via your webcam using multithread
There should be no delay as a result of getting the faces from the model.
Tested with OpenCV
"""

import torch
import joblib
# import numpy as np
import pandas as pd
from utils import features_training as ftrain


import facenet_pytorch as facenet
from torch.utils.data import DataLoader
from torchvision import datasets




net = facenet.InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = facenet.MTCNN(image_size=260)
path = "data/train"

def collate_fn(x):
    return x[0]

def unpack_loader(loader, mtcnn, p=False):
    faces = []
    classes = []
    probs = []
    for x, y in loader:
        face, prob = mtcnn(x, return_prob=True)
        if p: print(f'Detected {loader.dataset.dataset.classes[y]:20} face with a probability of {prob:.3f}%')
        if prob > 0.7:
            faces.append(face)
            classes.append(y)
            probs.append(prob)
    return (torch.stack(faces),
            torch.tensor(classes),
            torch.tensor(probs)
            )

def plot_features(X,y):
    pass



if __name__=='__main__':

    df = datasets.ImageFolder(path)
    train_n = int(len(df) *0.8)
    test_n = len(df) - train_n
    train_df, test_df = torch.utils.data.random_split(df, [train_n, test_n])

    train_loader = DataLoader(train_df, collate_fn=collate_fn)
    test_loader = DataLoader(test_df, collate_fn=collate_fn)

    faces, classes, probs = unpack_loader(train_loader, mtcnn, True)
    faces_t, classes_t, probs_t = unpack_loader(test_loader, mtcnn)



    features = net(faces).detach()
    features_t = net(faces_t).detach()

    classes_min = torch.bincount(classes).min().item()
    cv = min(classes_min, 4)

    search = ftrain.BestModel(cv=cv, params=ftrain.params)
    search.fit(features, classes)

    pd.DataFrame(df.classes).to_csv('results/classes.csv',header=['name'], index=False)
    joblib.dump(search.best_classifier, 'results/model.joblib')
    'gcloud beta ai-platform predict --model face --version v1 --json-instances filename.json'

