#%%
import cv2
import torch
import datetime
import argparse
import joblib
import numpy as np
from utils import features_training as ftrain


import facenet_pytorch as facenet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


net = facenet.InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = facenet.MTCNN(image_size=260)
path = "data\\train"

def collate_fn(x):
    return x[0]

def unpack_loader(loader, mtcnn, p=False):
    faces = []
    classes = []
    probs = []
    for x, y in loader:
        face, prob = mtcnn(x, return_prob=True)
        if p: print(f'Found {loader.dataset.dataset.classes[y]} face with a probability of {prob:3f}%')
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


#%%
if __name__=='__main__':

    df = datasets.ImageFolder(path)
    train_n = int(len(df) *0.8)
    test_n = len(df) - train_n
    train_df, test_df = torch.utils.data.random_split(df, [train_n, test_n])

    train_loader = DataLoader(train_df, collate_fn=collate_fn)
    test_loader = DataLoader(test_df, collate_fn=collate_fn)

    faces, classes, probs = unpack_loader(train_loader, mtcnn)
    faces_t, classes_t, probs_t = unpack_loader(test_loader, mtcnn)

    features = net(faces).detach()
    features_t = net(faces_t).detach()

    search = ftrain.BestModel(ftrain.params)
    search.fit(features, classes)


