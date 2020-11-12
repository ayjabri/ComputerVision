# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 08:42:23 2020

@author: ayjab
"""


import cv2
import numpy as np
import os
from PIL import Image

source = 'C:\\Users\\ayjab\\Videos\\Captures\\'
file = 'Zoom Meeting 2020-11-03 08-56-14.mp4'
destination = 'C:\\Users\\ayjab\\Videos\\Captures\\Extract\\'
name = 'zoom'

class ImageExtractor():
    '''
    Extract number of images from video. The scirpt will uniformally extract images based on the
    specificed number required. It can also write the images in the destination
    Source:     the path to video file
    n:          number of images needed
    destination:where to save the images
    name:       name of the images
    '''
    def __init__(self,source, n, destination=None, name=None):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.destination = destination
        self.n = n
        self.name = name

    def extract(self, save=False):
        if not self.cap.isOpened(): self.cap.open(source)
        total_n_frames = int(self.cap.get(7))
        step = int(total_n_frames/self.n)
        frames = []
        for i in range(0, total_n_frames,step):
            self.cap.set(1,i)
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
                if save:
                    if not os.path.exists(self.destination):
                        os.mkdir(self.destination)
                    cv2.imwrite(f'{self.destination}{self.name}_{i}.jpg', frame)
        self.cap.release()
        return np.array(frames)


def show(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)