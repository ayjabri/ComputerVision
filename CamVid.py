#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:42:53 2020

@author: aymanjabri
"""

import torch
import torch.nn as nn
import torchvision as tv
import os
from glob import glob
from torchvision import datasets,transforms,models
import matplotlib.pyplot as plt
import numpy as np

path = '/Users/aymanjabri/notebooks/SegNet-Tutorial/CamVid'

