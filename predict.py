# Imports here
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from PIL import Image

import time
import os
import random

import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torch.autograd import Variable

import argparse
import utility as u
import pandas as pd

args = argparse.ArgumentParser(
    description='predict.py')
args.add_argument('input_img', nargs='*', action="store", type=str, default='./flowers/test/98/image_07777.jpg', help='image path')
args.add_argument('checkpoint', nargs='*', action="store", type=str, default='./checkpoint.pth', help='path for model checkpoint')
args.add_argument('--top_k', dest="top_k", action="store", type=int,  default=5, help='return top K most likely classes')
args.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', help='category mapper')
args.add_argument('--gpu', action="store_true", dest="gpu", default="False", help='use gpu to speed up inference')

args = args.parse_args()
image_path = args.input_img
top_k = args.top_k
gpu = args.gpu
checkpoint = args.checkpoint
category_names=args.category_names
image_path = image_path[1]
path_list = image_path.split('/')

import json
with open(category_names, 'r') as f:
    flower_to_name = json.load(f)
flower_species=len(flower_to_name)
target_class=flower_to_name[path_list[-2]] # To get the category
print("Target class: " + target_class)

model = u.load_checkpoint(checkpoint)
value, kclass = u.predict(image_path, model, gpu, top_k)

idx_to_class = {model.class_to_idx[i]:i for i in model.class_to_idx.keys()} # dict comprehension
classes = [flower_to_name[idx_to_class[c]] for c in kclass] # list comprehension

data = {'Predicted Class': classes, 'Probablity': value}
dataframe = pd.DataFrame(data)
print(dataframe.to_string(index=False))
