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

args = argparse.ArgumentParser(description='Train.py')

args.add_argument('data_dir', nargs='?', action="store", default="./flowers", help='Image dataset directory')
args.add_argument('checkpoint_path', nargs='?', action="store", default="./checkpoint.pth", help='Checkpoint path')
args.add_argument('--gpu', dest="gpu", action="store_true", default="False", help='use gpu to speed up training')
args.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth", help='save a trained model to this directory')
args.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.01, help='learning rate')
args.add_argument('--epochs', dest="epochs", action="store", type=int, default=10, help='epochs')
args.add_argument('--arch', dest="arch", action="store", default="vgg19", type=str, help='select a network architecture')
args.add_argument('--hidden_units', dest="hidden_units", action="store", type=int, default=1024, help='hidden nodes')

args = args.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
lr = args.learning_rate
arch = args.arch
hidden_units= args.hidden_units
gpu = args.gpu
epochs = args.epochs
checkpoint = args.checkpoint_path

import json
with open('cat_to_name.json', 'r') as f:
    flower_to_name = json.load(f)
flower_species=len(flower_to_name)

image_datasets, dataloaders = u.loader(data_dir)
model = u.network(arch, gpu, hidden_units)
criterion, optimizer = u.optimizing(model, lr)

# Let's train
model = u.train(model, './ex_model.pth', epochs, optimizer, dataloaders, criterion, gpu)

# Let's test
u.test(dataloaders, model, criterion, gpu)
u.saver(arch, image_datasets, path, model, lr)