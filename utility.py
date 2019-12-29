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

def loader(data_dir):
    # Directories
    data_dir = './flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    # D-R-Y
    t_tensor = transforms.ToTensor()
    t_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          t_tensor,
                                          t_normalize
                                         ])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          t_tensor,
                                          t_normalize
                                         ])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         t_tensor,
                                         t_normalize
                                         ])


    # TODO: Load the datasets with ImageFolder
    training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_data  = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # TODO: Using the image datasets and the transforms, define the dataloaders
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    testing_loader  = torch.utils.data.DataLoader(testing_data, batch_size=64)

    image_datasets = [training_data, validation_data, testing_data]
    dataloaders = [training_loader, validation_loader, testing_loader]

    return image_datasets, dataloaders

def network(arch, gpu, hidden_units):
    # Get a VGG pre-trained network
    model = getattr(models, arch)(pretrained=True)
    
    # Disable autograd, to prevent "does not require grad and does not have a grad_fn" error
    for param in model.parameters():
        param.requires_grad = False
    
    # Fresh, feed-forward classifier network
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('drop', nn.Dropout(p=0.5)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    if gpu and torch.cuda.is_available():
        model = model.cuda()
    return model

def optimizing(model, lr):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
    return criterion, optimizer

def train(model, save_path, epochs, optimizer, dataloaders, criterion, gpu):
    # Start training
    epochs = epochs
    steps = 0
    if gpu and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    running_loss = 0
    accuracy = 0

    print('Begin training')
    start = time.time()

    for e in range(epochs):    
        train_mode = 0
        valid_mode = 1

        for mode in [train_mode, valid_mode]:   
            if mode == train_mode:
                model.train()
            else:
                model.eval()

            pass_count = 0

            for data in dataloaders[mode]:
                pass_count += 1
                inputs, labels = data
                if gpu and torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                    
                # Disable autograd
                optimizer.zero_grad()
                # Feed forward
                output = model.forward(inputs)
                loss = criterion(output, labels)

                if mode == train_mode:
                    # Back propagate
                    loss.backward()
                    optimizer.step()                

                running_loss += loss.item()
                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()

            if mode == train_mode:
                print("\nEpoch: {}/{} ".format(e+1, epochs),
                      "\nTraining Loss: {:.4f}  ".format(running_loss/pass_count))
            else:
                print("Validation Loss: {:.4f}  ".format(running_loss/pass_count),
                  "Accuracy: {:.4f}".format(accuracy))

            running_loss = 0
    end = time.time()

    duration = end - start
    print("\nTraining took: {:.0f}m {:.0f}s".format(duration//60, duration % 60))

def validate():
    # TODO: Do validation on the test set
    model.eval()
    accuracy = 0
    model.cuda()

    pass_count = 0

    for data in dataloaders[2]:
        pass_count += 1
        images, labels = data

        images, labels = Variable(images.cuda()), Variable(labels.cuda())

        output = model.forward(images)
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Testing Accuracy: {:.4f}".format(accuracy/pass_count))
    return None

def test(loaders, model, criterion, gpu):
    test_loss = 0.
    actual_loss = 0.
    total = 0.

    model.eval()
    
    for batch_idx, (data, target) in enumerate(loaders['test']):
        if gpu and torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        
        output = model(data)
        loss = criterion(output, target)
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        predicted = output.data.max(1, keepdim=True)[1]
        actual_loss += np.sum(np.squeeze(pred.eq(target.data.view_as(predicted))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100.0 * correct / total, correct, total))
    return None

def saver(arch, dataset, path, model, lr):
    # TODO: Save the checkpoint 
    model.class_to_idx = dataset[0].class_to_idx

    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'arch': arch,
                  'learning_rate': lr,
                  'batch_size': 64,
                  'classifier' : model.classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, path)
    return None

def load_checkpoint(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    learning_rate = checkpoint['learning_rate']
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
        
    return model

def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if gpu and torch.cuda.is_available():
        model.cuda()
    
    # model.eval()
    image = im_processor(image_path)
    # print(image)
    # image = torch.from_numpy(np.array([image]))
    # image = image.numpy()
    
    image = Variable(image)
    if gpu and torch.cuda.is_available():
        image = image.cuda() # Use the GPU
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    prob = torch.topk(probabilities, topk)[0].tolist()[0]
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(topk):
        label.append(ind[index[i]])

    return prob, label

def im_processor(image):
    img_pil = Image.open(image)
    img_np=np.array(img_pil)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return adjustments(img_pil)

