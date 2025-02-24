#!/usr/bin/python
# encoding: utf-8

import os
import random
from PIL import Image
import numpy as np
from image import *
import torch

from torch.utils.data import Dataset
from utils_ps import read_truths_args, read_truths, get_all_files, read_6d_poses

class listDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, cell_size=32, num_keypoints=9, max_gt_num=5):

        # root             : list of training or test images
        # shape            : shape of the image input to the network
        # shuffle          : whether to shuffle or not
        # tranform         : any pytorch-specific transformation to the input image
        # target_transform : any pytorch-specific tranformation to the target output
        # train            : whether it is training data or test data
        # max_gt_num       : maximum number of ground-truth labels an image can have

        # read the list of dataset images
        with open(root, 'r') as file:
            self.lines = file.readlines()

        # Shuffle
        if shuffle:
            random.shuffle(self.lines)

        # Initialize variables
        self.nSamples         = len(self.lines)
        self.transform        = transform
        self.target_transform = target_transform
        self.train            = train
        self.shape            = shape
        self.cell_size        = cell_size
        self.num_keypoints    = num_keypoints
        self.max_gt_num       = max_gt_num # maximum number of ground-truth labels an image can have
    
    # Get the number of samples in the dataset
    def __len__(self):
        return self.nSamples

    # Get a sample from the dataset
    def __getitem__(self, index):

        # Ensure the index is smallet than the number of samples in the dataset, otherwise return error
        assert index <= len(self), 'index range error'

        # Get the image path
        imgpath = self.lines[index].rstrip()

        if self.train:
            # Decide on how much data augmentation you are going to apply
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            # Get the data augmented image and their corresponding labels
            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure, self.num_keypoints, self.max_gt_num)

            # Convert the labels to PyTorch variables
            label = torch.from_numpy(label)
        
        else:
            # Get the validation image, resize it to the network input size
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            # Read the validation labels, allow upto max_gt_num ground-truth objects in an image
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.jpeg','.txt')
            num_labels = 2 * self.num_keypoints + 3  # +2 for ground-truth of width/height , +1 for class label
            label = torch.zeros(self.max_gt_num*num_labels)
            if os.path.getsize(labpath):
                ow, oh = img.size
                tmp = torch.from_numpy(read_truths_args(labpath))
                tmp = tmp.view(-1)
                tsz = tmp.numel()
                if tsz > self.max_gt_num*num_labels:
                    label = tmp[0:self.max_gt_num*num_labels]
                elif tsz > 0:
                    label[0:tsz] = tmp

        # Tranform the image data to PyTorch tensors
        if self.transform is not None:
            img = self.transform(img)

        # If there is any PyTorch-specific transformation, transform the label data
        if self.target_transform is not None:
            label = self.target_transform(label)


        # Return the retrieved image and its corresponding label
        return (img, label)


class PoseDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, max_gt_num=5):

        # root             : list of training or test images
        # shape            : shape of the image input to the network
        # shuffle          : whether to shuffle or not
        # tranform         : any pytorch-specific transformation to the input image
        # target_transform : any pytorch-specific tranformation to the target output
        # max_gt_num       : maximum number of ground-truth labels an image can have

        # read the list of dataset images
        with open(root, 'r') as file:
            self.lines = file.readlines()

        # Shuffle
        if shuffle:
            random.shuffle(self.lines)

        # Initialize variables
        self.nSamples = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.shape = shape
        self.max_gt_num = max_gt_num  # maximum number of ground-truth labels an image can have

    # Get the number of samples in the dataset
    def __len__(self):
        return self.nSamples

    # Get a sample from the dataset
    def __getitem__(self, index):

        # Ensure the index is smallet than the number of samples in the dataset, otherwise return error
        assert index <= len(self), 'index range error'

        # Get the image path
        imgpath = self.lines[index].rstrip()

        # Get the validation image, resize it to the network input size
        img = Image.open(imgpath).convert('RGB')
        if self.shape:
            img = img.resize(self.shape)

        # Read the validation labels, allow upto max_gt_num ground-truth objects in an image
        labpath = imgpath.replace('JPEGImages', 'ObjectPose').replace('.jpeg', '.txt')
        if os.path.getsize(labpath):
            pose = torch.from_numpy(read_6d_poses(labpath)).view(-1)

        # Tranform the image data to PyTorch tensors
        if self.transform is not None:
            img = self.transform(img)

        # If there is any PyTorch-specific transformation, transform the label data
        if self.target_transform is not None:
            pose = self.target_transform(pose)

        # Return the retrieved image and its corresponding label
        return (img, pose)