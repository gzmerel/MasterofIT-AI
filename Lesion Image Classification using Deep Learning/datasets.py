import collections
import csv
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image

# Define a dataset class for loading lesion images and their labels
# The dataset can apply data augmentation and five-crop transformations
# The labels are expected to be in a CSV file with the first column as image names
# and the subsequent columns as one-hot encoded labels

class LesionDataset(torch.utils.data.Dataset):
  def __init__(self, img_dir, labels_fname, augment = False, five_crop = False):
    self.img_dir = Path(img_dir)
    #Load the csv file 
    df = pd.read_csv(labels_fname)
    #Store filenames include .jpg extention and labels
    self.image_filenames = [self.img_dir /(name + ".jpg") for name in df['image']]
    self.labels  = df.values[:,1:].argmax(axis = 1).tolist()
    self.five_crop = five_crop
    #Transor PIL format to Tensor 
    if self.five_crop:
        self.transform = transforms.Compose([
        transforms.Resize((300,400)),
        transforms.FiveCrop((200,300)),
        transforms.Lambda(lambda crops: torch.stack([
        transforms.ToTensor()(crop) for crop in crops]))
          ])
    elif augment:
        self.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
        ])
    else:
        self.transform = transforms.ToTensor()
    

  def __len__(self):
    return len(self.labels)

   
  def __getitem__(self, idx):
    image_path = self.image_filenames[idx]
    image = Image.open(image_path).convert("RGB")
    if self.five_crop:
      input = self.transform(image)  # shape: [5, C, H, W]
    else:
      input = self.transform(image)
    label = self.labels[idx]
    return input, label
  def normalise_data(self, mean, std):
    self.inputs = ((self.original_inputs-mean)/std).astype(np.float32)
pass
