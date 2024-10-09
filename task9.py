from picamera2 import Picamera2, Preview
import time
import cv2

import time
import cv2


import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = mobilenet_v3_small(weights = None, num_classes = 2).to(device)
model.load_state_dict(torch.load('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-5/best_model.pth', map_location = device))
model.eval()

transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((64,64)), transforms.ToTensor()])

labels = ["laura", "Patrick"]

camera = cv2.VideoCapture(0)

ret, frame = camera.read()
if not ret:
    print("failed")
    camera.release()
    exit()

gret = cv