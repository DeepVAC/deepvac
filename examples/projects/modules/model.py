# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from torchvision import models


model = models.resnet50(pretrained=True)

in_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 3), nn.LogSoftmax(dim=1))
