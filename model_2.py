import torch
from torch import Tensor
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class SegmentNet(nn.Module):
  def __init__(self):
    '''
    Runs through convolutional layer, Relu, Batch Norm and MaxPool (2x2) 3 times.
    Then flattens and finalizes in a dim 4 softmax vector structure.
    '''
    super().__init__()
    
    self.segment_net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3), 
        nn.ReLU(True), 
        nn.BatchNorm2d(64),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), 
        nn.ReLU(True), 
        nn.BatchNorm2d(num_features=128),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3), 
        nn.ReLU(True),
        nn.BatchNorm2d(num_features=256),
        nn.MaxPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(50176, 256),
        nn.ReLU(True),
        nn.BatchNorm1d(num_features=256),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(True),
        nn.Linear(128, 5),
        nn.Sigmoid(),
        nn.Softmax(dim=1)
    )
    
  def forward(self, x): 
    """Performs forward pass

    Arguments
    ---------
    x: Tensor
      image tensor of shape (B, 3, 37, 37)
    
    Returns
    -------
    Tensor
      logits (ranging from 0 to 1) tensor with shape (B, 1000)
    """
    logits = self.segment_net(x)
    return logits