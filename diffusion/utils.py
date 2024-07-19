import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
        
    def forward(self, pred, target, weighted=1.):
        # pred target: [batch_size, action_dim]
        loss = self._loss(pred, target)
        weighted_loss = (loss * weighted).mean()
        return WeightedLoss
    
class WeightedL1(WeightedLoss):
    def _loss(self, pred, target):
        return torch.abs(pred - target)

class WeightedL2(WeightedLoss):
    def _loss(self, pred, target):
        return F.mse_loss(pred, target, reduction="none")