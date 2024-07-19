import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        # Here the position encoding is the same as transformer
        # PE(pos, 2i) = sin(pos / 10000 ^ {2i / dmodel})
        # PE(pos, 2i + 1) = cos(pos / 10000 ^ {2i / dmodel})
        # consider the frequency part in the above equation: 1/10000^{2i / d_{model}}
        # = 10000 ^ {-2i / d_{model}} = exp (-2i / d * log(10000)) = exp(-i / (d/2) * log (10000))
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # here means two broadcasting: x [seq_len, 1], emb [1, half_dim] => [seq_len, half_dim]
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DenoiseNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, time_dim):
        super(DenoiseNetwork, self).__init__()
        
        self.time_dim = time_dim
        self.action_dim = action_dim
        
        # encode the time
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 2),
            nn.Mish(),  # commonly used in transformer architecture
            nn.Linear(self.time_dim * 2, time_dim)
        )
        
        input_dim = state_dim + action_dim + time_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        
        self.final_layer = nn.Linear(hidden_dim, action_dim)
        
        # neural network initialization
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, time, state):
        # here x menas action
        time_emb = self.time_mlp(time)
        x = torch.cat((x, state, time_emb), dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)