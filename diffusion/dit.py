# Here, we implement the core part of DiT
# DiT Block adaLN-Zero

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp

def modulate(x, shift, scale):
    # shift and scale can both be initialized to zero
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DitBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        # hidden_size denotes the dimension of input tokens
        # mlp_ratio denotes the number of improvements in the PointWiseFeedForward
        super(DitBlock, self).__init__()
        # cancel the innate scale and shift in Layernorm
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approximate the GELU function for speed 
        # GELU(x) \approx 0.5x(1 + tanh(\sqrt{2/\pi} (x + 0.044715x^3)))
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.point_wise_ffd = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu)
        
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, input_token, condition):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(condition).chunk(6, dim=1)
        x = input_token + self.attn(modulate(self.norm1(x), gamma1, beta1)) * alpha1.unsqueeze(1)
        x = x + self.point_wise_ffd(modulate(self.norm2(x), gamma2, beta2)) * alpha2.unsqueeze(1)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, output_channels):
        super(FinalLayer, self).__init__()
        
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size*patch_size*output_channels, bias=True)
        
        # final layer still needs to be tuned
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
    def forward(self, x, condition):
        shift, scale = self.adaLN(condition).chunk(2, dim=1)
        x = self.linear(modulate(self.norm_final(x), shift, scale))
        return x