# LORA means low-rank fine-tuning
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LORALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, lora_alpha=16, dropout=0.5) -> None:
        # merge denotes whether using the pre-trained parameters
        # lora_alpha means the weight for loraLinear to combine the pre-trained weights
        super(LORALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.dropout_rate = dropout
        self.lora_alpha = lora_alpha
        
        self.linear = nn.Linear(in_features, out_features)
        
        # in_features * out_features => in_features * r, r * out_features
        # ! Caution, the dimension of nn.Parameter is converse w.r.t the setting of Linear
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
        self.scale = self.lora_alpha / self.rank
        self.linear.weight.requires_grad = False    # linear means the pre-trained weights
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()
            
        self.initial_weights()
        
    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        
    def forward(self, x):
        # note that, the dimension of self.linear.weight is [out_features, in_features]
        # construct the new parameter structures!
        outputs = F.linear(x, self.linear.weight + self.lora_b @ self.lora_a * self.scale, self.linear.bias)
        outputs = self.dropout(outputs)
        return outputs