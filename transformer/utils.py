import torch
from torch import nn
import torch.functional as F
import math

# layer norm: for all the features of every sample
# \mu <- \frac{1}{m}\sum_{i=1:m}x_i
# \sigma^2 <- \frac{1}{m}\sum_{i=1:m} (x_i - \mu)^2
# \hat{x_i} <- \frac{x_i - \mu}{\sqrt{\sigma^2 + \eps}}
# y_i <- \gamma \hat{x_i} + \beta
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10) -> None:
        super(LayerNorm, self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
        self.eps = eps
    
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        
        x_prime = (x - mu) / torch.sqrt(var + self.eps)
        output = self.gamma * x_prime + self.beta
        return output
    
# position-wise FFN 
# FFN(x) = relu(xW_1 + b1)W2 + b2
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1) -> None:
        super(PositionWiseFeedForward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
if __name__ == '__main__':
    X = torch.randn(128, 64, 512)   # batch, time, dimension

    d_model = 512   # the dimension of mapping to QKV space
    ln = LayerNorm(d_model=512)
    output = ln(X)
    print(output, output.shape)
    
    ffn = PositionWiseFeedForward(d_model=512, hidden=256)
    output = ffn(X)
    print(output, output.shape)
        
        
        