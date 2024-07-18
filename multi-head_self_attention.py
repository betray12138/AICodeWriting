import torch
from torch import nn
import torch.functional as F
import math

X = torch.randn(128, 64, 512)   # batch, time, dimension

d_model = 512   # the dimension of mapping to QKV space
n_heads = 8     # multi-head numbers

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super(MaskedMultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Q K V matrix
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_combine = nn.Linear(d_model, d_model)    # multi-head composition mapping
        
    def forward(self, q, k, v):
        # here q, k, v means input
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_heads
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # dimension separation => to n heads
        # note that when doing attention operation, n_head dimension cannot be put in the end
        q = q.view(batch, time, self.n_heads, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_heads, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_heads, n_d).permute(0, 2, 1, 3)
        
        # compute the attention score = qk^T / \sqrt{n_d}
        # here the dimension of score is batch, n_heads, time, time
        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        
        # generate a low-triangle matrix dimension: [time, time]
        mask = torch.tril(torch.ones(time, time, dtype=bool))
        
        # generate the mask for score, -inf can be computed as 0 when doing softmax
        # for example, the time = 4, the generated score is 
        # tensor([[[[ 0.1, -inf, -inf, -inf],
        #           [ 0.5,  0.6, -inf, -inf],
        #           [ 0.9,  1.0,  1.1, -inf],
        #           [ 1.3,  1.4,  1.5,  1.6]]]])
        score = score.masked_fill(mask == 0, float("-inf"))
        
        # execute softmax, dim=-1 would execute on the row, for example
        # tensor([[[[1.0000, 0.0000, 0.0000, 0.0000],
        #            [0.4750, 0.5250, 0.0000, 0.0000],
        #            [0.3000, 0.3320, 0.3680, 0.0000],
        #            [0.2190, 0.2430, 0.2690, 0.2960]]]])
        score = torch.softmax(score, dim=-1)
        
        score = score @ v
        
        # compose again, before permute, the dimension is batch, n_heads, time, n_d
        # transform it to batch, time, dimension
        score = score.permute(0, 2, 1, 3).reshape(batch, time, dimension)
        
        output = self.w_combine(score)
        return output
    
attention = MaskedMultiHeadAttention(d_model=512, n_heads=8)
output = attention(X, X, X)
print(output, output.shape)
        
        
        