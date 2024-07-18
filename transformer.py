import torch
from torch import nn
import torch.functional as F
import math
from attention import MaskedMultiHeadAttention
from embedding import TransformerEmbedding
from utils import LayerNorm, PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_p) -> None:
        super(EncoderLayer, self).__init__()
        
        self.attention = MaskedMultiHeadAttention(d_model=d_model, n_heads=n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_p)
        
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, dropout=drop_p)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_p)
    
    def forward(self, x):
        # generate a backoff for resnet connection
        res_X = x
        
        x = self.attention(x, x, x, mask=False)
        x = self.drop1(x)
        x = self.norm1(x + res_X)
        
        res_X = x
        x = self.ffn(x)
        
        x = self.drop2(x)
        x = self.norm2(x + res_X)
        return x
        
if __name__ == '__main__':
    X = torch.randn(128, 64, 512)   # batch, time, dimension

    d_model = 512   # the dimension of mapping to QKV space
    maxlen = 512     # multi-head numbers
    encoder = EncoderLayer(d_model=512, ffn_hidden=256, n_head=2, drop_p=0.1)
    output = encoder(X)
    print(output, output.shape)