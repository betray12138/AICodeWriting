import torch
from torch import nn
import torch.functional as F
import math

# nn.Embedding is used to map the discrete index vector to dense vector
# vocab_size means the size of voabulary
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


# PE(pos, 2i) = sin(pos / 10000 ^ {2i / dmodel})
# PE(pos, 2i + 1) = cos(pos / 10000 ^ {2i / dmodel})
class PositionEmbedding(nn.Module):
    def __init__(self, d_model, maxlen) -> None:
        super(PositionEmbedding, self).__init__()
        
        # dimension [maxlen, d_model]
        self.encoding = torch.zeros(maxlen, d_model).requires_grad_(False)
        
        # generate the position
        # dimension [maxlen, 1]
        # pos means the index of time/seq_len while i means the index of word embedding
        pos = torch.arange(0, maxlen)
        pos = pos.float().unsqueeze(1)
        
        _2i = torch.arange(0, d_model, 2)
        
        # fill the position information
        # for example, d_model = 4
        # PE(pos, 0) = sin(pos / 10000^0)
        # PE(pos, 1) = cos(pos / 10000^0)
        # PE(pos, 2) = sin(pos / 10000^{1/2})
        # PE(pos, 3) = cos(pos / 10000^{1/2})
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        
    
    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]
        

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_p) -> None:
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)    
        self.position_embedding = PositionEmbedding(d_model=d_model, maxlen=max_len)
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(x)
        return self.dropout(token_embedding + position_embedding)

    
if __name__ == '__main__':
    X = torch.randn(128, 64, 512)   # batch, time, dimension

    d_model = 512   # the dimension of mapping to QKV space
    maxlen = 512     # multi-head numbers
    position_embedding = PositionEmbedding(d_model=512, maxlen=512)
    output = position_embedding(X)
    print(output, output.shape)
        
        
        