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
    
    def forward(self, x, mask=None):
        # generate a backoff for resnet connection
        res_X = x
        
        x = self.attention(x, x, x, mask)
        x = self.drop1(x)
        x = self.norm1(x + res_X)
        
        res_X = x
        x = self.ffn(x)
        
        x = self.drop2(x)
        x = self.norm2(x + res_X)
        return x
    
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_p):
        super(DecoderLayer, self).__init__()
        self.attention1 = MaskedMultiHeadAttention(d_model=d_model, n_heads=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(drop_p)
        
        self.cross_attention = MaskedMultiHeadAttention(d_model=d_model, n_heads=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(drop_p)
        
        self.ffn = PositionWiseFeedForward(d_model=d_model, ffn_hidden=ffn_hidden, dropout=drop_p)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(drop_p)
        
    def forward(self, dec, enc, t_mask, s_mask):
        res_X = dec
        x = self.attention1(dec, dec, dec, t_mask)  # low-triangle mask
        
        x = self.dropout1(x)
        x = self.norm1(x + res_X)
        
        if enc is not None:
            # do cross attention
            res_X = x
            x = self.cross_attention(x, enc, enc, s_mask)   # do not focus on the padding information
            
            x = self.dropout2(x)
            x = self.norm2(x + res_X)
        
        res_X = x
        x = self.ffn(x)
        
        x = self.dropout3(x)
        x = self.norm3(x + res_X)
    
        return x
    
class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_p) -> None:
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=enc_vocab_size, d_model=d_model, max_len=max_len, drop_p=drop_p)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_p=drop_p) for _ in range(n_layer)]
        )
        
    def forward(self, x, s_mask):
        # s_mask is used to prevent focusing on padding area
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_p) -> None:
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=dec_vocab_size, d_model=d_model, max_len=max_len, drop_p=drop_p)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_p=drop_p) for _ in range(n_layer)]
        )
        self.fc = nn.Linear(d_model, dec_vocab_size)
        
    def forward(self, dec, enc, t_mask, s_mask):
        # s_mask is used to prevent focusing on padding area
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        dec = self.fc(dec)
        return dec
    
class Transformer(nn.Module):
    def __init__(self, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 enc_voc_size, 
                 dec_voc_size, 
                 max_len, 
                 d_model, 
                 n_heads, 
                 ffn_hidden, 
                 n_layer, 
                 drop_p, 
                 device) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layer, drop_p).to(device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layer, drop_p).to(device)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_t_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        
        mask = torch.tril(torch.ones(len_q, len_k, dtype=bool)).to(self.device)
        return mask
    
    def make_s_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        
        # the generated mask : batch, time, len_q, len_k
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        mask = q & k
        return mask
    
    def forward(self, src, trg):
        # encoder mask
        src_mask = self.make_s_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        # decoder mask
        trg_mask = self.make_s_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_t_mask(trg, trg)
        # cross mask
        src_trg_mask = self.make_s_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)
        
        enc = self.encoder(src, src_mask)
        output = self.decoder(trg, enc, trg_mask, src_trg_mask)
        
        return output
        
        
        
if __name__ == '__main__':
    X = torch.randn(128, 64, 512)   # batch, time, dimension

    d_model = 512   # the dimension of mapping to QKV space
    maxlen = 512     # multi-head numbers
    encoder = EncoderLayer(d_model=512, ffn_hidden=256, n_head=2, drop_p=0.1)
    output = encoder(X)
    print(output, output.shape)