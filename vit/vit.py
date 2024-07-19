import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature=10000):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    
    position_embedding = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return position_embedding.float()

# Here, a simple transformer version is provided
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim) -> None:
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, x):
        return self.net(x)
    
# A fancy attention codes !!
class Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, d_model=512):
        super(Attention, self).__init__()
        qkv_output_dim = num_heads * d_model
        self.num_heads = num_heads
        self.scale = d_model ** -0.5    # used to normalization the qk^T
        self.norm = nn.LayerNorm(embedding_dim)
    
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(embedding_dim, qkv_output_dim * 3, bias=False)  # chunk can divide the 3 parts
        
        self.to_out = nn.Linear(qkv_output_dim, embedding_dim, bias=False)
        
    def forward(self, x):
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        # [batch, seq, num_heads, feature] -> [batch num_heads seq feature]
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h = self.num_heads), qkv)
        
        attend_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attend_score = self.attend(attend_score)
        
        out = torch.matmul(attend_score, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_head=8, d_model=512):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.layers = nn.ModuleList([])
        for _ in range(2):
            self.layers.append(nn.ModuleList([
                Attention(embedding_dim, num_heads=num_head, d_model=d_model),
                FeedForward(embedding_dim, hidden_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, hidden_dim, embedding_dim=512, num_head=8, d_model=512):
        super(ViT, self).__init__()
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        patch_dim = patch_height * patch_width * 3
        
        # patch the image [batch, channel, (patch_h_num patch_height), (patch_w_num patch_width)] -> [batch, (patch_h_num patch_w_num), (patch_h_num patch_w_num channel)]
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # construct the positional_embedding
        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = embedding_dim
        )
        
        # mean denote mean over patches, cls denote an extra token
        self.pool = "mean"
        
        self.transformer = Transformer(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_head=num_head, d_model=d_model)
        self.linear_head = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, img):
        device = img.device
        
        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding.to(device)
        
        x = self.transformer(x).mean(dim = 1)   # mean over the patches
        
        x = self.linear_head(x)
        return x