import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from vit import ViT, Transformer

class MAE(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embedding_dim=512, encoder_heads=8, encoder_d=64, 
                 decoder_dim=512, mask_ratio=0.75, decoder_heads=8, decoder_d=64):
        super(MAE, self).__init__()
        
        self.mask_ratio = mask_ratio
        
        self.encoder = ViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            hidden_dim = encoder_d * 4,
            embedding_dim = embedding_dim,
            num_head = encoder_heads,
            d_model = encoder_d
        )
        
        num_patches, encoder_dim = self.encoder.pos_embedding.shape[-2:]
        
        self.to_patch = self.encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*self.encoder.to_patch_embedding[1:])
        
        pixel_values_per_patch = self.encoder.to_patch_embedding[2].weight.shape[-1]
        
        # decoder
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        
        # learnable masked token
        self.mask_token = nn.Parameter(torch.randn(decoder_dim)).requires_grad(True)
        self.decoder = Transformer(embedding_dim = decoder_dim, hidden_dim=decoder_d * 4, num_head = decoder_heads, d_model = decoder_d)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)   # generate a embedding dictionary, each item is a vector covering the dim of decoder_dim
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        
    def forward(self, img):
        device = img.device
        
        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        
        # get encoder tokens 
        # token + positional_encoding : mean -> directly add; cls -> cancel the first token
        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)].to(device)
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device)
        
        # get masks
        num_masked = int(self.mask_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches).argsort(dim = -1).to(device)  # generate [batch, num_patches] and sort, return the index 
        
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        
        # get the unmasked tokens and the reconstruction targets
        unmasked_tokens = tokens[:, unmasked_indices]
        pixel_values_target = patches[:, masked_indices]     # learning targets
        
        # let the unmasked_token through encoder
        unmasked_embedding = self.enc_to_dec(self.encoder.transformer(unmasked_tokens))
        
        # ready for decoder
        unmasked_decoder_tokens = unmasked_embedding + self.decoder_pos_emb(unmasked_indices)
        masked_decoder_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        masked_decoder_tokens = masked_decoder_tokens + self.decoder_pos_emb(masked_indices)
        
        # concat for decoder
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim).to(device)
        decoder_tokens[:, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[:, masked_indices] = masked_decoder_tokens
        decoder_embeddings = self.decoder(decoder_tokens)
        
        # get results
        pred_mask_embedding = decoder_embeddings[:, masked_indices]
        pred_pixel_values = self.to_pixels(pred_mask_embedding)
        
        recon_loss = F.mse_loss(pred_pixel_values, pixel_values_target)
        return recon_loss
        