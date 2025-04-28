import torch
import torch.nn as nn
import torch.nn.functional as F

class Seg_Head(nn.Module):
    def __init__(self, hidden_dim=256, num_queries=100):
        super().__init__()
        self.num_queries = num_queries

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.class_embed = nn.Linear(hidden_dim, 1)

        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, features, mask_features):
        """
        Args:
            features (Tensor): [batch_size, hidden_dim, H, W]
            mask_features (Tensor): [batch_size, hidden_dim, H, W]

        Returns:
            class_logits (Tensor): [batch_size, num_queries, 1]
            masks (Tensor): [batch_size, num_queries, H, W]
        """

        batch_size, hidden_dim, H, W = features.shape

        src = features.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer_decoder(tgt, src)
        hs = hs.permute(1, 0, 2)

        class_logits = self.class_embed(hs)
        mask_embed = self.mask_embed(hs)

        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # Added Upsample here
        masks = F.interpolate(masks, size=(512, 512), mode='bilinear', align_corners=False)

        return class_logits, masks

