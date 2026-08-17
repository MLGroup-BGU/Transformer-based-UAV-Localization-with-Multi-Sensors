import torch
import torch.nn as nn
import math
from .positional_encoding import PositionalEncodingDualBasis, PositionalEncoding1D


class StaticContextEncoder(nn.Module):
    """ 
    Embeds static properties that don't change frequently over time 
    (e.g., gain, sinr_threshold, sensitivity) to avoid wasting attention capacity.
    """
    def __init__(self, static_dim, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(static_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, static_features):
        return self.proj(static_features)


class SparseDynamicEncoder(nn.Module):
    """ 
    Embeds the dynamic sensor properties (sectors 1-6). 
    """
    def __init__(self, dynamic_dim, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dynamic_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, dynamic_features):
        return self.proj(dynamic_features)


class HybridSpatiotemporalPredictor(nn.Module):
    """
    A unified autoregressive predictor that handles panel data efficiently by combining:
    1. Spatial positioning explicitly via coordinates.
    2. Static sensor parameters vs Dynamic sensor readings.
    3. Autoregressive prediction (predicting next time step).
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1, 
                 dynamic_dim=6, static_dim=3, coords_dim=3):
        super().__init__()
        
        self.d_model = d_model
        
        self.dynamic_encoder = SparseDynamicEncoder(dynamic_dim, d_model)
        self.static_encoder = StaticContextEncoder(static_dim, d_model)
        self.spatial_pos_encoder = PositionalEncodingDualBasis(d_model=d_model, scale_coords=1.0, sigma=0.05)
        
        self.fusion_layer = nn.Linear(4 * d_model, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)

        self.temporal_pos_encoder = PositionalEncoding1D(d_model=d_model, max_len=100)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, dynamic_dim)

    def _prepare_embeddings(self, dynamic_features, static_features, spatial_coords):
        dyn_emb = self.dynamic_encoder(dynamic_features)
        stat_emb = self.static_encoder(static_features)
        pos_emb = self.spatial_pos_encoder(spatial_coords)
        temp_pos_emb = self.temporal_pos_encoder(torch.zeros(dynamic_features.size(0), dynamic_features.size(1), self.d_model, device=dynamic_features.device))

        combined = torch.cat([dyn_emb, stat_emb, pos_emb, temp_pos_emb], dim=-1)
        x = self.fusion_layer(combined)
        return self.layer_norm_in(x)

    def forward(self, dynamic_features, static_features, spatial_coords):
        x = self._prepare_embeddings(dynamic_features, static_features, spatial_coords)

        tgt = torch.zeros_like(x)
        tgt[:, 1:] = x[:, :-1]
        
        T = tgt.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
        
        out = self.transformer_decoder(tgt, tgt, tgt_mask=mask, memory_mask=mask)
        out = self.fc_out(out)
        return out

    def get_hidden_state(self, dynamic_features, static_features, spatial_coords, layer_idx=-1):
        x = self._prepare_embeddings(dynamic_features, static_features, spatial_coords)
        
        if layer_idx == 'embedding':
            return x
            
        tgt = torch.zeros_like(x)
        tgt[:, 1:] = x[:, :-1]
        
        T = tgt.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
        
        num_layers = len(self.transformer_decoder.layers)
        
        if layer_idx == -1 or layer_idx == num_layers - 1:
            return self.transformer_decoder(tgt, tgt, tgt_mask=mask, memory_mask=mask)
            
        if not isinstance(layer_idx, int) or not (0 <= layer_idx < num_layers):
            raise ValueError(f"layer_idx must be 'embedding', -1, or between 0 and {num_layers - 1}")
            
        output = tgt
        for i, layer_module in enumerate(self.transformer_decoder.layers):
            output = layer_module(output, tgt, tgt_mask=mask, memory_mask=mask)
            if i == layer_idx:
                return output

    def get_hidden_states_firstNlast(self, dynamic_features, static_features, spatial_coords):
        x = self._prepare_embeddings(dynamic_features, static_features, spatial_coords)
        f = x
            
        tgt = torch.zeros_like(x)
        tgt[:, 1:] = x[:, :-1]
        
        T = tgt.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
        
        output = self.transformer_decoder(tgt, tgt, tgt_mask=mask, memory_mask=mask)
        output = torch.cat((f, output), dim=2)
        return output
