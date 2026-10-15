import torch
import torch.nn as nn
import math
import numpy as np


class PositionalEncoding1D(nn.Module):
    """
    Standard sinusoidal positional encoding for temporal sequences.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].shape[1]]
            
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: [Batch, Seq_Len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionalEncoding3D(nn.Module):
    """
    Discrete Sinusoidal Positional Encoding for 3D coordinates.
    Concatenates PE(x), PE(y), PE(z) to form a unified spatial signature.
    """
    def __init__(self, d_model, scale_coords=100.0):
        super().__init__()
        self.d_model = d_model
        self.scale_coords = scale_coords
        
        self.d_x = d_model // 3
        self.d_y = d_model // 3
        self.d_z = d_model - 2 * self.d_x
        
        div_x = torch.exp(torch.arange(0, self.d_x, 2).float() * -(math.log(10000.0) / self.d_x))
        div_y = torch.exp(torch.arange(0, self.d_y, 2).float() * -(math.log(10000.0) / self.d_y))
        div_z = torch.exp(torch.arange(0, self.d_z, 2).float() * -(math.log(10000.0) / self.d_z))
        
        self.register_buffer('div_x', div_x)
        self.register_buffer('div_y', div_y)
        self.register_buffer('div_z', div_z)

    def forward(self, coords):
        """
        Args:
            coords: Tensor of shape (Batch, Seq_Len, 3) containing (x, y, z)
        """
        coords = coords * self.scale_coords
        
        x_pos = coords[..., 0:1]
        y_pos = coords[..., 1:2]
        z_pos = coords[..., 2:3]

        pe_x = torch.zeros(*x_pos.shape[:-1], self.d_x, device=coords.device)
        pe_x[..., 0::2] = torch.sin(x_pos * self.div_x)
        pe_x[..., 1::2] = torch.cos(x_pos * self.div_x)

        pe_y = torch.zeros(*y_pos.shape[:-1], self.d_y, device=coords.device)
        pe_y[..., 0::2] = torch.sin(y_pos * self.div_y)
        pe_y[..., 1::2] = torch.cos(y_pos * self.div_y)

        pe_z = torch.zeros(*z_pos.shape[:-1], self.d_z, device=coords.device)
        pe_z[..., 0::2] = torch.sin(z_pos * self.div_z)
        pe_z[..., 1::2] = torch.cos(z_pos * self.div_z)

        return torch.cat([pe_x, pe_y, pe_z], dim=-1)


class PositionalEncoding3DRFF(nn.Module):
    """
    Gaussian Random Fourier Feature (RFF) Encoding for 3D coordinates.
    """
    def __init__(self, d_model, scale_coords=100.0, sigma=1.0):
        super().__init__()
        self.d_model = d_model
        self.scale_coords = scale_coords
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for Fourier Features (got {d_model})")

        num_frequencies = d_model // 2
        B_matrix = torch.randn(3, num_frequencies) * sigma
        self.register_buffer("B", B_matrix)
        self.register_buffer("two_pi", torch.tensor(2 * np.pi))

    def forward(self, coords):
        x = coords * self.scale_coords
        projected = x @ self.B
        projected = self.two_pi * projected
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class PositionalEncodingDualBasis(nn.Module):
    """
    Encodes position using TWO coordinate systems simultaneously:
    1. Cartesian (x, y, z)
    2. Spherical (r, theta, phi)
    """
    def __init__(self, d_model, scale_coords=1.0, sigma=0.05):
        super().__init__()
        
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4 for Dual Basis (got {d_model})")
            
        self.d_model = d_model
        self.scale_coords = scale_coords
        
        self.d_cart = d_model // 2
        self.d_sphere = d_model // 2
        
        self.B_cart = torch.randn(3, self.d_cart // 2) * sigma
        self.register_buffer("B_cart_buf", self.B_cart)
        
        self.B_sphere = torch.randn(3, self.d_sphere // 2) * sigma
        self.register_buffer("B_sphere_buf", self.B_sphere)

        self.register_buffer("two_pi", torch.tensor(2 * np.pi))

    def _cartesian_to_spherical(self, coords):
        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]
        
        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
        theta = torch.atan2(y, x)
        phi = torch.acos(torch.clamp(z / r, -1.0, 1.0))
        
        return torch.stack([r, theta, phi], dim=-1)

    def forward(self, coords):
        cart_input = coords * self.scale_coords
        sphere_input = self._cartesian_to_spherical(coords)
        sphere_input = sphere_input * self.scale_coords

        proj_cart = (cart_input @ self.B_cart_buf) * self.two_pi
        emb_cart = torch.cat([torch.sin(proj_cart), torch.cos(proj_cart)], dim=-1)

        proj_sphere = (sphere_input @ self.B_sphere_buf) * self.two_pi
        emb_sphere = torch.cat([torch.sin(proj_sphere), torch.cos(proj_sphere)], dim=-1)

        return torch.cat([emb_cart, emb_sphere], dim=-1)


class PositionalEncodingLearnable(nn.Module):
    """
    Learnable Fourier Features for 3D coordinates.
    """
    def __init__(self, d_model, scale_coords=100.0, sigma=1.0):
        super().__init__()
        self.d_model = d_model
        self.scale_coords = scale_coords
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even (got {d_model})")

        num_frequencies = d_model // 2
        initial_B = torch.randn(3, num_frequencies) * sigma
        self.B = nn.Parameter(initial_B)

        self.register_buffer("two_pi", torch.tensor(2 * np.pi))

    def forward(self, coords):
        x = coords * self.scale_coords
        projected = x @ self.B
        projected = self.two_pi * projected
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
