import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding1D(nn.Module):
    """
    Standard sinusoidal positional encoding for temporal sequences.
    (Your original logic, slightly refactored)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Handle odd d_model sizes just in case
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[: , :pe[:, 1::2].shape[1]]
            
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
        self.scale_coords = scale_coords # Multiplier to ensure coords aren't too small for sin/cos
        
        # Calculate sub-dimensions
        self.d_x = d_model // 3
        self.d_y = d_model // 3
        self.d_z = d_model - 2 * self.d_x  # Assign remainder to Z axis
        
        # Pre-compute denominators for each axis
        # We use a smaller base (e.g. 100.0) if coordinates are normalized/small
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
        Returns:
            Tensor of shape (Batch, Seq_Len, d_model) representing spatial encoding
        """
        # Scale coords (important if inputs are normalized 0-1)
        coords = coords * self.scale_coords
        
        x_pos = coords[..., 0:1]
        y_pos = coords[..., 1:2]
        z_pos = coords[..., 2:3]

        # --- X Embedding ---
        pe_x = torch.zeros(*x_pos.shape[:-1], self.d_x, device=coords.device)
        pe_x[..., 0::2] = torch.sin(x_pos * self.div_x)
        pe_x[..., 1::2] = torch.cos(x_pos * self.div_x)

        # --- Y Embedding ---
        pe_y = torch.zeros(*y_pos.shape[:-1], self.d_y, device=coords.device)
        pe_y[..., 0::2] = torch.sin(y_pos * self.div_y)
        pe_y[..., 1::2] = torch.cos(y_pos * self.div_y)

        # --- Z Embedding ---
        pe_z = torch.zeros(*z_pos.shape[:-1], self.d_z, device=coords.device)
        pe_z[..., 0::2] = torch.sin(z_pos * self.div_z)
        pe_z[..., 1::2] = torch.cos(z_pos * self.div_z)

        # Concatenate along the feature dimension
        return torch.cat([pe_x, pe_y, pe_z], dim=-1)

class PositionalEncoding3DRFF(nn.Module):
    """
    Gaussian Random Fourier Feature (RFF) Encoding for 3D coordinates.
    
    This replaces the axis-aligned sinusoidal encoding with an isotropic
    encoding. The dot product between two embeddings approximates the 
    Euclidean distance (Gaussian RBF Kernel) between the physical points.
    """
    def __init__(self, d_model, scale_coords=100.0, sigma=1.0):
        """
        Args:
            d_model: Output embedding dimension (must be even).
            scale_coords: Multiplier for inputs before projection (e.g. to convert 
                          meters to cm or handle normalized data).
            sigma: Standard deviation of the Gaussian distribution.
                   - Lower sigma = "Blurry" positional encoding (focus on global structure).
                   - Higher sigma = "High res" encoding (focus on fine detail).
        """
        super().__init__()
        self.d_model = d_model
        self.scale_coords = scale_coords
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for Fourier Features (got {d_model})")

        # We need d_model/2 frequency vectors because we will output sin() and cos() for each.
        num_frequencies = d_model // 2
        
        # 1. Sample the random projection matrix B from a Normal Distribution
        # Shape: [3, num_frequencies]
        # We multiply by sigma^2 to control the spread of frequencies
        B_matrix = torch.randn(3, num_frequencies) * sigma
        
        # 2. Register B as a buffer. 
        # Buffers are part of the state_dict (saved with the model)
        # but are NOT updated by the optimizer (no gradients).
        self.register_buffer("B", B_matrix)

        # Constant for 2*pi is helpful
        self.register_buffer("two_pi", torch.tensor(2 * np.pi))

    def forward(self, coords):
        """
        Args:
            coords: [Batch, Seq_Len, 3]
        Returns:
            [Batch, Seq_Len, d_model]
        """
        # 1. Scale coordinates 
        # (Crucial if your inputs are small, e.g., 0.001 ranges)
        x = coords * self.scale_coords

        # 2. Projection: x @ B 
        # Shapes: [B, T, 3] @ [3, d_model/2] -> [B, T, d_model/2]
        projected = x @ self.B

        # 3. Scale by 2pi
        projected = self.two_pi * projected

        # 4. Compute Sin and Cos
        # Output shape: [B, T, d_model]
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
    
import torch
import torch.nn as nn
import numpy as np

class PositionalEncodingDualBasis(nn.Module):
    """
    Encodes position using TWO coordinate systems simultaneously:
    1. Cartesian (x, y, z) - Good for linear translation logic.
    2. Spherical (r, theta, phi) - Good for sensor FOV and rotation logic.
    
    The output embedding is a concatenation of both encodings.
    """
    def __init__(self, d_model, scale_coords=1.0, sigma=0.05):
        super().__init__()
        
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4 for Dual Basis (got {d_model})")
            
        self.d_model = d_model
        self.scale_coords = scale_coords
        
        # Split dimensions: half for Cartesian, half for Spherical
        self.d_cart = d_model // 2
        self.d_sphere = d_model // 2
        
        # --- Cartesian Basis (RFF) ---
        # Input: 3 dims (x,y,z), Output: d_cart
        # We need d_cart // 2 frequencies for sin/cos pairs
        self.B_cart = torch.randn(3, self.d_cart // 2) * sigma
        self.register_buffer("B_cart_buf", self.B_cart)
        
        # --- Spherical Basis (RFF) ---
        # Input: 3 dims (r, theta, phi), Output: d_sphere
        # We need d_sphere // 2 frequencies
        # Note: We might want different sigma for angles vs range, 
        # but sharing sigma is usually fine for initial testing.
        self.B_sphere = torch.randn(3, self.d_sphere // 2) * sigma
        self.register_buffer("B_sphere_buf", self.B_sphere)

        self.register_buffer("two_pi", torch.tensor(2 * np.pi))

    def _cartesian_to_spherical(self, coords):
        """
        Converts (x, y, z) to (r, theta, phi).
        """
        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]
        
        # Radius (r)
        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
        
        # Azimuth (theta) [-pi, pi]
        theta = torch.atan2(y, x)
        
        # Elevation (phi) [0, pi]
        phi = torch.acos(torch.clamp(z / r, -1.0, 1.0))
        
        return torch.stack([r, theta, phi], dim=-1)

    def forward(self, coords):
        """
        Args:
            coords: [Batch, Seq_Len, 3] (x, y, z)
        """
        # 1. Prepare Inputs
        cart_input = coords * self.scale_coords
        sphere_input = self._cartesian_to_spherical(coords)
        
        # Scale spherical coords? 
        # r is already scaled by x,y,z scale implicitly. 
        # theta/phi are radians (-3.14 to 3.14). 
        # We multiply angles by scale_coords to match the frequency domain of B_sphere.
        sphere_input = sphere_input * self.scale_coords

        # 2. Cartesian Projection
        proj_cart = (cart_input @ self.B_cart_buf) * self.two_pi
        emb_cart = torch.cat([torch.sin(proj_cart), torch.cos(proj_cart)], dim=-1)

        # 3. Spherical Projection
        proj_sphere = (sphere_input @ self.B_sphere_buf) * self.two_pi
        emb_sphere = torch.cat([torch.sin(proj_sphere), torch.cos(proj_sphere)], dim=-1)

        # 4. Concatenate
        return torch.cat([emb_cart, emb_sphere], dim=-1)
    
class PositionalEncodingLearnable(nn.Module):
    """
    Learnable Fourier Features.
    
    Starts as random Gaussian noise (isotropic), but the projection matrix 'B'
    is a trainable Parameter. The network learns which spatial directions 
    are most important for the task.
    """
    def __init__(self, d_model, scale_coords=100.0, sigma=1.0):
        super().__init__()
        self.d_model = d_model
        self.scale_coords = scale_coords
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even (got {d_model})")

        num_frequencies = d_model // 2
        
        # Initialize B with Gaussian distribution (same as RFF)
        initial_B = torch.randn(3, num_frequencies) * sigma
        
        # WRAP IN NN.PARAMETER to make it learnable
        self.B = nn.Parameter(initial_B)

        self.register_buffer("two_pi", torch.tensor(2 * np.pi))

    def forward(self, coords):
        # 1. Scale coordinates
        x = coords * self.scale_coords

        # 2. Projection (Using the LEARNABLE B)
        projected = x @ self.B

        # 3. Scale by 2pi
        projected = self.two_pi * projected

        # 4. Compute Sin and Cos
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)