import sys
import os

# Add parent directory to sys.path to allow importing HybridEncoder and train_hybrid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HybridEncoder import HybridSpatiotemporalPredictor
from train_hybrid import estimate_num_batches, get_streaming_dataloaders, plot_losses
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import comet_ml
import numpy as np
import pandas as pd

# --- Transformer Decoder ---
class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder that takes encoder hidden states and predicts xyz coordinates.
    """
    def __init__(self, input_size, d_model=128, num_heads=8, num_layers=2, dim_feedforward=512, dropout=0.1, output_size=3):
        super().__init__()
        self.d_model = d_model
        
        # Input projection (if encoder output != d_model)
        self.input_proj = nn.Linear(input_size, d_model) if input_size != d_model else nn.Identity()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to xyz coordinates
        self.fc_out = nn.Linear(d_model, output_size)  # output_size=3 for x, y, z

    def forward(self, encoder_outputs):
        """
        Args:
            encoder_outputs: [batch, seq_len, input_size] - hidden states from encoder
        Returns:
            out: [batch, seq_len, 3] - predicted xyz coordinates
        """
        # Project to d_model if needed
        x = self.input_proj(encoder_outputs)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)  # [batch, seq_len, d_model]
        
        # Use encoder outputs as both memory and target for self-attention
        # This allows the decoder to attend to the encoder's representations
        out = self.transformer_decoder(tgt=x, memory=x)  # [batch, seq_len, d_model]
        
        # Project to xyz coordinates
        out = self.fc_out(out)  # [batch, seq_len, 3]
        
        return out


class PositionalEncoding(nn.Module):
    """
    Positional encoding module (copied from encoder for consistency)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1, output_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Project encoder output to hidden_size if they differ
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        
        # LSTM decoder layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection to xyz
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs):
        # encoder_outputs shape: [batch, seq_len, input_size]
        x = self.input_proj(encoder_outputs)
        
        # LSTM returns: output, (h_n, c_n)
        # output shape: [batch, seq_len, hidden_size]
        out, _ = self.lstm(x)
        
        # Project every timestep to xyz coordinates
        out = self.fc_out(out) # [batch, seq_len, 3]
        return out
    
def train_transformer_decoder(encoder, decoder, train_loader, test_loader, device, 
                              estimated_train_batches, estimated_test_batches, 
                              optimizer, num_epochs=10, start_epoch=0, run=None):
    """
    Training loop for the Transformer decoder.
    
    Args:
        encoder: the frozen or trainable encoder (TransformerDecoderPredictor)
        decoder: the TransformerDecoder
        train_loader: yields (x, y) pairs, x is input sequence
        test_loader: test dataloader
        device: torch.device
    """

    save_epochs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
    train_losses = []
    val_losses = []

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    decoder.train()
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        # Training phase
        total_loss = 0
        num_batches = 0
        encoder.eval()
        decoder.train()
        pbar = tqdm(train_loader, desc=f"Epoch {start_epoch+epoch+1}/{start_epoch+num_epochs} [Train]", total=estimated_train_batches,
                    unit="batch",  dynamic_ncols=True)        
        
        for x, y in pbar:            
            x = x.to(device)
            
            # Get encoder hidden states
            dynamic_features = x[:, :, -15:-9]
            spatial_coords = x[:, :, -9:-6]
            static_features = x[:, :, -6:-3]
            with torch.no_grad():
                encoder_outputs = encoder.get_hidden_state(dynamic_features, static_features, spatial_coords, layer_idx='embedding')
            
            # Decode to xyz predictions
            pred_xyz_seq = decoder(encoder_outputs)  # [batch, seq_len, 3]
            
            # Target is the xyz coordinates at the last timestep
            target_xyz = x[:, -1, -3:]  # [batch, 3] - assuming last 3 features are x, y, z
            pred_xyz = pred_xyz_seq[:, -1, :]  # [batch, 3] - prediction at last timestep
            
            # Compute loss
            loss = criterion(pred_xyz, target_xyz)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        if run:
            run.log_metric("train/epoch_avg_loss", avg_train_loss)

        print(f"Epoch {start_epoch+epoch+1}/{start_epoch+num_epochs} | Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        if test_loader is not None:
            decoder.eval()
            encoder.eval()
            test_total_loss = 0
            test_num_batches = 0
            
            pbar_val = tqdm(test_loader, desc=f"Epoch {start_epoch+epoch+1}/{start_epoch+num_epochs} [Val]", total=estimated_test_batches,
                unit="batch", dynamic_ncols=True)
            
            with torch.no_grad():
                for x, y in pbar_val:
                    x = x.to(device)
                    
                    # Get encoder hidden states
                    dynamic_features = x[:, :, -15:-9]
                    spatial_coords = x[:, :, -9:-6]
                    static_features = x[:, :, -6:-3]
                    encoder_outputs = encoder.get_hidden_state(dynamic_features, static_features, spatial_coords, layer_idx='embedding')
                    
                    # Decode to xyz predictions
                    pred_xyz_seq = decoder(encoder_outputs)  # [batch, seq_len, 3]
                    
                    # Target and prediction at last timestep
                    target_xyz = x[:, -1, -3:]  # [batch, 3]
                    pred_xyz = pred_xyz_seq[:, -1, :]  # [batch, 3]
                    
                    # Compute loss
                    loss = criterion(pred_xyz, target_xyz)
                    test_total_loss += loss.item()
                    test_num_batches += 1
            
            test_avg_loss = test_total_loss / test_num_batches
            val_losses.append(test_avg_loss)
            if run:
                run.log_metric("val/batch_loss", test_avg_loss)

            print(f"Epoch {start_epoch+epoch+1}/{start_epoch+num_epochs} | Val Loss: {test_avg_loss:.4f}")
    
        if True: #epoch+1 in save_epochs:
            save_path = f"decoder_10sensors_{start_epoch+epoch+1}_epochs_baseline_30seq.pth"
            torch.save({
                'epoch': start_epoch+epoch,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, save_path)
            print(f"💾 Saved model checkpoint: {save_path}")
    
    plot_losses(train_losses, val_losses)



# Main execution
if __name__ == "__main__":

    # Get the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..'))

    # Robust paths using the script location as reference
    train_path = os.path.join(DATA_DIR, "processed_train_normalized.csv")
    test_path = os.path.join(DATA_DIR, "processed_test_normalized.csv")
    encoder_ckpt_path = os.path.join(ROOT_DIR, 'hybrid_predictor_10sensors_1_epochs.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters (easy to change)
    seq_len = 30
    batch_size = 32
    d_model = 128
    num_heads = 8
    num_layers = 3
    dim_ff = 512
    dropout = 0.1
    num_epochs = 15
    lr = 1e-4

    print(f"Checking data files:\n  Train: {train_path}\n  Test:  {test_path}")
    for p in [train_path, test_path, encoder_ckpt_path]:
        if not os.path.exists(p):
            print(f"❌ Error: File not found: {p}")
            sys.exit(1)

    print("Estimating dataset size for progress bars...")
    estimated_train_batches = estimate_num_batches(train_path, seq_len, batch_size)
    estimated_test_batches = estimate_num_batches(test_path, seq_len, batch_size)
    
    run = comet_ml.Experiment(
        api_key="KKs6LeghdLVP5fLtr3xeVsi9N",
        project_name="uav-detection",
        workspace="maidad-bgu-msc"
    )
    run.set_name("LSTM Decoder - Linear Layer")

    run.log_parameters({
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model*2,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dim_feedforward": dim_ff,
        "dropout": dropout
    })

    # Prepare dataloaders
    train_loader, test_loader = get_streaming_dataloaders(train_path, test_path, seq_len, batch_size)
    
    # Infer input size from dataset
    input_size = pd.read_csv(train_path, nrows=1).shape[1]


    # Initialize model from pre-trained weights
    encoder = HybridSpatiotemporalPredictor(
        d_model=d_model, 
        nhead=num_heads, 
        num_layers=num_layers, 
        dim_feedforward=dim_ff, 
        dropout=dropout,
        dynamic_dim=6,
        static_dim=3,
        coords_dim=3
    )
    
    encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    print(f"✅ Loaded pretrained encoder from: {encoder_ckpt_path}")

    decoder = LSTMDecoder(
        input_size=d_model,  # Matches encoder output dimension
        hidden_size=d_model*2,
        num_layers=2,        # Standard depth for LSTM
        dropout=dropout,
        output_size=3
    )
    
    decoder.load_state_dict(torch.load("decoder_10sensors_15_epochs_baseline_30seq.pth", map_location=device), strict=False)
    decoder.to(device)
    
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    start_epoch = 16
    
    # Train the decoder
    train_transformer_decoder(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        estimated_train_batches=estimated_train_batches,
        estimated_test_batches=estimated_test_batches,
        optimizer=optimizer,
        num_epochs=num_epochs,
        start_epoch=start_epoch,
        run=run
    )

