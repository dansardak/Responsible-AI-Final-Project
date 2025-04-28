import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import math


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # BCHW -> BNC (batch_size, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class HashBasedSpatialEncoding(nn.Module):
    """Hash-based 2D spatial encoding for multi-scale patches"""
    
    def __init__(self, d_model, hash_size=256):
        super().__init__()
        self.hash_size = hash_size
        self.spatial_embed = nn.Embedding(hash_size * hash_size, d_model)
        
    def forward(self, batch_size, h, w):
        """Generate spatial encodings for a grid of size h x w"""
        grid_h = torch.arange(h, device=self.spatial_embed.weight.device)
        grid_w = torch.arange(w, device=self.spatial_embed.weight.device)
        
        # Create 2D grid
        grid_y, grid_x = torch.meshgrid(grid_h, grid_w)
        
        # Hash the positions to indices
        grid_y = grid_y % self.hash_size
        grid_x = grid_x % self.hash_size
        indices = grid_y * self.hash_size + grid_x
        
        # Get embeddings
        embeddings = self.spatial_embed(indices.flatten())
        
        # Reshape and expand for batch dimension
        embeddings = embeddings.reshape(1, h * w, -1).expand(batch_size, -1, -1)
        
        return embeddings


class ScaleEncoding(nn.Module):
    """Scale encoding for different image resolutions"""
    
    def __init__(self, d_model, max_scales=10):
        super().__init__()
        self.scale_embed = nn.Embedding(max_scales, d_model)
        
    def forward(self, batch_size, num_patches, scale_idx):
        """Generate scale encodings for a given scale index"""
        scale_emb = self.scale_embed(torch.tensor([scale_idx], device=self.scale_embed.weight.device))
        return scale_emb.expand(batch_size, num_patches, -1)


class MultiScaleTransformer(nn.Module):
    """Multi-scale Image Quality Transformer (MUSIQ)"""
    
    def __init__(self, patch_size=16, num_scales=3, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4, hash_size=256, max_scales=10):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_scales = num_scales
        self.embed_dim = embed_dim
        
        # Patch embeddings for different scales
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, 
            in_chans=3, 
            embed_dim=embed_dim
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional encodings
        self.hash_spatial_enc = HashBasedSpatialEncoding(
            d_model=embed_dim,
            hash_size=hash_size
        )
        
        self.scale_enc = ScaleEncoding(
            d_model=embed_dim,
            max_scales=max_scales
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_ratio * embed_dim,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth
        )
        
        # MLP head for quality prediction
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, std=0.02)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward_single_scale(self, x, scale_idx):
        """Process a single scale of the input image"""
        B = x.shape[0]
        
        # Get patch embeddings
        x = self.patch_embed(x)  # [B, N, C]
        
        # Get spatial positions
        h = w = int(math.sqrt(x.size(1)))
        spatial_pos = self.hash_spatial_enc(B, h, w)
        
        # Get scale positions
        scale_pos = self.scale_enc(B, x.size(1), scale_idx)
        
        # Add positional encodings
        x = x + spatial_pos + scale_pos
        
        return x, h * w
    
    def forward(self, xs):
        """Forward pass of MUSIQ model
        
        Args:
            xs: List of image tensors at different scales
                Each tensor has shape [B, C, H, W]
        """
        B = xs[0].shape[0]
        features = []
        total_patches = 0
        
        # Process each scale
        for i, x in enumerate(xs):
            feat, num_patches = self.forward_single_scale(x, i)
            features.append(feat)
            total_patches += num_patches
        
        # Concatenate features from all scales
        x = torch.cat(features, dim=1)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Use CLS token for final prediction
        x = x[:, 0]
        
        # MLP head
        quality_score = self.mlp_head(x)
        
        return quality_score


class MUSIQMetric:
    """MUSIQ metric for multi-scale image quality assessment"""
    
    def __init__(self, model_path=None, scales=(224, 384, 512), patch_size=16):
        """Initialize the MUSIQ metric.
        
        Args:
            model_path: Path to pre-trained MUSIQ model weights.
                       If None, will use random initialization.
            scales: Tuple of scales to use for multi-scale processing.
            patch_size: Size of patches for the transformer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scales = scales
        self.patch_size = patch_size
        
        # Initialize model
        self.model = MultiScaleTransformer(
            patch_size=patch_size,
            num_scales=len(scales),
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4
        ).to(self.device)
        
        # Define transforms for each scale
        self.transforms = [
            transforms.Compose([
                transforms.Resize((scale, scale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]) for scale in scales
        ]
        
        # Load model weights
        if model_path is not None:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("MUSIQ model loaded successfully.")
            except Exception as e:
                print(f"Error loading MUSIQ model: {e}")
                print("Continuing with random weights. Results may not be accurate.")
        else:
            print("No pre-trained weights provided for MUSIQ. Using random initialization.")
            print("For better results, download weights from Google Research's MUSIQ repository.")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate(self, image_path):
        """Evaluate the quality of an image using MUSIQ.
        
        Args:
            image_path: Path to the image file to evaluate.
            
        Returns:
            float: The quality score between 0 and 100. Higher values indicate better quality.
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Process at multiple scales
            multi_scale_inputs = []
            for transform in self.transforms:
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                multi_scale_inputs.append(img_tensor)
            
            # Evaluate the image
            with torch.no_grad():
                score = self.model(multi_scale_inputs)
                
            # MUSIQ typically outputs a score between -1 and 1, scale to 0-100
            scaled_score = (score.item() + 1) * 50
            scaled_score = max(0, min(100, scaled_score))  # Clamp to [0, 100]
            
            return scaled_score
        
        except Exception as e:
            print(f"Error evaluating image {image_path}: {e}")
            return None
    
    def evaluate_batch(self, image_paths):
        """Evaluate multiple images at once.
        
        Args:
            image_paths: List of paths to image files.
            
        Returns:
            list: List of quality scores for each image.
        """
        scores = []
        for path in tqdm(image_paths, desc="Evaluating images with MUSIQ"):
            score = self.evaluate(path)
            scores.append(score)
        return scores


# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate image quality using MUSIQ')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights')
    parser.add_argument('--scales', nargs='+', type=int, default=[224, 384, 512],
                        help='Scales to use for multi-scale processing')
    
    args = parser.parse_args()
    
    # Initialize the MUSIQ metric
    musiq = MUSIQMetric(model_path=args.model, scales=tuple(args.scales))
    
    # Evaluate the image
    score = musiq.evaluate(args.image)
    print(f"MUSIQ quality score for {args.image}: {score}")