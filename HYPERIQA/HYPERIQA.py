import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


class HyperNet(nn.Module):
    """Hyper Network for modeling weights of target network."""
    
    def __init__(self, lda_out_channels, hyper_in_channels=112*112*3, target_size=112):
        super(HyperNet, self).__init__()
        
        self.target_size = target_size
        self.hyper_in_channels = hyper_in_channels
        
        # Feature extraction part
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # FC layers for predicting weights
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, lda_out_channels)
        
        # FC layers for predicting quality score
        self.fc3 = nn.Linear(1000, 256)
        self.fc4 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Extract features
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        
        # Predict weights for target network
        h = self.fc1(x)
        h = torch.relu(h)
        
        h = self.fc2(h)
        
        # Generate quality score
        q = self.fc3(x)
        q = torch.relu(q)
        
        q = self.fc4(q)
        
        return q


class HyperIQAMetric:
    """HyperIQA metric for No-Reference Image Quality Assessment"""
    
    def __init__(self, model_path=None, target_size=224):
        """Initialize the HyperIQA metric.
        
        Args:
            model_path: Path to the pre-trained HyperIQA model weights.
                       If None, will try to download from the original repo.
            target_size: Target image size for the model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        
        # Define model
        self.model = HyperNet(lda_out_channels=128, 
                             hyper_in_channels=target_size*target_size*3,
                             target_size=target_size).to(self.device)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Load model weights
        if model_path is None:
            # Default model path
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "models", "hyperIQA_koniq10k.pth")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Download weights if they don't exist
            if not os.path.exists(model_path):
                try:
                    import urllib.request
                    print("Downloading HyperIQA model weights...")
                    # Note: This URL might not be valid. Users might need to download weights manually.
                    url = "https://drive.google.com/file/d/1OOUmnbvpGea0LIGpIWEbOyxfWx6UCiiE/view"
                    urllib.request.urlretrieve(url, model_path)
                except Exception as e:
                    print(f"Error downloading HyperIQA weights: {e}")
                    print("Please download the weights manually from the HyperIQA GitHub repository.")
        
        # Load the weights if available
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("HyperIQA model loaded successfully.")
        except Exception as e:
            print(f"Error loading HyperIQA model: {e}")
            print("Continuing with random weights. Results may not be accurate.")
    
    def evaluate(self, image_path):
        """Evaluate the quality of an image using HyperIQA.
        
        Args:
            image_path: Path to the image file to evaluate.
            
        Returns:
            float: The quality score. Higher values indicate better quality.
        """
        try:
            # Load and transform the image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Evaluate the image
            with torch.no_grad():
                score = self.model(img_tensor)
                
            # Scale the raw score to a more interpretable range (0-100)
            # This scaling is approximate and may need adjustment
            scaled_score = (score.item() + 1) * 50  # Convert from [-1, 1] to [0, 100]
            
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
        for path in tqdm(image_paths, desc="Evaluating images with HyperIQA"):
            score = self.evaluate(path)
            scores.append(score)
        return scores


# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate image quality using HyperIQA')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights')
    parser.add_argument('--size', type=int, default=224, help='Target image size')
    
    args = parser.parse_args()
    
    # Initialize the HyperIQA metric
    hyper_iqa = HyperIQAMetric(model_path=args.model, target_size=args.size)
    
    # Evaluate the image
    score = hyper_iqa.evaluate(args.image)
    print(f"HyperIQA quality score for {args.image}: {score}")