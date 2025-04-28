import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class CNNIQAnet(nn.Module):
    """CNNIQA network structure"""
    def __init__(self):
        super(CNNIQAnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNIQAMetric:
    """CNNIQA metric for No-Reference Image Quality Assessment"""
    def __init__(self, model_path=None):
        """Initialize the CNNIQA metric.
        
        Args:
            model_path: Path to the pre-trained CNNIQA model weights.
                        If None, will try to download from the original repo.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNIQAnet().to(self.device)
        
        # Define the transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model weights
        if model_path is None:
            # Default model path
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "models", "CNNIQA-LIVE")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Download weights if they don't exist
            if not os.path.exists(model_path):
                import urllib.request
                print("Downloading CNNIQA model weights...")
                url = "https://github.com/lidq92/CNNIQA/raw/master/models/CNNIQA-LIVE"
                urllib.request.urlretrieve(url, model_path)
        
        # Load the weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("CNNIQA model loaded successfully.")
        except Exception as e:
            print(f"Error loading CNNIQA model: {e}")
            print("Continuing with random weights. Results may not be accurate.")
    
    def evaluate(self, image_path):
        """Evaluate the quality of an image using CNNIQA.
        
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
                
            return score.item()
        
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
        for path in image_paths:
            score = self.evaluate(path)
            scores.append(score)
        return scores


# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate image quality using CNNIQA')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights')
    
    args = parser.parse_args()
    
    # Initialize the CNNIQA metric
    cnniqa = CNNIQAMetric(model_path=args.model)
    
    # Evaluate the image
    score = cnniqa.evaluate(args.image)
    print(f"CNNIQA quality score for {args.image}: {score}")