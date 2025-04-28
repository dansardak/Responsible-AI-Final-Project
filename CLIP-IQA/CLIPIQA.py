import os
import torch
import numpy as np
from PIL import Image
import clip
from tqdm import tqdm


class CLIPIQAMetric:
    """CLIP-IQA metric for No-Reference Image Quality Assessment"""
    
    def __init__(self, clip_model="ViT-B/32"):
        """Initialize the CLIP-IQA metric.
        
        Args:
            clip_model: The CLIP model variant to use. Default is "ViT-B/32".
        """
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load CLIP model
            self.model, self.preprocess = clip.load(clip_model, device=self.device)
            print(f"Loaded CLIP model: {clip_model}")
            
            # These are example text prompts for assessing image quality
            # Based on typical IQA criteria
            self.quality_text_prompts = [
                "a high quality image",
                "a clear, sharp image",
                "a well-exposed image with good lighting",
                "a professional looking photograph",
                "an image with good contrast and color",
                "a beautiful, pristine image",
                "a low quality image",
                "a blurry, unclear image",
                "a poorly exposed image with bad lighting",
                "an amateur looking photograph",
                "an image with poor contrast and color",
                "a ugly, noisy image"
            ]
            
            # Encode text prompts
            self.encoded_text = self._encode_text_prompts()
            
        except Exception as e:
            print(f"Error initializing CLIP-IQA: {e}")
            self.model = None
    
    def _encode_text_prompts(self):
        """Encode the quality text prompts using CLIP."""
        with torch.no_grad():
            text_tokens = clip.tokenize(self.quality_text_prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def evaluate(self, image_path):
        """Evaluate the quality of an image using CLIP-IQA.
        
        Args:
            image_path: Path to the image file to evaluate.
            
        Returns:
            float: The quality score between 0 and 1. Higher values indicate better quality.
        """
        if self.model is None:
            print("CLIP-IQA model was not properly initialized.")
            return None
        
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Get image features
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity with quality prompts
                similarity = (100.0 * image_features @ self.encoded_text.T).softmax(dim=-1)
                
                # Calculate a quality score
                # First 6 prompts are positive qualities, next 6 are negative
                positive_score = similarity[0, :6].sum().item()
                negative_score = similarity[0, 6:].sum().item()
                
                # Normalize to 0-1 range
                quality_score = positive_score / (positive_score + negative_score)
                
            return quality_score
        
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
        for path in tqdm(image_paths, desc="Evaluating images with CLIP-IQA"):
            score = self.evaluate(path)
            scores.append(score)
        return scores


# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate image quality using CLIP-IQA')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default="ViT-B/32", 
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                        help='CLIP model variant to use')
    
    args = parser.parse_args()
    
    # Initialize the CLIP-IQA metric
    clip_iqa = CLIPIQAMetric(clip_model=args.model)
    
    # Evaluate the image
    score = clip_iqa.evaluate(args.image)
    print(f"CLIP-IQA quality score for {args.image}: {score}")