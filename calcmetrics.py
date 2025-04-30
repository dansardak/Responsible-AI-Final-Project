import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision
import cv2
import os
from torchvision.models import resnet18
import argparse
from niqatest.MetaIQA.model import MetaIQA, BaselineModel1
from niqatest.RankIQA.model import RankIQA, Vgg16
from niqatest.MetaIQA.preprocess import Preprocessor
from niqatest.RankIQA.preprocessor import Preprocessor as RankPreprocessor
from niqatest.brisque.brisque import brisque
from niqatest.niqe.niqe import niqe
from niqatest.piqe.piqe import piqe
from niqatest.hyperIQA.demo import predict_quality
from niqatest.CONTRIQUE.demo_score import get_contrique_score

from niqatest.tres.models import Net # Added import for TReS model
from niqatest.tres.folders import pil_loader # Added import for TReS image loader

from PIL import Image

from torchmetrics.multimodal import CLIPImageQualityAssessment
# Create an options object that both models need
class Options:
    def __init__(self):
        self.gpu = torch.cuda.is_available()  # Use GPU if available

opt = Options()

# Use absolute paths to the model files
base_path = r'c:/Users/karat/Desktop/responsibleai/niqatest'
metaiqa_path = os.path.join(base_path, 'MetaIQA', 'metaiqa.pth')
rankiqa_path = os.path.join(base_path, 'RankIQA', 'Rank_live.caffemodel.pt')


# Configuration for TReS model
tres_config = argparse.Namespace()
tres_config.network = 'resnet50' # Assuming resnet50 based on predict.py, adjust if needed
tres_config.nheadt = 16
tres_config.num_encoder_layerst = 2
tres_config.dim_feedforwardt = 64
tres_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformations for TReS model
tres_transforms = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(size=224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
])

# Modify the model classes to use the correct paths
class MetaIQAWithPath(nn.Module):
    def __init__(self, opt, model_path):
        super().__init__()
        self.opt = opt
        self.resnet_layer = resnet18(pretrained=False)
        self.net = BaselineModel1(1, 0.5, 1000)
        state_dict = torch.load(model_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=True)
        self.gpu = opt.gpu
        self.eval()
        if self.gpu:
            self.cuda()
        self.preprocessor = Preprocessor([224, 224])  # MetaIQA preprocessor expects resize dimensions
    
    @torch.no_grad()
    def forward(self, x):
        x = self.preprocessor(x)
        if self.gpu: x = x.cuda()
        x = self.resnet_layer(x)
        x = self.net(x)
        return x

class RankIQAWithPath(nn.Module):
    def __init__(self, opt, model_path):
        super().__init__()
        self.opt = opt
        self.vgg16 = Vgg16()
        self.vgg16.load_model(model_path)
        self.gpu = opt.gpu
        self.eval()
        if self.gpu:
            self.cuda()
        self.preprocessor = RankPreprocessor(30)  # RankIQA preprocessor expects patch_num
    
    @torch.no_grad()
    def forward(self, x):
        x = self.preprocessor(x)  # a image -> patches
        if self.gpu: x = x.cuda()
        x = self.vgg16(x)
        x = torch.mean(x)
        return x

# Initialize the models with correct paths
try:
    metaiqa_model = MetaIQAWithPath(opt, metaiqa_path)
    # print("MetaIQA model loaded successfully")
except Exception as e:
    print(f"Error loading MetaIQA model: {e}")

try:
    rankiqa_model = RankIQAWithPath(opt, rankiqa_path)
    # print("RankIQA model loaded successfully")
except Exception as e:
    print(f"Error loading RankIQA model: {e}")



tres_path = os.path.join(base_path, 'tres', 'bestmodel_1_2021') # Path for TReS model

try:
    tres_model = Net(tres_config, tres_device).to(tres_device)
    # Load state dict, handling potential 'module.' prefix if saved with DataParallel
    state_dict = torch.load(tres_path, map_location=tres_device)
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    tres_model.load_state_dict(state_dict)
    tres_model.eval()
    # print("TReS model loaded successfully")
except Exception as e:
    print(f"Error loading TReS model: {e}")






def get_metrics(imgpath, base_path):
    """
    Calculate all image quality metrics for a given image
    
    Args:
        imgpath: Path to the image file
        base_path: Base path to the niqatest directory
        
    Returns:
        tuple: (niqe_score, brisque_score, piqe_score, metaiqa_score, rankiqa_score)
    """
    tstimg = cv2.imread(imgpath)
    if tstimg is None:
        raise ValueError(f"Could not load image at {imgpath}")
    
    # from torchmetrics.multimodal import NIQE, BRISQUE, PIQE

    
    # Save current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the base directory so relative paths work
        os.chdir(base_path)
        # img = Image.open(imgpath).convert('RGB')
        
        # img_tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(tres_device)


        # Calculate scores
        niqe_score = niqe(tstimg)
        brisque_score = brisque(tstimg)
        piqe_score = piqe(tstimg)

        # niqe_score = NIQE(img_tensor).item()
        # brisque_score = BRISQUE(img_tensor).item()
        # piqe_score = PIQE(img_tensor).item()

        metaiqa_score = metaiqa_model(tstimg).item()   
        rankiqa_score = rankiqa_model(tstimg).item()
        hyperiqa_score = predict_quality(imgpath)
        contrique_score = get_contrique_score(imgpath, base_path)

        # Calculate CNNIQA score
        from niqatest.CNNIQA.main import CNNIQAnet
        cnniqa_model = CNNIQAnet()
        cnniqa_model.load_state_dict(torch.load(os.path.join(base_path, 'CNNIQA', 'CNNIQA-LIVE')))
        cnniqa_model.to(tres_device) # Move model to the same device as the input
        tstimg_gray = cv2.cvtColor(tstimg, cv2.COLOR_BGR2GRAY)
        tstimg_tensor = torch.from_numpy(tstimg_gray).unsqueeze(0).unsqueeze(0).float().to(tres_device)
        cnniqa_model.eval()
        cnniqa_score = cnniqa_model(tstimg_tensor).item()

        # Calculate TReS score
        tres_img = pil_loader(imgpath)
        tres_img_tensor = tres_transforms(tres_img).to(tres_device)
        with torch.no_grad():
            tres_score, _ = tres_model(tres_img_tensor.unsqueeze(0))
        tres_score = tres_score.item()
        

        # Calculate CLIPIQA score
        clipiqa_model = CLIPImageQualityAssessment()
        clipiqa_model.to(tres_device) # Move model to the target device
        img = Image.open(imgpath).convert('RGB')
        
        img_tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(tres_device) # Move input tensor to the same device
        clipiqa_score = clipiqa_model(img_tensor).item()

        # print(f"CLIPIQA score: {clipiqa_score}")


        return  [niqe_score, brisque_score, piqe_score, metaiqa_score, rankiqa_score, hyperiqa_score, contrique_score, cnniqa_score, tres_score, clipiqa_score]
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

# Base path for the project
base_path = r'c:/Users/karat/Desktop/responsibleai/niqatest'

# Create an options object that both models need
class Options:
    def __init__(self):
        self.gpu = torch.cuda.is_available()

opt = Options()

# Initialize models (using the same code as before)
try:
    metaiqa_model = MetaIQAWithPath(opt, os.path.join(base_path, 'MetaIQA', 'metaiqa.pth'))
    # print("MetaIQA model loaded successfully")
except Exception as e:
    print(f"Error loading MetaIQA model: {e}")

try:
    rankiqa_model = RankIQAWithPath(opt, os.path.join(base_path, 'RankIQA', 'Rank_live.caffemodel.pt'))
    # print("RankIQA model loaded successfully")
except Exception as e:
    print(f"Error loading RankIQA model: {e}")

import time

def get_all_metrics():
    metrics = [['Image', 'Object', 'Region', 'NIQE', 'BRISQUE', 'PIQE', 'MetaIQA', 'RankIQA', 'HyperIQA', 'Contrique', 'CNNIQA', 'TReS', 'CLIPIQA']]
    # Get base images directory
    images_dir = os.path.join(os.path.dirname(base_path), 'images')
    
    l = len(os.listdir(images_dir))
    jk = 0
    # Loop through all directories in images folder
    for prompt_dir in os.listdir(images_dir):
        jk += 1
        print(f"Processing {jk} of {l} images")
        prompt_path = os.path.join(images_dir, prompt_dir)
        
        # Skip if not a directory
        if not os.path.isdir(prompt_path):
            continue

        start_time = time.time()
            
        # Loop through all image files in prompt directory
        for img_file in os.listdir(prompt_path):
            # Skip if not an image file
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Extract OBJECT and REGION from directory name
            # Directory format is "OBJECT in REGION"
            try:
                parts = prompt_dir.split(" in ")
                object_name = parts[0]
                region_name = parts[1]
            except:
                object_name = None 
                region_name = None
                
            img_path = os.path.join(prompt_path, img_file)
            
            try:
                # Get metrics for this image
                results = get_metrics(img_path, base_path)
                
                # Add image name and results to metrics list
                metrics.append([img_file, object_name, region_name] + results)
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                # Add failed image with None values
                metrics.append([img_file, object_name, region_name] + [None] * 10)

        end_time = time.time()
        print(f"Time taken for {jk} directory: {end_time - start_time} seconds")
                
    return metrics


metrics = get_all_metrics()

import pandas as pd
# Save metrics to CSV
df = pd.DataFrame(metrics[1:], columns=metrics[0])
df.to_csv('full_complete_metrics.csv', index=False)

# imgpath = r'c:/Users/karat/Desktop/responsibleai/niqatest/images/demo1/I01_01_04.png'

# get_metrics(imgpath, base_path)
