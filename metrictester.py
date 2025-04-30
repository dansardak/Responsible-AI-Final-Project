import torch
import torch.nn as nn
import cv2
import os
from torchvision.models import resnet18
from niqatest.MetaIQA.model import MetaIQA, BaselineModel1
from niqatest.RankIQA.model import RankIQA, Vgg16
from niqatest.MetaIQA.preprocess import Preprocessor
from niqatest.RankIQA.preprocessor import Preprocessor as RankPreprocessor
from niqatest.brisque.brisque import brisque
from niqatest.niqe.niqe import niqe
from niqatest.piqe.piqe import piqe


from niqatest.hyperIQA.demo import predict_quality

base_path = r'c:/Users/karat/Desktop/responsibleai/niqatest'
img_path = os.path.join(base_path, 'images', 'demo1', 'I01_01_04.png')



score = predict_quality(img_path)
print(score)



