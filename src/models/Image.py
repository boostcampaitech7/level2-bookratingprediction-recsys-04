import torch
import torch.nn as nn
from torchvision import models

from ._helpers import Base_Image_DeepFM

class VGGNet(Base_Image_DeepFM):
    def __init__(self, args, data):
        # VGG16 모델 로드
        vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg.classifier = nn.Identity() 
        
        super().__init__(args, data, image_model=vgg, image_feature_dim=25088)
        self.image_model.vgg_embedding = nn.Linear(25088, self.embed_dim)  # 25088에서 embed_dim으로 변환

class ResNet(Base_Image_DeepFM):
    def __init__(self, args, data):
        # ResNet18 모델 로드
        resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        super().__init__(args, data, image_model=resnet, image_feature_dim=in_features)
