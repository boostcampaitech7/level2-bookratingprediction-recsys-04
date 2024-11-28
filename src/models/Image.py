import torch
import torch.nn as nn

from ._helpers import Base_Image_DeepFM

class VGGNet(Base_Image_DeepFM):
    def __init__(self, args, data):
        # VGG16 모델 로드
        vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
        in_features = vgg.classifier[-1].in_features
        vgg.classifier = nn.Identity()
        super().__init__(args, data, image_model=vgg, image_feature_dim=25088)

class ResNet(Base_Image_DeepFM):
    def __init__(self, args, data):
        # ResNet18 모델 로드
        resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        super().__init__(args, data, image_model=resnet, image_feature_dim=in_features)
