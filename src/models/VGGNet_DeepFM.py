import numpy as np
import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense, MLP_Base

class VGGNet_DeepFM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embed_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)

        # 이미지 feature를 VGGNet을 통해 임베딩하는 부분
        self.vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
        # in_features를 미리 저장해둠
        in_features = self.vgg.classifier[-1].in_features
        self.vgg.classifier = nn.Identity()

        # VGGNet을 통해 임베딩된 이미지 벡터를 embed_dim 크기로 변환
        self.vgg_embedding = nn.Linear(25088, self.embed_dim)  # 25088에서 embed_dim으로 변환

        
        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()

        # deep network를 통해 dense feature를 학습하는 부분
        self.deep = MLP_Base(
            input_dim=(self.embed_dim * len(self.field_dims)) + 25088,  # 25088은 img_feature_deep의 크기
            embed_dims=args.mlp_dims,
            batchnorm=args.batchnorm,
            dropout=args.dropout,
            output_layer=True
        )

    def forward(self, x):
        user_book_vector, img_vector = x[0], x[1]

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector)  # (batch_size, 1)

        # sparse to dense
        user_book_embedding = self.embedding(user_book_vector)  # (batch_size, num_fields, embed_dim)
        
        # image to dense
        img_feature = self.vgg(img_vector)  # (batch_size, 512, 7, 7)
        img_feature = img_feature.view(img_feature.size(0), -1)  # (batch_size, 25088)
        img_feature_deep = img_feature  # img_feature를 img_feature_deep로 초기화
        img_feature_fm = self.vgg_embedding(img_feature)  # (batch_size, embed_dim)
        img_feature_fm = img_feature_fm.view(-1, 1, self.embed_dim)

        # second-order interaction / dense feature
        dense_feature_fm = torch.cat([user_book_embedding, img_feature_fm], dim=1)
        second_order = self.fm(dense_feature_fm)

        output_fm = first_order.squeeze(1) + second_order

        # deep network를 통해 feature를 학습하는 부분
        dense_feature_deep = torch.cat([user_book_embedding.view(user_book_embedding.size(0), -1), img_feature_deep], dim=1)
        output_dnn = self.deep(dense_feature_deep).squeeze(1)

        return output_fm + output_dnn