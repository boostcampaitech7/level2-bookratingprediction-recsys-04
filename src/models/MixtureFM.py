import numpy as np
import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense, MLP_Base


class MixtureFM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embed_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)

        self.text_embedding = nn.Linear(args.word_dim, args.embed_dim)

        # 이미지 feature를 resnet을 통해 임베딩하는 부분
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        # in_features를 미리 저장해둠
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # resnet을 통해 임베딩된 이미지 벡터가 fm에 사용될 수 있도록 embed_dim 크기로 변환하는 부분
        self.resnet_embedding = nn.Linear(in_features, args.embed_dim)
        
        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()

        # deep network를 통해 dense feature를 학습하는 부분
        self.deep = MLP_Base(
                             input_dim=(args.embed_dim * len(self.field_dims)) + in_features+2 * args.word_dim,
                             embed_dims=args.mlp_dims,
                             batchnorm=args.batchnorm,
                             dropout=args.dropout,
                             output_layer=True
                            )


    def forward(self, x):
        user_book_vector, user_text_vector, book_text_vector, img_vector = x[0], x[1], x[2], x[3]

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector)  # (batch_size, 1)

        # sparse to dense
        user_book_embedding = self.embedding(user_book_vector)  # (batch_size, num_fields, embed_dim)

        # image to dense
        img_feature = self.resnet(img_vector)  # (batch_size, resnet.fc.in_features)
        img_feature_deep = img_feature
        img_feature_fm = self.resnet_embedding(img_feature)
        img_feature_fm = img_feature_fm.view(-1, 1, self.embed_dim)

        # text to dense
        user_text_feature = self.text_embedding(user_text_vector)  # (batch_size, embed_dim)
        user_text_feature = user_text_feature.view(-1, 1, user_text_feature.size(1))  # (batch_size, 1, embed_dim)
        item_text_feature = self.text_embedding(book_text_vector)  # (batch_size, embed_dim)
        item_text_feature = item_text_feature.view(-1, 1, item_text_feature.size(1))  # (batch_size, 1, embed_dim)

        # second-order interaction / dense feature
        dense_feature_fm = torch.cat([user_book_embedding,user_text_feature, item_text_feature, img_feature_fm], dim=1)

        second_order = self.fm(dense_feature_fm)

        output_fm = first_order.squeeze(1) + second_order

        # deep network를 통해 feature를 학습하는 부분
        dense_feature_deep = torch.cat([user_book_embedding.view(-1, len(self.field_dims) * self.embed_dim), 
                                        user_text_vector, book_text_vector,img_feature_deep], dim=1)
        output_dnn = self.deep(dense_feature_deep).squeeze(1)

        return output_fm + output_dnn