import torch
import torch.nn as nn
import torch.nn.functional as F
from ._helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense, MLP_Base
from transformers import CLIPModel


# DNN과 FM을 결합한 DeepFM 모델을 구현합니다.
class DeepFM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.factor_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)

        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()
        
        # deep network를 통해 feature를 학습하는 부분
        self.dnn = MLP_Base(
                             input_dim=(args.embed_dim * len(self.field_dims)),
                             embed_dims=args.mlp_dims,
                             batchnorm=args.batchnorm,
                             dropout=args.dropout,
                             output_layer=True
                            )


    def forward(self, x: torch.Tensor):
        # first-order interaction / sparse feature only
        first_order = self.linear(x).squeeze(1)

        # sparse to dense
        embedding = self.embedding(x)  # (batch_size, num_fields, embed_dim)

        # second-order interaction / dense
        second_order = self.fm(embedding)

        # deep network를 통해 feature를 학습하는 부분
        deep_out = self.dnn(embedding.view(-1, embedding.size(1) * embedding.size(2))).squeeze(1)

        return first_order + second_order + deep_out


class CLIP_DeepFM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.factor_dim = args.embed_dim

        # CLIP 모델 로드 및 학습 비활성화
        self.clip_model = CLIPModel.from_pretrained(args.pretrained_model)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)

        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()
        
        # CLIP 출력과 DeepFM 임베딩을 결합하여 처리하는 MLP
        clip_text_dim = self.clip_model.config.projection_dim  # 512
        clip_image_dim = self.clip_model.config.vision_config.hidden_size  # 768
        deepfm_dim = args.embed_dim * len(self.field_dims)  # 128
        total_dim = clip_text_dim + clip_image_dim + deepfm_dim  # 1408
                
        self.dnn = MLP_Base(
            input_dim=total_dim,
            embed_dims=args.mlp_dims,
            batchnorm=args.batchnorm,
            dropout=args.dropout,
            output_layer=True
        )

    def forward(self, batch):
        """
        Parameters
        ----------
        batch : tuple or list
            [0]: user_book_vector (B x F)
            [1]: text_input dict
            [2]: image_input dict
        """
        # DeepFM 부분 처리
        x = batch[0]
        first_order = self.linear(x).squeeze(1)
        embedding = self.embedding(x)
        second_order = self.fm(embedding)
        
        # CLIP 텍스트 처리
        text_outputs = self.clip_model.text_model(
            input_ids=batch[1]['input_ids'][:, :77]  # 최대 77 토큰으로 제한
        )
        text_features = text_outputs.pooler_output  # [batch_size, hidden_size]
        
        # CLIP 이미지 처리
        image_outputs = self.clip_model.vision_model(
            pixel_values=batch[2]['pixel_values']
        )
        image_features = image_outputs.pooler_output  # [batch_size, hidden_size]
        
        # DeepFM 임베딩과 CLIP 특성들을 결합
        combined_features = torch.cat([
            embedding.view(-1, self.factor_dim * len(self.field_dims)),
            text_features,
            image_features
        ], dim=1)

        # Deep 네트워크 통과
        deep_out = self.dnn(combined_features).squeeze(1)

        # 최종 출력
        return first_order + second_order + deep_out
