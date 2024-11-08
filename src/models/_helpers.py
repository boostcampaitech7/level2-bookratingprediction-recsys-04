import torch
import torch.nn as nn
from numpy import cumsum


# factorization을 통해 얻은 feature를 embedding 합니다.
# 사용되는 모델 : FM, CNN-FM, DCN
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims:list, embed_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = [0, *cumsum(field_dims)[:-1]]
        
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0)  # cold-start

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return self.embedding(x)  # (batch_size, num_fields, embed_dim)


class FeaturesLinear(nn.Module):
    def __init__(self, field_dims:list, output_dim:int=1, bias:bool=True):
        super().__init__()
        self.feature_dims = sum(field_dims)
        self.output_dim = output_dim
        self.offsets = [0, *cumsum(field_dims)[:-1]]

        self.fc = nn.Embedding(self.feature_dims, self.output_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty((self.output_dim,)), requires_grad=True)
    
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
            if isinstance(m, nn.Parameter):
                nn.init.constant_(m, 0)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return torch.sum(self.fc(x), dim=1) + self.bias if hasattr(self, 'bias') \
               else torch.sum(self.fc(x), dim=1)


class FMLayer_Dense(nn.Module):
    def __init__(self):
        super().__init__()

    def square(self, x:torch.Tensor):
        return torch.pow(x,2)

    def forward(self, x):
        square_of_sum = self.square(torch.sum(x, dim=1))
        sum_of_square = torch.sum(self.square(x), dim=1)
        
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)


class MLP_Base(nn.Module):
    def __init__(self, input_dim, embed_dims, 
                 batchnorm=True, dropout=0.2, output_layer=False):
        super().__init__()
        self.mlp = nn.Sequential()
        for idx, embed_dim in enumerate(embed_dims):
            self.mlp.add_module(f'linear{idx}', nn.Linear(input_dim, embed_dim))
            if batchnorm:
                self.mlp.add_module(f'batchnorm{idx}', nn.BatchNorm1d(embed_dim))
            self.mlp.add_module(f'relu{idx}', nn.ReLU())
            if dropout > 0:
                self.mlp.add_module(f'dropout{idx}', nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            self.mlp.add_module('output', nn.Linear(input_dim, 1))
        
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)


    def forward(self, x):
        return self.mlp(x)
    



class Text_DeepFM(nn.Module):
    def __init__(self, args, data):
        super(Text_DeepFM, self).__init__()
        self.field_dims = data['field_dims']
        self.embed_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)
        
        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        
        # 텍스트 벡터(베이스라인 기준 768차원)가 fm에 사용될 수 있도록 embed_dim 크기로 변환하는 부분
        # 유저와 아이템 각각에 대해 서로 다른 선형변환을 적용해도 되나, 여기서는 간편하게 하나의 선형변환을 적용합니다.
        self.text_embedding = nn.Linear(args.word_dim, args.embed_dim)

        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()

        # MLP를 통해 dense feature를 학습하는 부분
        self.mlp = MLP_Base(
                            input_dim=(len(self.field_dims) * self.embed_dim) + 2 * args.word_dim,
                            embed_dims=args.mlp_dims,
                            batchnorm=args.batchnorm,
                            dropout=args.dropout,
                            output_layer=True
                           )

    def forward(self, x):
        user_book_vector, user_text_vector, book_text_vector = x[0], x[1], x[2]  # (batch_size, num_fields), (batch_size, word_dim), (batch_size, word_dim)

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector)  # (batch_size, 1)

        # sparse to dense
        user_book_embedding = self.embedding(user_book_vector)  # (batch_size, num_fields, embed_dim)
        
        # text to dense
        user_text_feature = self.text_embedding(user_text_vector)  # (batch_size, embed_dim)
        user_text_feature = user_text_feature.view(-1, 1, user_text_feature.size(1))  # (batch_size, 1, embed_dim)
        item_text_feature = self.text_embedding(book_text_vector)  # (batch_size, embed_dim)
        item_text_feature = item_text_feature.view(-1, 1, item_text_feature.size(1))  # (batch_size, 1, embed_dim)

        # second-order interaction / dense
        dense_feature_fm = torch.cat([user_book_embedding, user_text_feature, item_text_feature], dim=1)
        second_order = self.fm(dense_feature_fm)

        output_fm = first_order.squeeze(1) + second_order

        # deep network를 통해 dense feature를 학습하는 부분
        dense_feature_deep = torch.cat([user_book_embedding.view(-1, len(self.field_dims) * self.embed_dim), 
                                        user_text_vector, book_text_vector], dim=1)
        output_dnn = self.mlp(dense_feature_deep).squeeze(1)

        return output_fm + output_dnn
    


class Base_Image_DeepFM(nn.Module):
    def __init__(self, args, data, image_model, image_feature_dim):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embed_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)

        # 텍스트 임베딩 부분 (텍스트 특성이 있을 경우에만 사용)
        self.text_embedding = nn.Linear(args.word_dim, args.embed_dim) if hasattr(args, 'word_dim') else None

        # 이미지 모델 설정
        self.image_model = image_model
        self.image_model_feature_dim = image_feature_dim

        # 이미지 임베딩을 embed_dim 크기로 변환
        self.image_embedding = nn.Linear(image_feature_dim, args.embed_dim)

        # dense feature 사이의 상호작용을 계산하는 FMLayer
        self.fm = FMLayer_Dense()

        # deep network를 통해 dense feature를 학습하는 부분
        input_dim = (self.embed_dim * len(self.field_dims)) + image_feature_dim
        if self.text_embedding:  # 텍스트 임베딩이 있으면 추가
            input_dim += 2 * args.word_dim
        self.deep = MLP_Base(
            input_dim=input_dim,
            embed_dims=args.mlp_dims,
            batchnorm=args.batchnorm,
            dropout=args.dropout,
            output_layer=True
        )

    def forward(self, x):
        user_book_vector = x[0]
        img_vector = x[1]
        user_text_vector = x[2] if self.text_embedding else None
        book_text_vector = x[3] if self.text_embedding else None

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector)  # (batch_size, 1)

        # sparse to dense
        user_book_embedding = self.embedding(user_book_vector)  # (batch_size, num_fields, embed_dim)

        # image to dense
        img_feature = self.image_model(img_vector)
        img_feature = img_feature.view(img_feature.size(0), -1)  # (batch_size, image_feature_dim)
        img_feature_deep = img_feature
        img_feature_fm = self.image_embedding(img_feature)  # (batch_size, embed_dim)
        img_feature_fm = img_feature_fm.view(-1, 1, self.embed_dim)

        # text to dense (optional)
        dense_feature_fm = [user_book_embedding, img_feature_fm]
        if self.text_embedding:
            user_text_feature = self.text_embedding(user_text_vector).view(-1, 1, self.embed_dim)
            item_text_feature = self.text_embedding(book_text_vector).view(-1, 1, self.embed_dim)
            dense_feature_fm.extend([user_text_feature, item_text_feature])

        # second-order interaction / dense feature
        dense_feature_fm = torch.cat(dense_feature_fm, dim=1)
        second_order = self.fm(dense_feature_fm)
        output_fm = first_order.squeeze(1) + second_order

        # deep network를 통해 feature를 학습하는 부분
        dense_feature_deep = [user_book_embedding.view(user_book_embedding.size(0), -1), img_feature_deep]
        if self.text_embedding:
            dense_feature_deep.extend([user_text_vector, book_text_vector])
        dense_feature_deep = torch.cat(dense_feature_deep, dim=1)
        output_dnn = self.deep(dense_feature_deep).squeeze(1)

        return output_fm + output_dnn