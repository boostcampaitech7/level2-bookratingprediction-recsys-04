import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor
from .basic_data import basic_data_split
from .text_data import text_preprocessing
from PIL import Image

class CLIPDataset(Dataset):
    def __init__(self, user_book_vector, text_data, image_paths, processor, rating=None):
        """
        Parameters
        ----------
        user_book_vector : np.ndarray
            모델 학습에 사용할 유저 및 책 정보(범주형 데이터)
        text_data : list
            텍스트 데이터 (유저 리뷰 + 책 요약)
        image_paths : list
            이미지 파일 경로
        processor : CLIPProcessor
            CLIP 전처리기
        rating : np.ndarray, optional
            정답 데이터
        """
        self.user_book_vector = user_book_vector
        self.text_data = text_data
        self.image_paths = image_paths
        self.processor = processor
        self.rating = rating

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):

        # 이미지 로드 및 전처리
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # resize
        image = image.resize((224, 224))  # CLIP의 기본 이미지 크기
        
        # 메모리 최적화를 위해 with torch.no_grad() 추가
        with torch.no_grad():
            image_input = self.processor(
                images=image,
                text=None,  # 텍스트는 따로 처리하므로 None
                return_tensors='pt',
                padding=True
            )['pixel_values']
            
            text_input = self.processor(
                text=self.text_data[idx], 
                images=None,  # 이미지는 따로 처리했으므로 None
                return_tensors='pt', 
                padding='max_length',
                max_length=256,
                truncation=True
            )

        if self.rating is not None:
            return {
                'user_book_vector': torch.tensor(self.user_book_vector[idx], dtype=torch.long),
                'text_input': text_input['input_ids'].squeeze(0),
                'image_input': image_input.squeeze(0),
                'rating': torch.tensor(self.rating[idx], dtype=torch.float32)
            }
        else:
            return {
                'user_book_vector': torch.tensor(self.user_book_vector[idx], dtype=torch.long),
                'text_input': text_input['input_ids'].squeeze(0),
                'image_input': image_input.squeeze(0),
            }

def combined_data_load(args):
    """데이터를 로드하고 전처리합니다."""
    # 기본 데이터 로드
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    # CLIP 프로세서만 초기화 (모델은 제외)
    processor = CLIPProcessor.from_pretrained(args.model_args[args.model].pretrained_model)

    # config에서 특성 정의 가져오기
    user_features = args.dataset.features.user
    book_features = args.dataset.features.book
    sparse_cols = ['user_id', 'isbn'] + list(set(user_features + book_features))

    # 텍스트 전처리
    books['summary'] = text_preprocessing(books['summary'])
    
    # 데이터프레임 병합
    train_df = train.merge(books[['isbn', 'summary', 'img_path'] + book_features], on='isbn', how='left')\
                    .merge(users[['user_id'] + user_features], on='user_id', how='left')
    test_df = test.merge(books[['isbn', 'summary', 'img_path'] + book_features], on='isbn', how='left')\
                  .merge(users[['user_id'] + user_features], on='user_id', how='left')
    
    all_df = pd.concat([train_df, test_df], axis=0)

    # 라벨 인코딩
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = train_df[col].map(label2idx[col])
        test_df[col] = test_df[col].map(label2idx[col])

    field_dims = [len(label2idx[col]) for col in sparse_cols]

    return {
        'train': train_df,
        'test': test_df,
        'field_names': sparse_cols,
        'field_dims': field_dims,
        'label2idx': label2idx,
        'idx2label': idx2label,
        'processor': processor,
        'sub': sub
    }

def combined_data_split(args, data):
    """학습 데이터를 학습/검증 데이터로 나눕니다."""
    return basic_data_split(args, data)

def combined_data_loader(args, data):
    """데이터로더를 생성합니다."""
    # 이미지 경로 앞에 'data/'를 추가
    train_image_paths = ['data/' + path for path in data['X_train']['img_path'].values]
    valid_image_paths = ['data/' + path for path in data['X_valid']['img_path'].values]
    test_image_paths = ['data/' + path for path in data['test']['img_path'].values]

    train_dataset = CLIPDataset(
        data['X_train'][data['field_names']].values,
        data['X_train']['summary'].values,  # 책 요약만 사용
        train_image_paths,
        data['processor'],
        data['y_train'].values
    )
    
    valid_dataset = CLIPDataset(
        data['X_valid'][data['field_names']].values,
        data['X_valid']['summary'].values,  # 책 요약만 사용
        valid_image_paths,
        data['processor'],
        data['y_valid'].values
    ) if args.dataset.valid_ratio != 0 else None

    test_dataset = CLIPDataset(
        data['test'][data['field_names']].values,
        data['test']['summary'].values,  # 책 요약만 사용
        test_image_paths,
        data['processor']
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.dataloader.batch_size,
        shuffle=args.dataloader.shuffle,
        num_workers=args.dataloader.num_workers
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.dataloader.batch_size,
        shuffle=False,
        num_workers=args.dataloader.num_workers
    ) if args.dataset.valid_ratio != 0 else None

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.dataloader.batch_size,
        shuffle=False,
        num_workers=args.dataloader.num_workers
    )

    data['train_dataloader'] = train_dataloader
    data['valid_dataloader'] = valid_dataloader 
    data['test_dataloader'] = test_dataloader

    return data
