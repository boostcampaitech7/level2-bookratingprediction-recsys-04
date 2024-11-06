import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from .basic_data import basic_data_split
import regex

def str2list(x: str) -> list:
    '''문자열을 리스트로 변환하는 함수'''
    return x[1:-1].split(', ')


def split_location(x: str) -> list:
    '''
    Parameters
    ----------
    x : str
        location 데이터

    Returns
    -------
    res : list
        location 데이터를 나눈 뒤, 정제한 결과를 반환합니다.
        순서는 country, state, city, ... 입니다.
    '''
    res = x.split(',')
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]  # remove special characters
    res = [i if i not in ['n/a', ''] else np.nan for i in res]  # change 'n/a' into NaN
    res.reverse()  # reverse the list to get country, state, city, ... order

    for i in range(len(res)-1, 0, -1):
        if (res[i] in res[:i]) and (not pd.isna(res[i])):  # remove duplicated values if not NaN
            res.pop(i)

    return res


def process_context_data(users, books):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    
    Returns
    -------
    label_to_idx : dict
        데이터를 인덱싱한 정보를 담은 딕셔너리
    idx_to_label : dict
        인덱스를 다시 원래 데이터로 변환하는 정보를 담은 딕셔너리
    train_df : pd.DataFrame
        train 데이터
    test_df : pd.DataFrame
        test 데이터
    """

    users_ = users.copy()
    books_ = books.copy()

    # 데이터 전처리 (전처리는 각자의 상황에 맞게 진행해주세요!)
    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else np.nan)
    books_['language'] = books_['language'].fillna(books_['language'].mode()[0])
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // 10 * 10)  # 1990년대, 2000년대, 2010년대, ...

    users_['age'] = users_['age'].fillna(users_['age'].mode()[0])
    users_['age_range'] = users_['age'].apply(lambda x: x // 10 * 10)  # 10대, 20대, 30대, ...

    users_['location_list'] = users_['location'].apply(lambda x: split_location(x)) 
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0])
    users_['location_state'] = users_['location_list'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    users_['location_city'] = users_['location_list'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
    for idx, row in users_.iterrows():
        if (not pd.isna(row['location_state'])) and pd.isna(row['location_country']):
            fill_country = users_[users_['location_state'] == row['location_state']]['location_country'].mode()
            fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
            users_.loc[idx, 'location_country'] = fill_country
        elif (not pd.isna(row['location_city'])) and pd.isna(row['location_state']):
            if not pd.isna(row['location_country']):
                fill_state = users_[(users_['location_country'] == row['location_country']) 
                                    & (users_['location_city'] == row['location_city'])]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                users_.loc[idx, 'location_state'] = fill_state
            else:
                fill_state = users_[users_['location_city'] == row['location_city']]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                fill_country = users_[users_['location_city'] == row['location_city']]['location_country'].mode()
                fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
                users_.loc[idx, 'location_country'] = fill_country
                users_.loc[idx, 'location_state'] = fill_state

               
    
    users_ = users_.drop(['location'], axis=1)

    return users_, books_



def text_preprocessing(summary):
    """
    Parameters
    ----------
    summary : pd.Series
        정규화와 같은 기본적인 전처리를 하기 위한 텍스트 데이터를 입력합니다.
    
    Returns
    -------
    summary : pd.Series
        전처리된 텍스트 데이터를 반환합니다.
        베이스라인에서는 특수문자 제거, 공백 제거를 진행합니다.
    """
    summary = re.sub("[^0-9a-zA-Z.,!?]", " ", summary)  # .,!?를 제외한 특수문자 제거
    summary = re.sub("\s+", " ", summary)  # 중복 공백 제거

    return summary


def text_to_vector(text, tokenizer, model):
    """
    Parameters
    ----------
    text : str
        `summary_merge()`를 통해 병합된 요약 데이터
    tokenizer : Tokenizer
        텍스트 데이터를 `model`에 입력하기 위한 토크나이저
    model : 사전학습된 언어 모델
        텍스트 데이터를 벡터로 임베딩하기 위한 모델
    ----------
    """
    text_ = "[CLS] " + text + " [SEP]"
    tokenized = tokenizer.encode(text_, add_special_tokens=True)
    token_tensor = torch.tensor([tokenized], device=model.device)
    with torch.no_grad():
        outputs = model(token_tensor)  # attention_mask를 사용하지 않아도 됨
        ### BERT 모델의 경우, 최종 출력물의 사이즈가 (토큰길이, 임베딩=768)이므로, 이를 평균내어 사용하거나 pooler_output을 사용하여 [CLS] 토큰의 임베딩만 사용
        # sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0)  # 방법1) 모든 토큰의 임베딩을 평균내어 사용
        sentence_embedding = outputs.pooler_output.squeeze(0)  # 방법2) pooler_output을 사용하여 맨 첫 토큰인 [CLS] 토큰의 임베딩만 사용
    
    return sentence_embedding.cpu().detach().numpy() 


def process_text_data(ratings, users, books, tokenizer, model,name, vector_create=False):
    """
    Parameters
    ----------
    users : pd.DataFrame
        유저 정보에 대한 데이터 프레임을 입력합니다.
    books : pd.DataFrame
        책 정보에 대한 데이터 프레임을 입력합니다.
    vector_create : bool
        사전에 텍스트 데이터 벡터화가 된 파일이 있는지 여부를 입력합니다.

    Returns
    -------
    `users_` : pd.DataFrame
        각 유저가 읽은 책에 대한 요약 정보를 병합 및 벡터화하여 추가한 데이터 프레임을 반환합니다.

    `books_` : pd.DataFrame
        텍스트 데이터를 벡터화하여 추가한 데이터 프레임을 반환합니다.
    """
    num2txt = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
    users_ = users.copy()
    books_ = books.copy()
    nan_value = 'None'
    books_['summary'] = books_['summary'].fillna(nan_value)\
                                         .apply(lambda x: text_preprocessing(x))\
                                         .replace({'': nan_value, ' ': nan_value})
    
    books_['summary_length'] = books_['summary'].apply(lambda x:len(x))
    books_['review_count'] = books_['isbn'].map(ratings['isbn'].value_counts())

    users_['books_read'] = users_['user_id'].map(ratings.groupby('user_id')['isbn'].apply(list))

    if vector_create:
        if not os.path.exists('./data/text_vector'):
            os.makedirs('./data/text_vector')

        print('Create Item Summary Vector')
        book_summary_vector_list = []
        for title, summary in tqdm(zip(books_['book_title'], books_['summary']), total=len(books_)):
            # 책에 대한 텍스트 프롬프트는 아래와 같이 구성됨
            # '''
            # Book Title: {title}
            # Summary: {summary}
            # '''
            prompt_ = f'Book Title: {title}\n Summary: {summary}\n'
            vector = text_to_vector(prompt_, tokenizer, model)
            book_summary_vector_list.append(vector)
        
        book_summary_vector_list = np.concatenate([
                                                books_['isbn'].values.reshape(-1, 1),
                                                np.asarray(book_summary_vector_list, dtype=np.float32)
                                                ], axis=1)
        
        np.save(f'./data/text_vector/{name}_book_summary_vector.npy', book_summary_vector_list)        


        print('Create User Summary Merge Vector')
        user_summary_merge_vector_list = []
        for books_read in tqdm(users_['books_read']):
            if not isinstance(books_read, list) and pd.isna(books_read):  # 유저가 읽은 책이 없는 경우, 텍스트 임베딩을 0으로 처리
                user_summary_merge_vector_list.append(np.zeros((768)))
                continue
            
            read_books = books_[books_['isbn'].isin(books_read)][['book_title', 'summary', 'review_count']]
            read_books = read_books.sort_values('review_count', ascending=False).head(5)  # review_count가 높은 순으로 5개의 책을 선택
            # 유저에 대한 텍스트 프롬프트는 아래와 같이 구성됨
            # DeepCoNN에서 유저의 리뷰를 요약하여 하나의 벡터로 만들어 사용함을 참고 (https://arxiv.org/abs/1701.04783)
            # '''
            # Five Books That You Read
            # 1. Book Title: {title}
            # Summary: {summary}
            # ...
            # 5. Book Title: {title}
            # Summary: {summary}
            # '''
            prompt_ = f'{num2txt[len(read_books)]} Books That You Read\n'
            for idx, (title, summary) in enumerate(zip(read_books['book_title'], read_books['summary'])):
                summary = summary if len(summary) < 100 else f'{summary[:100]} ...'
                prompt_ += f'{idx+1}. Book Title: {title}\n Summary: {summary}\n'
            vector = text_to_vector(prompt_, tokenizer, model)
            user_summary_merge_vector_list.append(vector)
        
        user_summary_merge_vector_list = np.concatenate([
                                                         users_['user_id'].values.reshape(-1, 1),
                                                         np.asarray(user_summary_merge_vector_list, dtype=np.float32)
                                                        ], axis=1)
        
        np.save(f'./data/text_vector/{name}_user_summary_merge_vector.npy', user_summary_merge_vector_list)        
        
    else:
        print('Check Vectorizer')
        print('Vector Load')
        book_summary_vector_list = np.load(f'./data/text_vector/{name}_book_summary_vector.npy', allow_pickle=True)
        user_summary_merge_vector_list = np.load(f'./data/text_vector/{name}_user_summary_merge_vector.npy', allow_pickle=True)

    book_summary_vector_df = pd.DataFrame({'isbn': book_summary_vector_list[:, 0]})
    book_summary_vector_df['book_summary_vector'] = list(book_summary_vector_list[:, 1:].astype(np.float32))
    user_summary_vector_df = pd.DataFrame({'user_id': user_summary_merge_vector_list[:, 0]})
    user_summary_vector_df['user_summary_merge_vector'] = list(user_summary_merge_vector_list[:, 1:].astype(np.float32))

    books_ = pd.merge(books_, book_summary_vector_df, on='isbn', how='left')
    users_ = pd.merge(users_, user_summary_vector_df, on='user_id', how='left')

    return users_, books_






def image_vector(path, img_size):

    img = Image.open(path)
    transform = v2.Compose([
        v2.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(img).numpy()


def process_img_data(books, args):
    """
    Parameters
    ----------
    books : pd.DataFrame
        책 정보에 대한 데이터 프레임을 입력합니다.
    
    Returns
    -------
    books_ : pd.DataFrame
        이미지 정보를 벡터화하여 추가한 데이터 프레임을 반환합니다.
    """
    books_ = books.copy()
    books_['img_path'] = books_['img_path'].apply(lambda x: f'data/{x}')
    img_vecs = []
    for idx in tqdm(books_.index):
        img_vec = image_vector(books_.loc[idx, 'img_path'], args.model_args[args.model].img_size)
        img_vecs.append(img_vec)

    books_['img_vector'] = img_vecs

    return books_

def mixture_data_split(args, data):
    """학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다."""
    print(data['train'].head())
    return basic_data_split(args, data)

 
class mixture_Dataset(Dataset):
    def __init__(self, user_book_vector, user_summary_vector, book_summary_vector, img_vector, rating=None):
        self.user_book_vector = user_book_vector
        self.user_summary_vector = user_summary_vector
        self.book_summary_vector = book_summary_vector
        self.img_vector = img_vector
        self.rating = rating
    def __len__(self):
        return self.user_book_vector.shape[0]
    def __getitem__(self, i):
        return {
                'user_book_vector' : torch.tensor(self.user_book_vector[i], dtype=torch.long),
                'user_summary_vector' : torch.tensor(self.user_summary_vector[i], dtype=torch.float32),
                'book_summary_vector' : torch.tensor(self.book_summary_vector[i], dtype=torch.float32),
                'img_vector' : torch.tensor(self.img_vector[i], dtype=torch.float32),
                'rating' : torch.tensor(self.rating[i], dtype=torch.float32)
                } if self.rating is not None else \
                {
                'user_book_vector' : torch.tensor(self.user_book_vector[i], dtype=torch.long),
                'user_summary_vector' : torch.tensor(self.user_summary_vector[i], dtype=torch.float32),
                'book_summary_vector' : torch.tensor(self.book_summary_vector[i], dtype=torch.float32),
                'img_vector' : torch.tensor(self.img_vector[i], dtype=torch.float32)
                }




def mixture_data_loader(args, data):
    train_dataset = mixture_Dataset(
                                data['X_train'][data['field_names']].values,
                                data['X_train']['user_summary_merge_vector'].values,
                                data['X_train']['book_summary_vector'].values,
                                data['X_train']['img_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = mixture_Dataset(
                                data['X_valid'][data['field_names']].values,
                                data['X_valid']['user_summary_merge_vector'].values,
                                data['X_valid']['book_summary_vector'].values,
                                data['X_train']['img_vector'].values,
                                data['y_valid'].values
                                ) if args.dataset.valid_ratio != 0 else None
    test_dataset = mixture_Dataset(
                                data['test'][data['field_names']].values,
                                data['test']['user_summary_merge_vector'].values,
                                data['test']['book_summary_vector'].values,
                                data['X_train']['img_vector'].values,
                                )


    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    
    return data


def mixture_data_load(args):
    users = pd.read_csv(args.dataset.data_path + 'users.csv')  
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    users_, books_ = process_context_data(users, books)

    books_ = process_img_data(books_, args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_args[args.model].pretrained_model)
    model = AutoModel.from_pretrained(args.model_args[args.model].pretrained_model).to(device=args.device)
    model.eval()
    
    users_, books_ = process_text_data(train, users_, books_, tokenizer, model, args.model_args[args.model].pretrained_model,args.model_args[args.model].vector_create)

    user_features = [ 'age', 'age_range', 'location_list', 'location_country',
                    'location_state', 'location_city', 'books_read']
    book_features = ['book_title', 'book_author', 'year_of_publication', 'publisher',
                    'language', 'category', 'publication_range', 'summary_length','review_count']
    
    user_features = []
    book_features = []

    sparse_cols = ['user_id', 'isbn'] + list(set(user_features + book_features) - {'user_id', 'isbn'})
    
    train_df = train.merge(users_, on='user_id', how='left')\
                    .merge(books_, on='isbn', how='left')[sparse_cols + ['img_vector','user_summary_merge_vector', 'book_summary_vector', 'rating']]
    test_df = test.merge(users_, on='user_id', how='left')\
                  .merge(books_, on='isbn', how='left')[sparse_cols + ['img_vector','user_summary_merge_vector', 'book_summary_vector']]

    all_df = pd.concat([train_df, test_df], axis=0)


    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = train_df[col].map(label2idx[col])
        test_df[col] = test_df[col].map(label2idx[col])

    field_dims = [len(label2idx[col]) for col in sparse_cols]

    data = {
            'train':train_df,
            'test':test_df,
            'field_names':sparse_cols,
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            'sub':sub,
            }

    return data