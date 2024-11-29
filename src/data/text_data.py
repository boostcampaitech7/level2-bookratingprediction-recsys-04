import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from .basic_data import basic_data_split
from .context_data import process_context_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def text_preprocessing(summary):
    if pd.isna(summary): 
        return ""
    else :
        summary = re.sub("[^0-9a-zA-Z.,!?]", " ", summary)
        summary = re.sub("\s+", " ", summary)
    return summary

def text_to_vector(text, tokenizer, model,name):

    if 'bert' in name :
        text_ = "[CLS] " + text + " [SEP]"
        tokenized = tokenizer.encode(text_, add_special_tokens=True)
        token_tensor = torch.tensor([tokenized], device=model.device)
        with torch.no_grad():
            outputs = model(token_tensor)  # attention_mask를 사용하지 않아도 됨
            ### BERT 모델의 경우, 최종 출력물의 사이즈가 (토큰길이, 임베딩=768)이므로, 이를 평균내어 사용하거나 pooler_output을 사용하여 [CLS] 토큰의 임베딩만 사용
            # sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0)  # 방법1) 모든 토큰의 임베딩을 평균내어 사용
            sentence_embedding = outputs.pooler_output.squeeze(0)  # 방법2) pooler_output을 사용하여 맨 첫 토큰인 [CLS] 토큰의 임베딩만 사용
        return sentence_embedding.cpu().detach().numpy() 
    
    else :
        tokenized = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**tokenized)
            # ELECTRA는 last_hidden_state의 [CLS] 토큰 위치의 벡터를 사용
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        
        return sentence_embedding.squeeze(0).cpu().detach().numpy()



def process_text_data(ratings, users, books, tokenizer, model, name, vector_create=False):
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
    
    sanitized_name = name.replace("/", "_")

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
            vector = text_to_vector(prompt_, tokenizer, model,name)
            book_summary_vector_list.append(vector)
        
        book_summary_vector_list = np.concatenate([
                                                books_['isbn'].values.reshape(-1, 1),
                                                np.asarray(book_summary_vector_list, dtype=np.float32)
                                                ], axis=1)
        
        np.save(f'./data/text_vector/{sanitized_name}_book_summary_vector.npy', book_summary_vector_list)        



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
            vector = text_to_vector(prompt_, tokenizer, model,name)
            user_summary_merge_vector_list.append(vector)
        
        user_summary_merge_vector_list = np.concatenate([
                                                         users_['user_id'].values.reshape(-1, 1),
                                                         np.asarray(user_summary_merge_vector_list, dtype=np.float32)
                                                        ], axis=1)

        np.save(f'./data/text_vector/{sanitized_name}_user_summary_merge_vector.npy', user_summary_merge_vector_list)        

        
    else:
        print('Check Vectorizer')
        print('Vector Load')

        book_summary_vector_list = np.load(f'./data/text_vector/{sanitized_name}_book_summary_vector.npy', allow_pickle=True)
        user_summary_merge_vector_list = np.load(f'./data/text_vector/{sanitized_name}_user_summary_merge_vector.npy', allow_pickle=True)


    book_summary_vector_df = pd.DataFrame({'isbn': book_summary_vector_list[:, 0]})
    book_summary_vector_df['book_summary_vector'] = list(book_summary_vector_list[:, 1:].astype(np.float32))
    user_summary_vector_df = pd.DataFrame({'user_id': user_summary_merge_vector_list[:, 0]})
    user_summary_vector_df['user_summary_merge_vector'] = list(user_summary_merge_vector_list[:, 1:].astype(np.float32))

    books_ = pd.merge(books_, book_summary_vector_df, on='isbn', how='left')
    users_ = pd.merge(users_, user_summary_vector_df, on='user_id', how='left')

    return users_, books_

# Text_Dataset 클래스는 그대로 유지
class Text_Dataset(Dataset):
    def __init__(self, user_book_vector, user_summary_vector, book_summary_vector, rating=None):

        self.user_book_vector = user_book_vector
        self.user_summary_vector = user_summary_vector
        self.book_summary_vector = book_summary_vector
        self.rating = rating
        
    def __len__(self):
        return self.user_book_vector.shape[0]
    
    def __getitem__(self, i):
        return {
            'user_book_vector': torch.tensor(self.user_book_vector[i], dtype=torch.long),
            'user_summary_vector': torch.tensor(self.user_summary_vector[i], dtype=torch.float32),
            'book_summary_vector': torch.tensor(self.book_summary_vector[i], dtype=torch.float32),
            'rating': torch.tensor(self.rating[i], dtype=torch.float32),
        } if self.rating is not None else {
            'user_book_vector': torch.tensor(self.user_book_vector[i], dtype=torch.long),
            'user_summary_vector': torch.tensor(self.user_summary_vector[i], dtype=torch.float32),
            'book_summary_vector': torch.tensor(self.book_summary_vector[i], dtype=torch.float32),
        }

def text_data_load(args):
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    tokenizer = AutoTokenizer.from_pretrained(args.model_args[args.model].pretrained_model)
    model = AutoModel.from_pretrained(args.model_args[args.model].pretrained_model).to(device=args.device)
    model.eval()
    
    users_, books_ = process_text_data(train, users, books, tokenizer, model, args.model_args[args.model].pretrained_model,args.model_args[args.model].vector_create)
    users_, books_ = process_context_data(users_, books_)
    
    user_features = args.dataset.features.user
    book_features = args.dataset.features.book
    sparse_cols = ['user_id', 'isbn'] + list(set(user_features + book_features) - {'user_id', 'isbn'})
    
    train_df = train.merge(books_, on='isbn', how='left')\
                    .merge(users_, on='user_id', how='left')[sparse_cols + ['user_summary_merge_vector', 'book_summary_vector', 'rating']]
    test_df = test.merge(books_, on='isbn', how='left')\
                  .merge(users_, on='user_id', how='left')[sparse_cols + ['user_summary_merge_vector', 'book_summary_vector']]
    all_df = pd.concat([train_df, test_df], axis=0)

    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = train_df[col].astype("category").cat.codes
        test_df[col] = test_df[col].astype("category").cat.codes

    field_dims = [len(label2idx[col]) for col in sparse_cols]

    data = {
        'train': train_df,
        'test': test_df,
        'field_names': sparse_cols,
        'field_dims': field_dims,
        'label2idx': label2idx,
        'idx2label': idx2label,
        'sub': sub,
    }
    
    return data

# text_data_split과 text_data_loader 함수는 그대로 유지
def text_data_split(args, data):
    return basic_data_split(args, data)

def text_data_loader(args, data):
    train_dataset = Text_Dataset(
        data['X_train'][data['field_names']].values,
        data['X_train']['user_summary_merge_vector'].values,
        data['X_train']['book_summary_vector'].values,
        data['y_train'].values
    )
    
    valid_dataset = Text_Dataset(
        data['X_valid'][data['field_names']].values,
        data['X_valid']['user_summary_merge_vector'].values,
        data['X_valid']['book_summary_vector'].values,
        data['y_valid'].values
    ) if args.dataset.valid_ratio != 0 else None
    
    test_dataset = Text_Dataset(
        data['test'][data['field_names']].values,
        data['test']['user_summary_merge_vector'].values,
        data['test']['book_summary_vector'].values,
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
