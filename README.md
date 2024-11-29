# 🏠 Book Rating Prediction

## 📌 프로젝트 개요  
소비자들의 책 구매 결정에 대한 도움을 주기 위한 개인화된 상품 추천 대회입니다.

## 📊 프로젝트 데이터

**books.csv** : 책에 대한 고유번호, 제목, 작가, 연도, 출판사, 표지 url, 출판 언어, 장르 카테고리, 요약, 표지에 대한 파일 경로

**user.csv** : 사용자에 대한 아이디, 위치 정보와 나이

**train_rating.csv, test_train.csv** : 사용자 아이디, 책 고유번호와 평점

## 🗂️ 파일 구조
```
├── ensemble.py
├── main.py
├── readme.md
├── requirements.txt
├── config
│   ├── CLIP.yaml
│   ├── ELECTRA.yaml
│   ├── ResNet.yaml
│   ├── RoBERTa.yaml
│   ├── VGGNet.yaml
│   └── ensemble.yaml
├── saved
│   └── submit
│       ├── CLIP.csv
│       ├── ELECTRA.csv
│       ├── ensemble.csv
│       ├── ResNet.csv
│       ├── RoBERTa.csv
│       └── VGGNet.csv
└── src
    ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   ├── basic_data.py
    │   ├── combined_data.py
    │   ├── context_data.py
    │   ├── image_data.py
    │   └── text_data.py
    ├── ensembles
    │   └── ensembles.py
    ├── loss
    │   └── loss.py
    ├── models
    │   ├── Image.py
    │   ├── Multimodal.py
    │   ├── Text.py
    │   ├── __init__.py
    │   └── _helpers.py
    ├── train
    │   ├── __init__.py
    │   └── trainer.py
    └── utils.py
```
### 폴더 및 파일 설명
- **ensemble.py**
  
  `esemble.py`는 model의 결과들을 soft voting 방식으로 앙상블해주는 코드입니다. yaml 파일을 읽어와 가중치와 함께 예측을 진행합니다.

- **main.py**

  `main.py`는 model의 yaml 파일을 읽어서 개별 모델의 학습을 진행시킵니다.
     
- **config 폴더**
  
  `CLIP.yaml`, `ELECTRA.yaml`, `ResNet.yaml`, `RoBERTa.yaml`, `VGGNet.yaml`, 들은 단일 모델을 위한 파라미터가 적혀있는 YAML 파일입니다.
  `ensemble.yaml`는 `ensemble.py`를 실행할 때 사용하는 YAML 파일입니다. 앙상블하고 싶은 CSV 파일과 각 모델에 할당할 가중치가 적혀 있습니다.

- **saved 폴더**
  
  log파일과 checkpoint 파일들이 생성되는 곳입니다. 또한 각 모델의 결과가 생성됩니다.
  
- **src 폴더**  
  이 폴더에는 프로젝트의 핵심 Python 코드가 포함되어 있습니다.

  - `data 폴더` : 각종 모델 이름에 따른 데이터 전처리 파일들이 들어있습니다.
    
  - `ensembles 폴더`: `ensemble.py` 을 실행하기 위한 `ensembles.py`가 들어있습니다. 

  - `loss 폴더` : 각종 loss 가 구현되어 있는 `loss.py`가 들어있습니다.
    
  - `models 폴더`: 각종 모델들이 구현된 파일들이 들어 있습니다.

  - `train 폴더`: 모델을 학습하기 위한 train 코드가 구현되어 있습니다.

  - `utils.py`: 각종 학습에 도움이 되는 함수들이 구현되어 있습니다.
 
    
## 🛠️ 사용 방법
1. **개별 모델 실행:**  
    ```bash
    python main.py -c config/model.yaml
    ```
2. **앙상블 실행:**
   ```bash
    python ensemble.py --config config/ensemble.yaml
   ```
## 🎯 파이널 제출 내역

|     | Public | Private |
|--------|--------|--------|
| CLIP |  2.1849 |  2.1780 |
|esemble |  2.2149  | 2.2108 |



## 😊 팀 구성원
<div align="center">
<table>
  <tr>
        <td align="center"><a href="https://github.com/annakong23"><img src="https://avatars.githubusercontent.com/u/102771961?v=4" width="100px;" alt=""/><br /><sub><b>공지원</b></sub><br />
    </td>
         <td align="center"><a href="https://github.com/Timeisfast"><img src="https://avatars.githubusercontent.com/u/120894109?v=4" width="100px;" alt=""/><br /><sub><b>김성윤</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/kimjueun028"><img src="https://avatars.githubusercontent.com/u/92249116?v=4" width="100px;" alt=""/><br /><sub><b>김주은</b></sub><br />
    </td>
    </td>
        <td align="center"><a href="https://github.com/zip-sa"><img src="https://avatars.githubusercontent.com/u/49730616?v=4" width="100px;" alt=""/><br /><sub><b>박승우</b></sub><br /> 
    </td>
        <td align="center"><a href="https://github.com/gagoory7"><img src="https://avatars.githubusercontent.com/u/163074222?v=4" width="100px;" alt=""/><br /><sub><b>백상민</b></sub><br />
    </td>
  </tr>
</table>
</div>

<br />
