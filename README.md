# Spell Correction Seq2Seq Model (PyTorch)

PyTorch 기반 Encoder–Decoder 구조를 사용해 **오타가 포함된 문장을 자동으로 교정하는 Seq2Seq 모델**입니다. LSTM 기반 Encoder–Decoder로 구현하고, Attention 사용하여 성능을 비교합니다.

---

## 프로젝트 개요

사용자가 입력한 문장에서 발생하는 철자 오류(typo)를 자동으로 수정하는 딥러닝 모델을 구현합니다.
비정형 텍스트 데이터를 기반으로 **오타 문장 → 정상 문장** 형태의 병렬 데이터를 구성하고,
LSTM 기반 Seq2Seq 모델을 학습하여 오타 교정 기능을 수행합니다.

---

## 프로젝트 목표

- PyTorch로 **Encoder–Decoder 기반 Seq2Seq 모델 직접 구현**
- 정상 문장 데이터를 기반으로 **Synthetic typo 데이터 생성**
- Teacher Forcing 및 Attention 적용
- 모델의 forward 흐름을 **top-down 구조로 시각적으로 설명 가능**하도록 구현
- 문장 단위 오타 교정 모델 완성
- attention 사용 시 성능 비교

---

## 사용 데이터셋

#### C4 200M Grammar Error Correction dataset

https://www.kaggle.com/datasets/dariocioni/c4200m/data

- input → target 문장

- 매우 좋음, 필수 핵심

- 문장-level Seq2Seq

#### 허깅 페이스 (torinriley/spell-correction)

https://huggingface.co/datasets/torinriley/spell-correction/viewer?views%5B%5D=train&sql=--+The+SQL+console+is+powered+by+DuckDB+WASM+and+runs+entirely+in+the+browser.%0A--+Get+started+by+typing+a+query+or+selecting+a+view+from+the+options+below.%0ASELECT+*+FROM+train+LIMIT+10%3B

- misspelled → correct 단어

- 단독으론 부족, 보조 데이터로 좋음

- 단어-level correction

“문장 단위 병렬 데이터(C4 200M Grammar Error Correction dataset)”을 메인으로 하고
(단어 단위 오타 데이터(torinriley/spell-correction)는 사전학습 + augmentation에 추가)

## 데이터셋

### ✔ 오타 합성(Synthetic Typo Generation)

오타 생성 규칙:

- 문자 삭제 (deletion)
- 문자 교체 (substitution)
- 인접 문자로 치환 (neighbor typo)
- 중복 삽입 (duplication)
- 임의 문자 삽입 (insertion)

```python
Input: “This is a sample sentence.”
Typo : “Ths is a sampl seentence.”
```

Synthetic parallel dataset이므로 학습 데이터가 무한하게 생성 가능.

---

## 모델 구조

### ✔ 기본 Seq2Seq 아키텍처

[Embedding]

↓

[Encoder LSTM]

↓

[Context Vector]

↓

[Decoder LSTM]

↓

[Linear → Softmax]

### ✔ Attention 적용 버전

Input Sentence

↓

Embedding

↓

Encoder LSTM

↓

Attention Layer

↓

Decoder LSTM

↓

Output Tokens

---

## 🔧 주요 PyTorch 구성 요소

- `nn.Embedding`
- `nn.LSTM`
- `nn.Linear`
- `nn.CrossEntropyLoss`
- `nn.Module` 기반 Encoder/Decoder
- Teacher Forcing
- Custom Dataset, Collate Function
- Greedy Decoding or Beam Search

---

## 실행 방법

### 1) 패키지 설치

pip install -r requirements.txt

### 2) 데이터 준비 및 오타 생성

python src/utils.py –create-typo-data

### 3) 학습

python src/train.py

### 4) 평가

python src/evaluate.py

### 5) 데모 실행

python demo.py

---

## 모델 평가 지표

- Character-level Accuracy
- Word-level Accuracy
- BLEU Score
- CER (Character Error Rate)
- Attention Heatmap 시각화

---

## 결과 예시

| 입력(오타 문장) | 출력(모델 교정)   | 정답              |
| --------------- | ----------------- | ----------------- |
| `Ths is smple.` | `This is simple.` | `This is simple.` |
| `I lov pytoch.` | `I love pytorch.` | `I love pytorch.` |

<img width="454" height="207" alt="스크린샷 2025-11-20 오전 10 50 42" src="https://github.com/user-attachments/assets/8c2c5ce9-7379-46f9-8c3e-133e497bcfc0" />

---

## 개발 일정 (4주)

### ✔ 1주차 : 데이터 수집 & 오타 생성 모듈 구현

### ✔ 2주차 : 기본 Seq2Seq 구현 및 학습

### ✔ 3주차 : Attention 및 성능 향상

### ✔ 4주차 : 모델 튜닝, 평가, 보고서/PPT 완성

---

## 향후 확장 가능성

- Transformer 기반 Spell Correction
- Beam Search 적용
- 한국어 오타 교정 모델로 확장
- 실시간(online) correction 서비스 API 구축

---

## 📧 문의

궁금한 부분은 언제든지 질문해주세요!

---

- 가상 환경 생성 : source .venv/bin/activate
- pip install -r requirements.txt
- requirements.txt 자동 생성

- README에 들어갈 아키텍처 다이어그램(Mermaid) 제작

- 전체 프로젝트 파일 생성

<details>
  <summary>주제 후보였던 것</summary>
  
## 주제 1.  한식 이미지 분류기 (Korean Food Classifier)
### 사진을 입력 받아, 이게 어떤 음식인지 분류하는 모델 
- 입력: 음식 이미지, 출력: 음식명 (예: 비빔밥, 김치찌개, 불고기 등))
- 데이터 수집 : https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=79
- 구현 방법 : Hugging Face의 BERT 대신, Torchvision의 사전 학습된 ResNet 또는 ViT(Vision Transformer)를 '기반(Base)'으로 가져온 모델로 이미지의 특징(Feature)을 추출
- 분류용 손실 함수 : nn.CrossEntropyLoss()
- 평가 지표 : 정확도(Accuracy) 또는 F1-Score

- CNN 기반 이미지 분류 모델 (ResNet, EfficientNet, ViT)

## 주제 2. 멜론 가사로 "아티스트 스타일 모방" 가사 생성기 만들기

### 아티스트들의 기존 노래 가사들을 학습하여, 아티스트 스타일을 모방한 가사를 생성하는 모델 (짧게 생성)

- 입력: 가수명, 출력: 아티스트 스타일을 모방한 가사 생성
- 데이터 수집 : 멜론 노래 가사 및 정보 크롤링
- 구현 방법 : RNN(LSTM 또는 GRU) 기반 시퀀스 생성 모델을 nn.Module로 직접 구현
- 생성용 손실 함수 : nn.CrossEntropyLoss()
- 언어모델 평가지표 : Perplexity (PPL)

## 주제 3. 멜론 가사로 아티스트 분류하는 모델

### 크롤링한 가사 한 구절을 모델에 입력하면, 이 가사를 '아이유'가 썼는지, 'BTS'가 썼는지, '김광석'이 썼는지 맞추는(분류하는) 모델

- 입력 : 크롤링한 가사 한 구절, 출력 : 아티스트 명
- 데이터 수집 : 멜론 노래 가사 및 정보 크롤링
- 구현 방법 : Hugging Face의 BERT 모델을 '기반(Base)'으로 가져온 뒤, 그 위에 '분류용 헤드(Head)' 레이어를 nn.Module을 사용해 직접 구현
- 분류용 손실 함수 : nn.CrossEntropyLoss()
- 평가 지표 : 정확도(Accuracy)

---

#### 데이터셋 후보 : C4 200M Grammar Error Correction dataset

https://www.kaggle.com/datasets/dariocioni/c4200m/data

</details>
