# SpellGram 기반 LSTM Seq2Seq 오타·문장 교정 프로젝트 (PyTorch)

PyTorch 기반 Encoder–Decoder 구조를 사용하여 영어 문장에서 발생하는 오타와 문법적 오류를 자동으로 교정하는 Seq2Seq 모델을 구현한 프로젝트입니다.  
단어 수준의 철자 오류부터 문장 구성 오류까지 다양한 패턴을 학습하도록 구성했습니다.
Sequence-to-Sequence(Seq2Seq) 형태의 자연어 처리(NLP) 문제이며, LSTM Encoder–Decoder 구조와 Attention 메커니즘을 통해 문장 단위의 오타를 효과적으로 교정합니다.

---

## 프로젝트 개요

본 프로젝트는 오타가 포함된 문장을 입력받아 정상적인 문장으로 변환하는 문장 교정 모델을 구축하는 것을 목표로 한다.  
비정형 텍스트 데이터를 기반으로 병렬 데이터(source → target)를 구성하여 학습하며,  
PyTorch로 직접 구현한 LSTM Encoder–Decoder(Seq2Seq) 모델을 사용한다.

---
## 문제 정의 (Problem Definition)
- **입력(Input)**  
  오타, 철자 오류, 단어 치환, 문법 오류 등이 포함된 영어 문장 (source)

- **출력(Output)**  
  정상적인 철자와 문법을 갖춘 영어 문장 (target)

이 문제는 **Sequence-to-Sequence(Seq2Seq)** 기반의 자연어 처리(NLP) 작업이며,  
LSTM Encoder–Decoder 구조를 통해 문장 단위 오류를 교정한다.

---

## 프로젝트 목표

- PyTorch로 **Encoder–Decoder 기반 Seq2Seq 모델 직접 구현**
- 정상 문장을 기반으로 **오타 합성(Synthetic Typo Generation)** 가능하도록 설계
- Teacher Forcing 적용 및 학습 안정화
- LSTM 모델의 forward 흐름을 **top-down 구조**로 명확히 설명할 수 있도록 구현
- 문장 단위 오타 교정 모델 완성

---
## 사용 데이터셋 

### SpellGram Dataset (HuggingFace)

- URL: https://huggingface.co/datasets/vishnun/SpellGram
- 크기: 약 40,000 sentence pairs
- 구성:
  - `source`: 오타가 포함된 문장
  - `target`: 정상 문장
    
### 데이터 특징
- 평균 문장 길이: 약 8~12 단어
- 다양한 오타 형태 포함
  - 단어 단위 오류
  - 철자 오류
  - 단어 추가/삭제
  - 단어 치환 오류
- Levenshtein distance 기반으로 차이 분석 가능

### SpellGram 외 참고 데이터
- torinriley/spell-correction  
  단어 단위 오타 → 정답 단어 형태  
  → 본 프로젝트에서는 **보조 데이터**로 활용 가능

### 데이터 분석 수행 내용
- 문장 길이 통계 분석
- 오류 유형(error_type) 자동 태깅
  - 단일 단어 오타
  - 다중 오타
  - 단어 삽입/삭제 오류
- vocab 생성, UNK 비율 분석
- Levenshtein distance 기반 diff 시각화
- 랜덤 샘플 기반 source-target 차이 분석

<details>
  <summary>오타 합성(Synthetic Typo Generation)</summary>
      오타 생성 규칙

      : 문장 교정 모델의 학습 범위를 넓히기 위해 오타 합성 규칙을 추가할 수 있다.  
      적용 가능한 규칙 예시는 다음과 같다.
      
      - 문자 삭제 (deletion)
  
      - 문자 교체 (substitution)
      
      - 인접 문자로 치환 (neighbor typo)
      
      - 중복 삽입 (duplication)
      
      - 임의 문자 삽입 (insertion)
      
      예시:
      
      Input: “This is a sample sentence.”
      
      Typo : “Ths is a sampl seentence.”
      
      Synthetic 데이터의 장점은 무한하게 생성 가능한 parallel dataset을 얻을 수 있다는 것이다.
</details>

---

## Seq2Seq 모델 구조

이 모델은 Encoder와 Decoder로 구성된 기본 LSTM Seq2Seq 아키텍처를 따른다.

### Encoder
- Embedding Layer
- LSTM Layer (hidden state, cell state 출력)

### Decoder
- Embedding Layer
- LSTM Layer
- Linear Projection Layer to vocab size
- Teacher forcing 기법 적용

### Special Tokens
- `<PAD>`: 패딩
- `<UNK>`: 어휘집에 없는 단어
- `<SOS>`: 문장 시작
- `<EOS>`: 문장 종료


## 실험 환경 및 Hyperparameter 변경

### 기본 설정
| 항목              | 값         |
|-------------------|------------|
| vocab_size        | 10000      |
| embed_dim         | 256        |
| hidden_dim        | 512        |
| num_layers        | 1          |
| batch_size        | 32         |
| learning_rate     | 1e-3       |
| max_len           | 40         |
| teacher_forcing   | 0.5        |

### Hyperparameter 변경 실험
- vocab_size 5000 → 10000 증가 시 UNK 비율 감소
- hidden_dim 256 → 512 변경 시 학습 안정성 증가
- teacher_forcing_ratio 0.3, 0.5, 0.7 비교 실험 수행

---

## 모델 성능

### 평가 기준
- Word-level Accuracy
- Sentence-level Accuracy
- Levenshtein Distance
- BLEU Score(선택)

### 예시 결과
| 평가 항목               | 값         |
|-------------------------|------------|
| Word Accuracy          | 92~95%     |
| Sentence Accuracy      | 72~78%     |
| 평균 Levenshtein 거리 | 1.4~2.1    |

---

## 결론 (예시)

- LSTM Seq2Seq 모델은 오타 및 문장 교정 작업에서 기본적인 성능을 보여주었다.
- vocab size 증가 시 UNK 비율 감소 및 성능 개선 효과 확인됨.
- Levenshtein 기반 비교 결과, 대다수의 오타 패턴을 안정적으로 교정함.
- 향후 Transformer 기반 모델 또는 Attention 기반 Seq2Seq로 확장 가능성이 높음.

---

## 향후 발전 방향

- Attention 메커니즘 추가  
- Bidirectional LSTM Encoder  
- Transformer 기반 Encoder–Decoder로 확장  
- Beam Search 기반 문장 생성  
- WordPiece/BPE 기반 Subword Tokenizer 도입
- Transformer 기반 Spell Correction
- Beam Search 적용
- 한국어 오타 교정 모델로 확장
- 실시간(online) correction 서비스 API 구축

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

## 주요 PyTorch 구성 요소
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

| 입력(오타 문장) | 출력(모델 교정) | 정답 |
|----------------|------------------|-------|
| `Ths is smple.` | `This is simple.` | `This is simple.` |
| `I lov pytoch.` | `I love pytorch.` | `I love pytorch.` |


<img width="454" height="207" alt="스크린샷 2025-11-20 오전 10 50 42" src="https://github.com/user-attachments/assets/8c2c5ce9-7379-46f9-8c3e-133e497bcfc0" />
  
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
- ----
#### 데이터셋 후보 : C4 200M Grammar Error Correction dataset
https://www.kaggle.com/datasets/dariocioni/c4200m/data

</details>




