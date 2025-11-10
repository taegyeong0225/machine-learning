# machine-learning 프로젝트
pytorch의 nn.Module을 사용해 직접 구현

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
- 입력: 음식 이미지, 출력: 음식명 (예: 비빔밥, 김치찌개, 불고기 등))
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

컴퓨터비전 데이터셋?
https://www.nexdata.ai/datasets/computervision?page=2
