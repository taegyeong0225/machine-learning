"""
Tokenizer for Spell Correction Model

SimpleTokenizer 클래스: 단어 기반 토크나이저 + 어휘집 관리
"""

import pickle
from collections import Counter
from pathlib import Path


class SimpleTokenizer:
    """
    간단한 단어 기반 토크나이저 + 어휘집(Vocabulary) 관리 클래스
    PAD, UNK, SOS, EOS 4개 기본 토큰 포함
    """
    
    def __init__(self, vocab_size=10000, verbose=True):
        """
        Args:
            vocab_size (int): 목표 어휘 크기
            verbose (bool): 출력 메시지 표시 여부
        """
        self.vocab_size = vocab_size
        self.verbose = verbose
        
        # 기본 토큰 정의
        self.word2idx = {
            '<PAD>': 0,  # Padding Token, 배치 연산 위해 문장 길이 맞추기
            '<UNK>': 1,  # Unknown Token, 어휘집에 없는 단어 처리
            '<SOS>': 2,  # Start of Sentence, 문장 생성 시작 신호
            '<EOS>': 3   # End of Sentence, 문장 종료 신호
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        self.word_freq = Counter()  # 단어 빈도
        
        if self.verbose:
            print(f"[Init] 기본 vocab size: {len(self.word2idx)} / 목표 vocab: {vocab_size}")
    
    def build_vocab(self, texts):
        """
        전체 문장 리스트를 받아 **글자(Character)** 빈도 기반 어휘집 구축
        
        Args:
            texts (list): 문장 리스트
        """
        if self.verbose:
            print("[Vocab] 어휘집 생성 중... (Character-level)")
        
        # None 또는 빈 문자열 방지
        for text in texts:
            if text:
                # 글자 단위로 분리 (공백 포함)
                self.word_freq.update(list(text.lower()))
        
        # 가장 빈도 높은 글자부터 vocab_size만큼 추가
        next_idx = len(self.word2idx)
        
        for char, _ in self.word_freq.most_common(self.vocab_size - next_idx):
            self.word2idx[char] = next_idx
            self.idx2word[next_idx] = char
            next_idx += 1
        
        if self.verbose:
            print(f"[Vocab] 구축 완료! 총 vocab size = {len(self.word2idx)}")
    
    def encode(self, text):
        """
        문장을 **글자 단위**로 ID 시퀀스로 변환
        
        Args:
            text (str): 입력 문장
        
        Returns:
            list: 토큰 ID 시퀀스 [SOS, ...chars..., EOS]
        """
        if not text:
            return []
            
        # 소문자 변환 후 글자 단위 분리
        chars = list(text.lower())
        ids = [self.word2idx['<SOS>']]
        
        for c in chars:
            ids.append(self.word2idx.get(c, self.word2idx['<UNK>']))
        
        ids.append(self.word2idx['<EOS>'])
        return ids
    
    def decode(self, ids):
        """
        토큰 ID 시퀀스를 다시 문장으로 변환
        
        Args:
            ids (list): 토큰 ID 시퀀스
        
        Returns:
            str: 복원된 문장
        """
        tokens = []
        for idx in ids:
            char = self.idx2word.get(idx, '<UNK>')
            if char not in ['<SOS>', '<EOS>', '<PAD>']:
                tokens.append(char)
        return "".join(tokens)  # 글자 단위이므로 공백 없이 연결
    
    def get_vocab_size(self):
        """어휘 크기 반환"""
        return len(self.word2idx)
    
    def save(self, filepath):
        """
        토크나이저 저장 (word2idx만 저장)
        
        Args:
            filepath (str): 저장 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # word2idx만 저장 (idx2word는 복원 가능)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'vocab_size': self.vocab_size
            }, f)
        
        if self.verbose:
            print(f"[Save] 토크나이저 저장 완료: {filepath}")
    
    @classmethod
    def load(cls, filepath, verbose=True):
        """
        토크나이저 로드
        
        Args:
            filepath (str): 로드 경로
            verbose (bool): 출력 메시지 표시 여부
        
        Returns:
            SimpleTokenizer: 로드된 토크나이저 인스턴스
        """
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 토크나이저 인스턴스 생성
        tokenizer = cls(vocab_size=data['vocab_size'], verbose=verbose)
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = {idx: word for word, idx in tokenizer.word2idx.items()}
        
        if verbose:
            print(f"[Load] 토크나이저 로드 완료: {filepath}")
            print(f"[Load] 어휘 크기: {tokenizer.get_vocab_size()}")
        
        return tokenizer


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 60)
    print("SimpleTokenizer 테스트")
    print("=" * 60)
    
    # 샘플 텍스트
    texts = [
        "this is a sample sentence",
        "hello world",
        "machine learning is fun",
        "this is another example"
    ]
    
    # 토크나이저 생성 및 학습
    tokenizer = SimpleTokenizer(vocab_size=100, verbose=True)
    tokenizer.build_vocab(texts)
    
    # 인코딩/디코딩 테스트
    test_text = "this is a sample sentence"
    print(f"\n원본: {test_text}")
    
    encoded = tokenizer.encode(test_text)
    print(f"인코딩: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"디코딩: {decoded}")
    
    # 저장/로드 테스트
    save_path = "test_tokenizer.pkl"
    tokenizer.save(save_path)
    
    loaded_tokenizer = SimpleTokenizer.load(save_path)
    
    # 로드된 토크나이저로 테스트
    test_encoded = loaded_tokenizer.encode(test_text)
    test_decoded = loaded_tokenizer.decode(test_encoded)
    print(f"\n로드 후 디코딩: {test_decoded}")
    
    print("\n SimpleTokenizer 테스트 완료!")

