"""
LSTM Encoder for Spell Correction Seq2Seq Model

오타 문장(source)을 인코딩하여 문맥 정보를 압축한 hidden state와 cell state를 생성합니다.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    LSTM 기반 Encoder
    
    Args:
        vocab_size (int): 입력 어휘 크기
        embedding_dim (int): 임베딩 차원
        hidden_dim (int): LSTM hidden state 차원
        num_layers (int): LSTM 레이어 수
        dropout (float): Dropout 비율 (default: 0.5)
    
    Input:
        src: [batch_size, seq_len] - 입력 문장 (토큰 ID)
    
    Output:
        outputs: [batch_size, seq_len, hidden_dim] - 모든 타임스텝의 출력
        hidden: [num_layers, batch_size, hidden_dim] - 최종 hidden state
        cell: [num_layers, batch_size, hidden_dim] - 최종 cell state
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding 레이어
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # <PAD> 토큰의 인덱스
        )
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # 1층이면 dropout 불필요
            batch_first=True  # [batch, seq, feature] 순서
        )
        
        # Dropout 레이어
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, src):
        """
        Forward pass
        
        Args:
            src: [batch_size, seq_len] - 입력 토큰 시퀀스
        
        Returns:
            outputs: [batch_size, seq_len, hidden_dim] - LSTM 출력
            hidden: [num_layers, batch_size, hidden_dim] - 마지막 hidden state
            cell: [num_layers, batch_size, hidden_dim] - 마지막 cell state
        """
        # 1. Embedding
        # [batch_size, seq_len] → [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(src)
        
        # 2. Dropout 적용
        embedded = self.dropout_layer(embedded)
        
        # 3. LSTM
        # outputs: [batch_size, seq_len, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell: [num_layers, batch_size, hidden_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, hidden, cell


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 60)
    print("Encoder 테스트")
    print("=" * 60)
    
    # 하이퍼파라미터
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    SEQ_LEN = 20
    
    # Encoder 인스턴스 생성
    encoder = Encoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.5
    )
    
    # 더미 입력 생성 (랜덤 토큰 ID)
    src = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    print(f"\n입력 shape: {src.shape}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Sequence length: {SEQ_LEN}")
    
    # Forward pass
    outputs, hidden, cell = encoder(src)
    
    print(f"\n출력:")
    print(f"  - outputs shape: {outputs.shape}")
    print(f"  - hidden shape: {hidden.shape}")
    print(f"  - cell shape: {cell.shape}")
    
    # 파라미터 개수 확인
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\n모델 파라미터:")
    print(f"  - 전체: {total_params:,}")
    print(f"  - 학습 가능: {trainable_params:,}")
    
    print("\nEncoder 테스트 완료!")
