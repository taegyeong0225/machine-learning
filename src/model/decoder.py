"""
LSTM Decoder for Spell Correction Seq2Seq Model

Encoder의 context를 받아 한 토큰씩 순차적으로 정답 문장을 생성합니다.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    LSTM 기반 Decoder
    
    Args:
        vocab_size (int): 출력 어휘 크기
        embedding_dim (int): 임베딩 차원
        hidden_dim (int): LSTM hidden state 차원
        num_layers (int): LSTM 레이어 수
        dropout (float): Dropout 비율 (default: 0.5)
    
    Input:
        tgt: [batch_size, 1] - 이전 타임스텝의 출력 토큰 (teacher forcing)
        hidden: [num_layers, batch_size, hidden_dim] - Encoder의 hidden state
        cell: [num_layers, batch_size, hidden_dim] - Encoder의 cell state
    
    Output:
        output: [batch_size, vocab_size] - 다음 토큰 예측 확률
        hidden: [num_layers, batch_size, hidden_dim] - 업데이트된 hidden state
        cell: [num_layers, batch_size, hidden_dim] - 업데이트된 cell state
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding 레이어
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # <PAD> 토큰
        )
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 출력 레이어 (hidden → vocab_size)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout 레이어
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, tgt, hidden, cell):
        """
        Forward pass (한 타임스텝)
        
        Args:
            tgt: [batch_size, 1] - 입력 토큰 (이전 출력 또는 정답)
            hidden: [num_layers, batch_size, hidden_dim] - 이전 hidden state
            cell: [num_layers, batch_size, hidden_dim] - 이전 cell state
        
        Returns:
            output: [batch_size, vocab_size] - 다음 토큰 예측 로짓
            hidden: [num_layers, batch_size, hidden_dim] - 업데이트된 hidden
            cell: [num_layers, batch_size, hidden_dim] - 업데이트된 cell
        """
        # 1. Embedding
        # [batch_size, 1] → [batch_size, 1, embedding_dim]
        embedded = self.embedding(tgt)
        
        # 2. Dropout
        embedded = self.dropout_layer(embedded)
        
        # 3. LSTM (한 타임스텝)
        # output: [batch_size, 1, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell: [num_layers, batch_size, hidden_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # 4. 출력 레이어
        # [batch_size, 1, hidden_dim] → [batch_size, 1, vocab_size]
        prediction = self.fc_out(output)
        
        # 5. squeeze로 차원 축소
        # [batch_size, 1, vocab_size] → [batch_size, vocab_size]
        prediction = prediction.squeeze(1)
        
        return prediction, hidden, cell


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 60)
    print("Decoder 테스트")
    print("=" * 60)
    
    # 하이퍼파라미터
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    
    # Decoder 인스턴스 생성
    decoder = Decoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.5
    )
    
    # 더미 입력 생성
    tgt = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1))  # 한 타임스텝
    hidden = torch.randn(NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM)
    cell = torch.randn(NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM)
    
    print(f"\n입력:")
    print(f"  - tgt shape: {tgt.shape}")
    print(f"  - hidden shape: {hidden.shape}")
    print(f"  - cell shape: {cell.shape}")
    
    # Forward pass
    output, hidden_out, cell_out = decoder(tgt, hidden, cell)
    
    print(f"\n출력:")
    print(f"  - output shape: {output.shape}")
    print(f"  - hidden shape: {hidden_out.shape}")
    print(f"  - cell shape: {cell_out.shape}")
    
    # Softmax 적용하여 확률로 변환
    probs = torch.softmax(output, dim=-1)
    print(f"  - 확률 분포 shape: {probs.shape}")
    print(f"  - 확률 합 (첫 번째 배치): {probs[0].sum():.4f}")
    
    # 가장 높은 확률의 토큰 선택
    predicted_tokens = output.argmax(dim=-1)
    print(f"  - 예측 토큰 shape: {predicted_tokens.shape}")
    print(f"  - 예측 토큰 예시: {predicted_tokens[:5]}")
    
    # 파라미터 개수 확인
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    print(f"\n모델 파라미터:")
    print(f"  - 전체: {total_params:,}")
    print(f"  - 학습 가능: {trainable_params:,}")
    
    print("\n✅ Decoder 테스트 완료!")
