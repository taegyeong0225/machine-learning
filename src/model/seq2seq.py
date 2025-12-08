"""
Seq2Seq Model for Spell Correction

Encoder와 Decoder를 결합한 전체 Seq2Seq 모델입니다.
Teacher Forcing을 지원하며, 학습 시와 추론 시의 동작이 다릅니다.
"""

import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class Seq2Seq(nn.Module):
    """
    LSTM 기반 Seq2Seq 모델
    
    Args:
        src_vocab_size (int): 입력 어휘 크기
        tgt_vocab_size (int): 출력 어휘 크기 (보통 같음)
        embedding_dim (int): 임베딩 차원
        hidden_dim (int): LSTM hidden state 차원
        num_layers (int): LSTM 레이어 수
        dropout (float): Dropout 비율
        device (torch.device): 사용할 디바이스
    
    Input:
        src: [batch_size, src_seq_len] - 입력 문장 (오타)
        tgt: [batch_size, tgt_seq_len] - 정답 문장 (teacher forcing용)
        teacher_forcing_ratio (float): Teacher forcing 비율 (0.0 ~ 1.0)
    
    Output:
        outputs: [batch_size, tgt_seq_len, vocab_size] - 각 타임스텝의 예측 로짓
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, 
                 hidden_dim, num_layers, dropout=0.5, device=None):
        super(Seq2Seq, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Encoder와 Decoder 생성
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 특수 토큰 인덱스 (일반적으로 0: PAD, 2: SOS, 3: EOS)
        self.pad_idx = 0
        self.sos_idx = 2
        self.eos_idx = 3
    
    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5, max_len=50):
        """
        Forward pass
        
        Args:
            src: [batch_size, src_seq_len] - 입력 문장
            tgt: [batch_size, tgt_seq_len] - 정답 문장 (학습 시 사용)
            teacher_forcing_ratio (float): Teacher forcing 비율
            max_len (int): 최대 생성 길이 (추론 시 사용)
        
        Returns:
            outputs: [batch_size, tgt_seq_len, vocab_size] - 예측 로짓
        """
        batch_size = src.size(0)
        
        # Encoder forward
        # outputs: [batch_size, src_seq_len, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell: [num_layers, batch_size, hidden_dim]
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # 정답(tgt)이 주어지면 Loss 계산을 위해 forward_train 사용 (학습/검증 공통)
        if tgt is not None:
            # 학습/검증 모드: Teacher Forcing 비율에 따라 동작
            # 검증 시에는 teacher_forcing_ratio=1.0으로 설정하여 정답을 넣어줌
            return self._forward_train(src, tgt, hidden, cell, teacher_forcing_ratio)
        else:
            # 추론 모드: Greedy Decoding (정답 없음)
            return self._forward_inference(src, hidden, cell, max_len)
    
    def _forward_train(self, src, tgt, hidden, cell, teacher_forcing_ratio):
        """
        학습 시 forward (Teacher Forcing)
        
        Args:
            src: [batch_size, src_seq_len]
            tgt: [batch_size, tgt_seq_len] - 정답 문장
            hidden: [num_layers, batch_size, hidden_dim]
            cell: [num_layers, batch_size, hidden_dim]
            teacher_forcing_ratio (float): Teacher forcing 비율
        
        Returns:
            outputs: [batch_size, tgt_seq_len-1, vocab_size]
        """
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.tgt_vocab_size
        
        # 출력 저장용
        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size).to(self.device)
        
        # 첫 번째 입력은 <SOS> 토큰
        # tgt의 첫 번째 토큰이 <SOS>이므로, 두 번째부터 사용
        decoder_input = tgt[:, 0:1]  # [batch_size, 1]
        
        # 각 타임스텝마다 디코딩
        for t in range(1, tgt_len):
            # Decoder forward (한 타임스텝)
            # output: [batch_size, vocab_size]
            # hidden: [num_layers, batch_size, hidden_dim]
            # cell: [num_layers, batch_size, hidden_dim]
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            # 출력 저장
            outputs[:, t - 1] = output
            
            # Teacher Forcing 결정
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                # 정답 토큰 사용
                decoder_input = tgt[:, t:t+1]  # [batch_size, 1]
            else:
                # 모델이 예측한 토큰 사용
                decoder_input = output.argmax(dim=-1, keepdim=True)  # [batch_size, 1]
        
        return outputs
    
    def _forward_inference(self, src, hidden, cell, max_len):
        """
        추론 시 forward (Greedy Decoding)
        
        Args:
            src: [batch_size, src_seq_len]
            hidden: [num_layers, batch_size, hidden_dim]
            cell: [num_layers, batch_size, hidden_dim]
            max_len (int): 최대 생성 길이
        
        Returns:
            outputs: [batch_size, max_len, vocab_size]
            predicted_ids: [batch_size, max_len] - 예측된 토큰 ID 시퀀스
        """
        batch_size = src.size(0)
        vocab_size = self.tgt_vocab_size
        
        # 출력 저장용
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
        predicted_ids = torch.zeros(batch_size, max_len, dtype=torch.long).to(self.device)
        
        # 첫 번째 입력은 <SOS> 토큰
        decoder_input = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long).to(self.device)
        
        # 각 타임스텝마다 디코딩
        for t in range(max_len):
            # Decoder forward (한 타임스텝)
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            # 출력 저장
            outputs[:, t] = output
            
            # 가장 높은 확률의 토큰 선택
            predicted_token = output.argmax(dim=-1, keepdim=True)  # [batch_size, 1]
            predicted_ids[:, t] = predicted_token.squeeze(1)
            
            # 다음 입력으로 사용
            decoder_input = predicted_token
            
            # 모든 배치에서 <EOS>가 나왔는지 확인 (조기 종료)
            if (predicted_token == self.eos_idx).all():
                break
        
        return outputs, predicted_ids
    
    def generate(self, src, max_len=50):
        """
        문장 생성 (추론용 편의 함수)
        
        Args:
            src: [batch_size, src_seq_len] 또는 [src_seq_len] - 입력 문장
            max_len (int): 최대 생성 길이
        
        Returns:
            predicted_ids: [batch_size, max_len] - 예측된 토큰 ID 시퀀스
        """
        self.eval()
        
        # 배치 차원이 없으면 추가
        if src.dim() == 1:
            src = src.unsqueeze(0)
        
        with torch.no_grad():
            # Encoder forward를 먼저 수행
            _, hidden, cell = self.encoder(src)
            _, predicted_ids = self._forward_inference(src, hidden, cell, max_len)
        
        return predicted_ids


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 60)
    print("Seq2Seq 모델 테스트")
    print("=" * 60)
    
    # 하이퍼파라미터
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    SRC_LEN = 20
    TGT_LEN = 25
    
    # 모델 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.5,
        device=device
    ).to(device)
    
    # 더미 입력 생성
    src = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SRC_LEN)).to(device)
    tgt = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, TGT_LEN)).to(device)
    # 첫 번째 토큰을 <SOS>로 설정
    tgt[:, 0] = 2  # SOS
    
    print(f"\n입력:")
    print(f"  - src shape: {src.shape}")
    print(f"  - tgt shape: {tgt.shape}")
    
    # 학습 모드 테스트
    model.train()
    outputs = model(src, tgt, teacher_forcing_ratio=0.5)
    print(f"\n학습 모드 출력:")
    print(f"  - outputs shape: {outputs.shape}")
    
    # 추론 모드 테스트
    model.eval()
    with torch.no_grad():
        outputs, predicted_ids = model._forward_inference(
            src, 
            torch.randn(NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM).to(device),
            torch.randn(NUM_LAYERS, BATCH_SIZE, HIDDEN_DIM).to(device),
            max_len=30
        )
        print(f"\n추론 모드 출력:")
        print(f"  - outputs shape: {outputs.shape}")
        print(f"  - predicted_ids shape: {predicted_ids.shape}")
    
    # 파라미터 개수 확인
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n모델 파라미터:")
    print(f"  - 전체: {total_params:,}")
    print(f"  - 학습 가능: {trainable_params:,}")
    
    print("\nSeq2Seq 모델 테스트 완료!")

