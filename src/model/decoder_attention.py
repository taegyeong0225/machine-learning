
import torch
import torch.nn as nn
from .attention import Attention

class DecoderAttention(nn.Module):
    """
    Attention 기반 Decoder
    
    Args:
        vocab_size (int): 출력 어휘 크기
        embedding_dim (int): 임베딩 차원
        enc_hidden_dim (int): Encoder hidden state 차원
        dec_hidden_dim (int): Decoder hidden state 차원
        num_layers (int): LSTM 레이어 수
        dropout (float): Dropout 비율
    """
    def __init__(self, vocab_size, embedding_dim, enc_hidden_dim, dec_hidden_dim, num_layers=1, dropout=0.5):
        super(DecoderAttention, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Attention 모듈
        self.attention = Attention(enc_hidden_dim, dec_hidden_dim)
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM Layer
        # num_layers 추가
        self.lstm = nn.LSTM(
            input_size=enc_hidden_dim + embedding_dim, 
            hidden_size=dec_hidden_dim, 
            num_layers=num_layers, # 여기 수정됨
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output Layer
        # 입력: Decoder hidden + Context vector + Embedding
        self.fc_out = nn.Linear(enc_hidden_dim + dec_hidden_dim + embedding_dim, vocab_size)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, cell, encoder_outputs):
        """
        한 타임스텝 Forward Pass
        
        Args:
            input_token: [batch_size, 1] - 입력 토큰
            hidden: [1, batch_size, dec_hidden_dim] - 이전 hidden state (LSTM 입력용)
            cell: [1, batch_size, dec_hidden_dim] - 이전 cell state
            encoder_outputs: [batch_size, src_len, enc_hidden_dim] - Encoder 출력 전체
            
        Returns:
            prediction: [batch_size, vocab_size]
            hidden: [1, batch_size, dec_hidden_dim]
            cell: [1, batch_size, dec_hidden_dim]
        """
        # 1. Embedding
        # [batch_size, 1] -> [batch_size, 1, embedding_dim]
        embedded = self.dropout_layer(self.embedding(input_token))
        
        # 2. Attention Weight 계산
        # hidden: [num_layers, batch_size, dec_hidden_dim]
        # Attention에는 마지막 레이어의 Hidden State를 사용 (hidden[-1])
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        
        # 3. Context Vector 계산 (Attention Weight * Encoder Outputs)
        # attn_weights: [batch_size, 1, src_len]
        attn_weights = attn_weights.unsqueeze(1)
        
        # [batch_size, 1, src_len] batch_matmul [batch_size, src_len, enc_hidden_dim]
        # -> [batch_size, 1, enc_hidden_dim]
        context = torch.bmm(attn_weights, encoder_outputs)
        
        # 4. LSTM Input 구성 (Embedding + Context)
        # [batch_size, 1, embedding_dim + enc_hidden_dim]
        lstm_input = torch.cat((embedded, context), dim=2)
        
        # 5. LSTM 실행
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # 6. 최종 예측 (hidden state + context + embedding을 모두 활용하기도 함)
        # 여기서는 concat(embedded, output, context) -> FC -> vocab
        # output: [batch_size, 1, dec_hidden_dim]
        
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        context = context.squeeze(1)
        
        # [batch_size, embedding_dim + dec_hidden_dim + enc_hidden_dim]
        prediction_input = torch.cat((output, context, embedded), dim=1)
        
        prediction = self.fc_out(prediction_input)
        
        return prediction, hidden, cell, attn_weights.squeeze(1)
