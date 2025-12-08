
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Bahdanau Attention (Additive Attention)
    
    Decoder의 hidden state와 Encoder의 outputs 간의 관련성을 계산합니다.
    """
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super(Attention, self).__init__()
        
        # 가중치 계산을 위한 Linear Layers
        # W_a * s_{t-1} + U_a * h_j
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False) # 스칼라 점수 계산
        
    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: [batch_size, dec_hidden_dim] - Decoder의 현재(또는 이전) hidden state
            encoder_outputs: [batch_size, src_len, enc_hidden_dim] - Encoder의 모든 hidden states
            
        Returns:
            attention_weights: [batch_size, src_len] - 각 encoder output에 대한 가중치
        """
        src_len = encoder_outputs.shape[1]
        
        # hidden을 src_len만큼 반복해서 차원 맞춤
        # [batch_size, dec_hidden_dim] -> [batch_size, src_len, dec_hidden_dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Energy 계산 (tanh(W*h + U*s))
        # [batch_size, src_len, enc_hidden_dim + dec_hidden_dim]
        # -> [batch_size, src_len, dec_hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Attention Score 계산 (v^T * energy)
        # [batch_size, src_len, dec_hidden_dim] -> [batch_size, src_len, 1]
        attention = self.v(energy).squeeze(2)
        
        # Softmax를 적용하여 확률값(가중치)으로 변환
        # [batch_size, src_len]
        return F.softmax(attention, dim=1)
