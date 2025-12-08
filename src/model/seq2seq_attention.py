
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder_attention import DecoderAttention

class Seq2SeqAttention(nn.Module):
    """
    Attention 기반 Seq2Seq 모델
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, 
                 hidden_dim, num_layers=1, dropout=0.5, device=None):
        super(Seq2SeqAttention, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.device = device
        
        # Attention 모델에서는 보통 Bidirectional Encoder를 많이 쓰지만,
        # 여기서는 기존 구조와 비교를 위해 단방향 + 1 Layer 기준으로 작성
        # (Attention Decoder 구현상 num_layers=1 가정)
        
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers, # 보통 1
            dropout=dropout
        )
        
        self.decoder = DecoderAttention(
            vocab_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            enc_hidden_dim=hidden_dim, # enc, dec hidden 차원 같다고 가정
            dec_hidden_dim=hidden_dim, 
            num_layers=num_layers, # 추가됨
            dropout=dropout
        )
        
        self.pad_idx = 0
        self.sos_idx = 2
        self.eos_idx = 3
        
    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5, max_len=50):
        """
        Forward Pass
        
        Returns:
            outputs: [batch_size, tgt_len, vocab_size]
        """
        batch_size = src.size(0)
        
        # 1. Encoder 실행
        # encoder_outputs: [batch_size, src_len, hidden_dim]
        # hidden, cell: [num_layers, batch_size, hidden_dim]
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # 정답(tgt) 존재 여부에 따라 모드 결정
        if tgt is not None:
            return self._forward_train(src, tgt, hidden, cell, encoder_outputs, teacher_forcing_ratio)
        else:
            return self._forward_inference(src, hidden, cell, encoder_outputs, max_len)

    def _forward_train(self, src, tgt, hidden, cell, encoder_outputs, teacher_forcing_ratio):
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.tgt_vocab_size
        
        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size).to(self.device)
        
        # 첫 입력 <SOS>
        decoder_input = tgt[:, 0:1] # [batch, 1]
        
        for t in range(1, tgt_len):
            # Attention Decoder Forward
            # hidden, cell: [1, batch, hidden] (Layer size 1 가정)
            output, hidden, cell, _ = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            
            outputs[:, t-1] = output
            
            # Teacher Forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1) # [batch]
            
            decoder_input = tgt[:, t:t+1] if teacher_force else top1.unsqueeze(1)
            
        return outputs

    def _forward_inference(self, src, hidden, cell, encoder_outputs, max_len):
        batch_size = src.size(0)
        vocab_size = self.tgt_vocab_size
        
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
        predicted_ids = torch.zeros(batch_size, max_len, dtype=torch.long).to(self.device)
        attn_weights_list = []
        
        # 첫 입력 <SOS>
        decoder_input = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long).to(self.device)
        
        for t in range(max_len):
            output, hidden, cell, attn_weights = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            
            outputs[:, t] = output
            attn_weights_list.append(attn_weights.unsqueeze(1)) # [batch, 1, src_len]
            
            top1 = output.argmax(1)
            predicted_ids[:, t] = top1
            decoder_input = top1.unsqueeze(1)
            
            # EOS 체크 (배치 단위라 완벽하진 않음)
            if (top1 == self.eos_idx).all():
                break
                
        # (선택) Attention Weight도 반환 가능
        return outputs, predicted_ids
    
    def generate(self, src, max_len=50):
        """Inference wrapper"""
        self.eval()
        if src.dim() == 1: src = src.unsqueeze(0)
        
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)
            _, predicted_ids = self._forward_inference(src, hidden, cell, encoder_outputs, max_len)
            
        return predicted_ids
