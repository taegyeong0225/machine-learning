"""
Training Script for Spell Correction Seq2Seq Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.seq2seq import Seq2Seq
from src.model.seq2seq_attention import Seq2SeqAttention
from src.dataset.gec_dataset import GECDataset, collate_fn
from src.utils.tokenize import SimpleTokenizer
from src.utils.config import Config


def train_epoch(model, dataloader, criterion, optimizer, device, teacher_forcing_ratio=0.5):
    """
    한 에폭 학습
    
    Returns:
        float: 평균 loss
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        src = batch['src'].to(device)  # [batch_size, src_seq_len]
        tgt = batch['tgt'].to(device)  # [batch_size, tgt_seq_len]
        
        # Forward pass
        # outputs: [batch_size, tgt_seq_len-1, vocab_size]
        outputs = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # Loss 계산
        # tgt의 첫 번째 토큰(<SOS>)을 제외하고 사용
        tgt_input = tgt[:, 1:]  # [batch_size, tgt_seq_len-1]
        
        # Reshape for loss calculation
        outputs = outputs.reshape(-1, outputs.size(-1))  # [batch_size * (tgt_seq_len-1), vocab_size]
        tgt_input = tgt_input.reshape(-1)  # [batch_size * (tgt_seq_len-1)]
        
        # PAD 토큰은 loss 계산에서 제외
        loss = criterion(outputs, tgt_input)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (선택사항)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 로그 출력
        if (batch_idx + 1) % Config.LOG_INTERVAL == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    검증
    
    Returns:
        float: 평균 loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # Forward pass (teacher forcing ratio = 1.0 for validation)
            outputs = model(src, tgt, teacher_forcing_ratio=1.0)
            
            # Loss 계산
            tgt_input = tgt[:, 1:]
            outputs = outputs.reshape(-1, outputs.size(-1))
            tgt_input = tgt_input.reshape(-1)
            
            loss = criterion(outputs, tgt_input)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """체크포인트 저장"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"체크포인트 저장: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """체크포인트 로드"""
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"체크포인트 로드: {filepath}")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss

def greedy_decode(model, tokenizer, text, device, max_len=50):
    """간단한 greedy decoding (샘플 확인용)"""
    model.eval()
    src_ids = tokenizer.encode(text)
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)  # [1, src_len]
    
    with torch.no_grad():
        # model.generate 사용
        predicted_ids = model.generate(src_tensor, max_len=max_len)
        
    # [1, max_len] -> list
    pred_list = predicted_ids.squeeze(0).tolist()
    decoded = tokenizer.decode(pred_list)
    return decoded

def main():
    """메인 학습 함수"""
    print("=" * 60)
    print("Spell Correction Seq2Seq 모델 학습 (Vanilla LSTM)")
    print("=" * 60)
    
    # 강제로 Vanilla 모드로 설정
    Config.USE_ATTENTION = False
    
    # 설정 출력
    Config.print_config()
    
    # 디바이스 설정
    device = Config.DEVICE
    print(f"\n사용 디바이스: {device}")
    
    # 데이터 로드
    print("\n데이터 로드 중...")
    from datasets import load_from_disk
    
    if Config.TRAIN_DATA_PATH.exists():
        print(f"로컬 저장소에서 데이터 로드: {Config.TRAIN_DATA_PATH}")
        raw_train_data = load_from_disk(str(Config.TRAIN_DATA_PATH))
        
        # HuggingFace dataset을 list of dicts로 변환
        train_data = []
        for item in raw_train_data:
            if item['source'] and item['target']:
                train_data.append({
                    'source': item['source'], 
                    'target': item['target']
                })
        print(f"학습 데이터: {len(train_data)}개")
    else:
        print(f"저장된 데이터가 없습니다: {Config.TRAIN_DATA_PATH}")
        print("먼저 src/data_processing/prepare_data.py를 실행하세요.")
        return

    
    if Config.VAL_DATA_PATH.exists():
        print(f"검증 데이터 로드: {Config.VAL_DATA_PATH}")
        raw_val_data = load_from_disk(str(Config.VAL_DATA_PATH))
        val_data = [{'source': item['source'], 'target': item['target']} 
                   for item in raw_val_data if item['source'] and item['target']]
        print(f"검증 데이터: {len(val_data)}개")
    else:
        print("검증 데이터가 없습니다. 학습 데이터의 일부를 사용하거나 prepare_data.py를 다시 실행하세요.")
        val_data = []

    
    # 토크나이저 로드 또는 생성
    tokenizer_path = Config.TOKENIZER_PATH
    if tokenizer_path.exists():
        print(f"\n토크나이저 로드: {tokenizer_path}")
        tokenizer = SimpleTokenizer.load(tokenizer_path)
    else:
        print(f"\n토크나이저 생성 중...")
        tokenizer = SimpleTokenizer(vocab_size=Config.VOCAB_SIZE)
        all_texts = [ex['source'] for ex in train_data] + [ex['target'] for ex in train_data]
        tokenizer.build_vocab(all_texts)
        tokenizer.save(tokenizer_path)
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"어휘 크기: {vocab_size}")
    
    # 데이터셋 생성
    print("\n데이터셋 생성 중...")
    train_dataset = GECDataset(train_data, tokenizer, max_len=Config.MAX_LEN)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = None
    if val_data:
        val_dataset = GECDataset(val_data, tokenizer, max_len=Config.MAX_LEN)
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False, # 검증은 섞을 필요 없음
            collate_fn=collate_fn,
            num_workers=0
        )
    
    print(f"학습 데이터: {len(train_dataset)}개 샘플, {len(train_loader)}개 배치")
    if val_loader:
        print(f"검증 데이터: {len(val_dataset)}개 샘플, {len(val_loader)}개 배치")
    
    # 모델 생성
    print("\n모델 생성 중...")
    
    if Config.USE_ATTENTION:
        print("Using Attention-based Seq2Seq Model")
        model = Seq2SeqAttention(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT,
            device=device
        ).to(device)
    else:
        print("Using Vanilla Seq2Seq Model")
        model = Seq2Seq(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT,
            device=device
        ).to(device)
    
    # 파라미터 개수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터: {total_params:,}개 (학습 가능: {trainable_params:,}개)")
    
    # 체크포인트 디렉토리 확보
    checkpoint_dir = Config.get_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss 함수 및 Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD 토큰 무시
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 체크포인트 확인 및 로드 (자동 재개 기능)
    start_epoch = 1
    best_loss = float('inf')
    
    # 가장 최근 체크포인트 찾기
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoints:
        # epoch 번호 추출하여 정렬
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        print(f"\n>>>> 체크포인트 발견! 학습을 재개합니다: {latest_checkpoint}")
        
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        previous_loss = checkpoint['loss']
        print(f"  - 이전 Epoch: {checkpoint['epoch']} (Loss: {previous_loss:.4f})")
        print(f"  - 시작 Epoch: {start_epoch}")
        
        # Best loss 복원
        best_model_path = checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            best_ckpt = torch.load(best_model_path, map_location=device)
            best_loss = best_ckpt.get('loss', float('inf'))
            print(f"  - Best Loss 기록: {best_loss:.4f}")
    else:
        print("\n>>>> 새로운 학습을 시작합니다.")

    # 학습 루프
    print("\n" + "=" * 60)
    print(f"학습 시작 (Total Epochs: {Config.NUM_EPOCHS})")
    print("=" * 60)
    
    patience_counter = 0
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
        print("-" * 60)
        
        # 학습
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            teacher_forcing_ratio=Config.TEACHER_FORCING_RATIO
        )
        
        print(f"\nEpoch {epoch} 결과:")
        print(f"  Train Loss: {train_loss:.4f}")
        
        # 검증
        current_val_loss = train_loss # Val set 없으면 Train loss 사용
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"  Val Loss  : {val_loss:.4f}")
            current_val_loss = val_loss
        
        # 여기다가 inference 코드 넣기
        print("\n=== Sample Inference ===")
        sample_inputs = ["helo world", "I am hppy today"]
        for text in sample_inputs:
            out = greedy_decode(model, tokenizer, text, device)
            print(f"[Sample] {text} -> {out}")
        print("========================\n")

        # 체크포인트 저장
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        save_checkpoint(model, optimizer, epoch, current_val_loss, checkpoint_path)
        
        # Best model 저장 및 Early Stopping 체크
        if current_val_loss < best_loss:
            best_loss = current_val_loss
            best_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, current_val_loss, best_path)
            print(f"  Best model 저장 (Loss: {best_loss:.4f})")
            patience_counter = 0  # 성능 향상 시 카운터 초기화
        else:
            patience_counter += 1
            print(f"  성능 향상 없음 ({patience_counter}/{Config.PATIENCE})")
            
            if patience_counter >= Config.PATIENCE:
                print(f"\n 조기 종료 (Early Stopping): {Config.PATIENCE} Epoch 동안 성능 향상이 없어 학습을 중단합니다.")
                break
    
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

