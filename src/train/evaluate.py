"""
Evaluation Script for Spell Correction Seq2Seq Model

평가 지표:
- Character-level Accuracy
- Word-level Accuracy
- BLEU Score
- CER (Character Error Rate)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path
from collections import Counter

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.seq2seq import Seq2Seq
from src.dataset.gec_dataset import GECDataset, collate_fn
from src.utils.tokenize import SimpleTokenizer
from src.utils.config import Config


def calculate_character_accuracy(pred_text, target_text):
    """
    Character-level Accuracy 계산
    
    Returns:
        float: 정확도 (0.0 ~ 1.0)
    """
    pred_chars = list(pred_text.replace(' ', ''))
    target_chars = list(target_text.replace(' ', ''))
    
    if len(target_chars) == 0:
        return 1.0 if len(pred_chars) == 0 else 0.0
    
    correct = sum(1 for p, t in zip(pred_chars, target_chars) if p == t)
    return correct / len(target_chars)


def calculate_word_accuracy(pred_text, target_text):
    """
    Word-level Accuracy 계산
    
    Returns:
        float: 정확도 (0.0 ~ 1.0)
    """
    pred_words = pred_text.split()
    target_words = target_text.split()
    
    if len(target_words) == 0:
        return 1.0 if len(pred_words) == 0 else 0.0
    
    correct = sum(1 for p, t in zip(pred_words, target_words) if p == t)
    return correct / len(target_words)


def calculate_cer(pred_text, target_text):
    """
    Character Error Rate (CER) 계산
    
    Returns:
        float: CER (낮을수록 좋음)
    """
    from Levenshtein import distance as levenshtein_distance
    
    pred_chars = list(pred_text.replace(' ', ''))
    target_chars = list(target_text.replace(' ', ''))
    
    if len(target_chars) == 0:
        return 0.0 if len(pred_chars) == 0 else float('inf')
    
    edit_distance = levenshtein_distance(''.join(pred_chars), ''.join(target_chars))
    return edit_distance / len(target_chars)


def calculate_bleu_score(pred_text, target_text, n=4):
    """
    BLEU Score 계산 (간단한 버전)
    
    Args:
        pred_text (str): 예측 문장
        target_text (str): 정답 문장
        n (int): n-gram 최대 길이
    
    Returns:
        float: BLEU Score (0.0 ~ 1.0)
    """
    pred_words = pred_text.split()
    target_words = target_text.split()
    
    if len(pred_words) == 0 or len(target_words) == 0:
        return 0.0
    
    # Precision 계산
    precisions = []
    for i in range(1, n + 1):
        pred_ngrams = Counter(zip(*[pred_words[j:] for j in range(i)]))
        target_ngrams = Counter(zip(*[target_words[j:] for j in range(i)]))
        
        matches = sum((pred_ngrams & target_ngrams).values())
        total = sum(pred_ngrams.values())
        
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)
    
    # Brevity penalty
    if len(pred_words) < len(target_words):
        bp = 1.0
    else:
        bp = min(1.0, len(target_words) / len(pred_words))
    
    # BLEU Score
    if any(p == 0 for p in precisions):
        return 0.0
    
    bleu = bp * (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
    return bleu


def evaluate_model(model, dataloader, tokenizer, device, max_samples=None):
    """
    모델 평가
    
    Args:
        model: Seq2Seq 모델
        dataloader: DataLoader
        tokenizer: SimpleTokenizer
        device: torch.device
        max_samples (int): 최대 평가 샘플 수 (None이면 전체)
    
    Returns:
        dict: 평가 결과
    """
    model.eval()
    
    char_accuracies = []
    word_accuracies = []
    cers = []
    bleu_scores = []
    
    all_predictions = []
    all_targets = []
    all_sources = []
    
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_samples and num_samples >= max_samples:
                break
            
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # 예측 생성
            batch_size = src.size(0)
            predictions = []
            
            for i in range(batch_size):
                src_seq = src[i:i+1]  # [1, seq_len]
                
                # Encoder forward 먼저 수행
                encoder_outputs, hidden, cell = model.encoder(src_seq)
                _, predicted_ids = model._forward_inference(
                    src_seq, hidden, cell, max_len=Config.MAX_LEN
                )
                
                # 토큰 ID를 텍스트로 변환
                pred_ids = predicted_ids[0].cpu().tolist()
                # EOS 토큰까지만 사용
                eos_token_id = tokenizer.word2idx.get('<EOS>', 3)
                if eos_token_id in pred_ids:
                    eos_idx = pred_ids.index(eos_token_id)
                    pred_ids = pred_ids[:eos_idx]
                
                pred_text = tokenizer.decode(pred_ids)
                predictions.append(pred_text)
            
            # 정답 텍스트 추출
            targets = batch['tgt_text']
            sources = batch['src_text']
            
            # 각 샘플에 대해 메트릭 계산
            for pred, target, source in zip(predictions, targets, sources):
                char_acc = calculate_character_accuracy(pred, target)
                word_acc = calculate_word_accuracy(pred, target)
                cer = calculate_cer(pred, target)
                bleu = calculate_bleu_score(pred, target)
                
                char_accuracies.append(char_acc)
                word_accuracies.append(word_acc)
                cers.append(cer)
                bleu_scores.append(bleu)
                
                all_predictions.append(pred)
                all_targets.append(target)
                all_sources.append(source)
                
                num_samples += 1
                
                if max_samples and num_samples >= max_samples:
                    break
    
    # 평균 계산
    results = {
        'char_accuracy': sum(char_accuracies) / len(char_accuracies) if char_accuracies else 0.0,
        'word_accuracy': sum(word_accuracies) / len(word_accuracies) if word_accuracies else 0.0,
        'cer': sum(cers) / len(cers) if cers else float('inf'),
        'bleu_score': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        'num_samples': num_samples,
        'predictions': all_predictions,
        'targets': all_targets,
        'sources': all_sources
    }
    
    return results


def print_evaluation_results(results, num_examples=5):
    """평가 결과 출력"""
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"평가 샘플 수: {results['num_samples']}")
    print(f"\nCharacter-level Accuracy: {results['char_accuracy']:.4f}")
    print(f"Word-level Accuracy: {results['word_accuracy']:.4f}")
    print(f"CER (Character Error Rate): {results['cer']:.4f}")
    print(f"BLEU Score: {results['bleu_score']:.4f}")
    
    # 샘플 예시 출력
    print("\n" + "-" * 60)
    print("샘플 예시:")
    print("-" * 60)
    
    for i in range(min(num_examples, len(results['sources']))):
        print(f"\n예시 {i+1}:")
        print(f"  입력 (오타): {results['sources'][i]}")
        print(f"  예측:        {results['predictions'][i]}")
        print(f"  정답:        {results['targets'][i]}")
        
        # 정확도 표시
        char_acc = calculate_character_accuracy(
            results['predictions'][i], 
            results['targets'][i]
        )
        word_acc = calculate_word_accuracy(
            results['predictions'][i], 
            results['targets'][i]
        )
        print(f"  Char Acc: {char_acc:.4f}, Word Acc: {word_acc:.4f}")


def main():
    """메인 평가 함수"""
    print("=" * 60)
    print("Spell Correction Seq2Seq 모델 평가")
    print("=" * 60)
    
    # 디바이스 설정
    device = Config.DEVICE
    print(f"\n사용 디바이스: {device}")
    
    # 체크포인트 로드
    checkpoint_path = Config.CHECKPOINT_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"\n❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        print("먼저 학습을 실행하세요.")
        return
    
    # 토크나이저 로드
    tokenizer_path = Config.TOKENIZER_PATH
    if not tokenizer_path.exists():
        print(f"\n❌ 토크나이저를 찾을 수 없습니다: {tokenizer_path}")
        return
    
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # 모델 생성
    model = Seq2Seq(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        device=device
    ).to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✅ 모델 로드 완료: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    # 데이터 로드 (예시 - 실제 데이터셋 경로로 변경 필요)
    print("\n데이터 로드 중...")
    # TODO: 실제 테스트 데이터 로드
    test_data = [
        {'source': 'this is a smple sentence', 'target': 'this is a sample sentence'},
        {'source': 'helo world', 'target': 'hello world'},
        {'source': 'machne lernig', 'target': 'machine learning'},
    ] * 10  # 임시 데이터
    
    # 데이터셋 생성
    test_dataset = GECDataset(test_data, tokenizer, max_len=Config.MAX_LEN)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"테스트 데이터: {len(test_dataset)}개 샘플")
    
    # 평가 실행
    print("\n평가 실행 중...")
    results = evaluate_model(model, test_loader, tokenizer, device)
    
    # 결과 출력
    print_evaluation_results(results)


if __name__ == "__main__":
    main()

