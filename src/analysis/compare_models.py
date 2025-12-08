
import torch
import sys
from pathlib import Path
import pandas as pd

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.seq2seq import Seq2Seq
from src.model.seq2seq_attention import Seq2SeqAttention
from src.utils.tokenize import SimpleTokenizer
from src.utils.config import Config

def load_model(model_type, checkpoint_dir, vocab_size, device):
    """모델 로드 헬퍼 함수"""
    best_path = checkpoint_dir / "best_model.pt"
    if not best_path.exists():
        # best_model 없으면 가장 마지막 epoch 로드
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None
        best_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    
    print(f"Loading {model_type} from {best_path}...")
    
    if model_type == 'attention':
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
        model = Seq2Seq(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT,
            device=device
        ).to(device)
        
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def greedy_decode(model, tokenizer, text, device, max_len=50):
    model.eval()
    src_ids = tokenizer.encode(text)
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if isinstance(model, Seq2SeqAttention):
             # Attention 모델은 generate 메서드 사용
            predicted_ids = model.generate(src_tensor, max_len=max_len)
        else:
            # Vanilla 모델은 내부 로직이 조금 다름 (기존 train.py 참조)
            # 여기서는 편의상 forward_inference 직접 호출 또는 모델에 generate 구현 가정
            # Vanilla Seq2Seq에도 generate 메서드가 없다면 추가 필요.
            # 하지만 src/model/seq2seq.py 에는 generate가 구현되어 있음.
            predicted_ids = model.generate(src_tensor, max_len=max_len)
            
    pred_list = predicted_ids.squeeze(0).tolist()
    decoded = tokenizer.decode(pred_list)
    return decoded

def main():
    device = torch.device('cpu') # Inference는 CPU로 충분
    tokenizer_path = Config.TOKENIZER_PATH
    
    if not tokenizer_path.exists():
        print("Tokenizer not found.")
        return
        
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # 모델 로드
    vanilla_model = load_model('vanilla', project_root / "saved" / "checkpoints_vanilla", vocab_size, device)
    attention_model = load_model('attention', project_root / "saved" / "checkpoints_attention", vocab_size, device)
    
    # 테스트 문장 (오타 포함)
    test_sentences = [
        "helo world",
        "I am hppy today",
        "Ths is a tst message",
        "machne learnin is fun",
        "please correc this sentense",
        "waht is your name",
        "I want to go to scool"
    ]
    
    results = []
    
    print("\nRunning Inference...")
    for text in test_sentences:
        row = {"Input": text}
        
        if vanilla_model:
            out_v = greedy_decode(vanilla_model, tokenizer, text, device)
            row["Vanilla LSTM"] = out_v
        else:
            row["Vanilla LSTM"] = "N/A"
            
        if attention_model:
            out_a = greedy_decode(attention_model, tokenizer, text, device)
            row["Attention LSTM"] = out_a
        else:
            row["Attention LSTM"] = "N/A"
            
        results.append(row)
        print(f"Input: {text}")
        print(f"  Vanilla: {row['Vanilla LSTM']}")
        print(f"  Attention: {row['Attention LSTM']}")
    
    # 결과 저장
    df = pd.DataFrame(results)
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Markdown으로 저장
    md_path = output_dir / "model_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Model Comparison Results\n\n")
        f.write(df.to_markdown(index=False))
        
    print(f"\nComparison table saved to: {md_path}")

if __name__ == "__main__":
    main()
