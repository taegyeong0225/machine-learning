
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def load_losses(checkpoint_dir):
    """
    체크포인트 디렉토리에서 Epoch별 Loss를 추출
    """
    losses = {}
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"Warnings: {checkpoint_dir} does not exist.")
        return [], []

    files = list(checkpoint_path.glob("checkpoint_epoch_*.pt"))
    
    for f in files:
        try:
            # 파일명에서 epoch 추출 (checkpoint_epoch_1.pt)
            epoch = int(f.stem.split('_')[-1])
            # CPU로 로드 (GPU 메모리 절약)
            checkpoint = torch.load(f, map_location='cpu')
            loss = checkpoint['loss']
            losses[epoch] = loss
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    # Epoch 순서대로 정렬
    sorted_epochs = sorted(losses.keys())
    sorted_losses = [losses[e] for e in sorted_epochs]
    
    return sorted_epochs, sorted_losses

def main():
    vanilla_dir = project_root / "saved" / "checkpoints_vanilla"
    attention_dir = project_root / "saved" / "checkpoints_attention"
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    print("Loading process logs...")
    v_epochs, v_losses = load_losses(vanilla_dir)
    a_epochs, a_losses = load_losses(attention_dir)
    
    if not v_epochs and not a_epochs:
        print("No checkpoints found.")
        return

    plt.figure(figsize=(10, 6))
    
    if v_epochs:
        plt.plot(v_epochs, v_losses, marker='o', linestyle='-', label='Vanilla LSTM', color='blue')
        print(f"Vanilla LSTM: {len(v_epochs)} epochs loaded.")
        
    if a_epochs:
        plt.plot(a_epochs, a_losses, marker='s', linestyle='--', label='Attention LSTM', color='red')
        print(f"Attention LSTM: {len(a_epochs)} epochs loaded.")
        
    plt.title("Training Loss Comparison: Vanilla vs Attention")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Cross Entropy)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_path = output_dir / "loss_comparison.png"
    plt.savefig(output_path)
    print(f"\nGraph saved to: {output_path}")
    # plt.show() # 서버 환경 등에서는 주석 처리

if __name__ == "__main__":
    main()
