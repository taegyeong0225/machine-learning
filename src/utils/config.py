
import torch
from pathlib import Path

class Config:
    """
    Model and Training Configuration
    """
    # 프로젝트 루트 경로
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # 저장 경로
    CHECKPOINT_DIR = PROJECT_ROOT / "saved" / "checkpoints"
    TOKENIZER_PATH = PROJECT_ROOT / "saved" / "tokenizer.pkl"
    TRAIN_DATA_PATH = PROJECT_ROOT / "saved" / "data" / "train"
    VAL_DATA_PATH = PROJECT_ROOT / "saved" / "data" / "val"
    TEST_DATA_PATH = PROJECT_ROOT / "saved" / "data" / "test"
    
    # 데이터셋 설정
    DATASET_NAME = "vishnun/SpellGram"
    
    # 모델 하이퍼파라미터
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.5
    
    # 학습 하이퍼파라미터
    BATCH_SIZE = 32
    MAX_LEN = 50
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    TEACHER_FORCING_RATIO = 0.5
    
    # 로깅 설정
    LOG_INTERVAL = 10
    
    # 디바이스 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def print_config(cls):
        print("\n" + "=" * 30)
        print("Configuration")
        print("=" * 30)
        print(f"Device: {cls.DEVICE}")
        print(f"Dataset: {cls.DATASET_NAME}")
        print(f"Vocab Size: {cls.VOCAB_SIZE}")
        print(f"Embedding Dim: {cls.EMBEDDING_DIM}")
        print(f"Hidden Dim: {cls.HIDDEN_DIM}")
        print(f"Num Layers: {cls.NUM_LAYERS}")
        print(f"Max Length: {cls.MAX_LEN}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print("=" * 30)

# 체크포인트 디렉토리 생성
Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
