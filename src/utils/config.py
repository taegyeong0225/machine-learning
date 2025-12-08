
import torch
from pathlib import Path

class Config:
    """
    Model and Training Configuration
    """
    # 프로젝트 루트 경로
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # 저장 경로
    # CHECKPOINT_DIR는 get_checkpoint_dir() 메서드로 대체
    TOKENIZER_PATH = PROJECT_ROOT / "saved" / "tokenizer.pkl"
    TRAIN_DATA_PATH = PROJECT_ROOT / "saved" / "data" / "train"
    VAL_DATA_PATH = PROJECT_ROOT / "saved" / "data" / "val"
    TEST_DATA_PATH = PROJECT_ROOT / "saved" / "data" / "test"
    
    # 데이터셋 설정
    DATASET_NAME = "vishnun/SpellGram"
    
    # 모델 하이퍼파라미터
    VOCAB_SIZE = 200  # Character-level이므로 작게 설정 (알파벳+숫자+특수문자)
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.5
    
    # 학습 하이퍼파라미터
    BATCH_SIZE = 32
    MAX_LEN = 50
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    TEACHER_FORCING_RATIO = 0.5
    PATIENCE = 3 # 성능 향상이 없을 때 기다려줄 Epoch 수
    USE_ATTENTION = True  # Attention 모델 사용 여부
    
    PATIENCE = 3 # 성능 향상이 없을 때 기다려줄 Epoch 수
    USE_ATTENTION = True  # Attention 모델 사용 여부
    
    # 로깅 설정
    LOG_INTERVAL = 10
    
    # 디바이스 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def get_checkpoint_dir(cls):
        """모델 타입에 따른 체크포인트 디렉토리 반환"""
        if cls.USE_ATTENTION:
            return cls.PROJECT_ROOT / "saved" / "checkpoints_attention"
        else:
            return cls.PROJECT_ROOT / "saved" / "checkpoints_vanilla"
    
    @classmethod
    def print_config(cls):
        print("\n" + "=" * 30)
        print("Configuration")
        print("=" * 30)
        print(f"Device: {cls.DEVICE}")
        print(f"Dataset: {cls.DATASET_NAME}")
        print(f"Model Type: {'Attention Seq2Seq' if cls.USE_ATTENTION else 'Vanilla Seq2Seq'}")
        print(f"Vocab Size: {cls.VOCAB_SIZE}")
        print(f"Embedding Dim: {cls.EMBEDDING_DIM}")
        print(f"Hidden Dim: {cls.HIDDEN_DIM}")
        print(f"Num Layers: {cls.NUM_LAYERS}")
        print(f"Max Length: {cls.MAX_LEN}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Checkpoint Dir: {cls.get_checkpoint_dir()}")
        print("=" * 30)

# 체크포인트 디렉토리 생성 (사용 시점에 생성하도록 변경)
# Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
