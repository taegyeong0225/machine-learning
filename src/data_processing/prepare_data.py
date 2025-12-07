##############################
# prepare_data.py
# vishnun/SpellGram 데이터셋을 로드하고 Train/ValTest 분리하여 저장

# 데이터 분할 (80:10:10) 

# Train: 32,000개 (80%) - 모델 학습용
# Validation: 4,000개 (10%) - 학습 중 성능 평가용
# Test: 4,000개 (10%) - 최종 성능 평가용

# 저장 경로: saved/data/train, saved/data/val, saved/data/test
##############################

import os
import sys
from pathlib import Path
from datasets import load_dataset

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config

def prepare_data():
    print("=" * 60)
    print("SpellGram 데이터셋 로드 & 분리")
    print("=" * 60)

    # 데이터셋 로드
    try:
        ds = load_dataset("vishnun/SpellGram")
        full_data = ds["train"]
        print(f"- 전체 데이터 개수: {len(full_data)}")
    except Exception as e:
        print(f"데이터셋 다운로드 실패: {e}")
        return

    # Train/Val/Test 분리 (8:1:1)
    # 1. 먼저 Train(80%)과 Temp(20%)로 분리
    train_temp_split = full_data.train_test_split(test_size=0.2, seed=42)
    train_data = train_temp_split["train"]
    temp_data = train_temp_split["test"]
    
    # 2. Temp(20%)를 Val(10%)과 Test(10%)로 절반씩 분리
    val_test_split = temp_data.train_test_split(test_size=0.5, seed=42)
    val_data = val_test_split["train"]
    test_data = val_test_split["test"]

    print(f"- Train : {len(train_data)} (80%)")
    print(f"- Val   : {len(val_data)} (10%)")
    print(f"- Test  : {len(test_data)} (10%)")

    # 저장 경로 설정
    # Config에 정의된 데이터 경로가 있다면 사용하고, 없다면 기본 경로 사용
    data_dir = project_root / "saved" / "data"
    
    train_path = data_dir / "train"
    val_path = data_dir / "val"
    test_path = data_dir / "test"

    print("\n데이터 저장 중...")
    train_data.save_to_disk(train_path)
    val_data.save_to_disk(val_path)
    test_data.save_to_disk(test_path)
    
    print(f"저장 완료:")
    print(f"  - Train: {train_path}")
    print(f"  - Val  : {val_path}")
    print(f"  - Test : {test_path}")

if __name__ == "__main__":
    prepare_data()
