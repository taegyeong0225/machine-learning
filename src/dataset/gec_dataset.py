"""
Dataset for Grammar Error Correction (GEC)

SpellGram 데이터셋을 위한 PyTorch Dataset 클래스
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional


class GECDataset(Dataset):
    """
    Grammar Error Correction Dataset
    
    Args:
        data (list): 데이터 리스트, 각 항목은 {'source': str, 'target': str} 형태
        tokenizer: SimpleTokenizer 인스턴스
        max_len (int): 최대 시퀀스 길이
    """
    
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_len: int = 32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        데이터 항목 반환
        
        Returns:
            dict: {
                'src': torch.Tensor - 소스 토큰 ID [max_len]
                'tgt': torch.Tensor - 타겟 토큰 ID [max_len]
                'src_text': str - 원본 소스 텍스트
                'tgt_text': str - 원본 타겟 텍스트
            }
        """
        ex = self.data[idx]
        
        # 인코딩 (SOS와 EOS 포함)
        src_ids = self.tokenizer.encode(ex['source'])
        tgt_ids = self.tokenizer.encode(ex['target'])
        
        # 길이 제한 (max_len보다 길면 자르기)
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]
        
        # 패딩 (max_len까지)
        src_ids = src_ids + [0] * (self.max_len - len(src_ids))  # 0 = <PAD>
        tgt_ids = tgt_ids + [0] * (self.max_len - len(tgt_ids))
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
            'src_text': ex['source'],
            'tgt_text': ex['target']
        }


def collate_fn(batch):
    """
    DataLoader용 collate 함수
    
    Args:
        batch: Dataset의 __getitem__ 반환값들의 리스트
    
    Returns:
        dict: 배치 데이터
    """
    # 배치에서 텐서 추출
    src_batch = torch.stack([item['src'] for item in batch])
    tgt_batch = torch.stack([item['tgt'] for item in batch])
    
    # 텍스트 정보도 함께 반환
    src_texts = [item['src_text'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]
    
    return {
        'src': src_batch,
        'tgt': tgt_batch,
        'src_text': src_texts,
        'tgt_text': tgt_texts
    }


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 60)
    print("GECDataset 테스트")
    print("=" * 60)
    
    from src.utils.tokenize import SimpleTokenizer
    
    # 샘플 데이터
    sample_data = [
        {'source': 'this is a smple sentence', 'target': 'this is a sample sentence'},
        {'source': 'helo world', 'target': 'hello world'},
        {'source': 'machne lernig', 'target': 'machine learning'}
    ]
    
    # 토크나이저 생성
    tokenizer = SimpleTokenizer(vocab_size=1000, verbose=False)
    all_texts = [ex['source'] for ex in sample_data] + [ex['target'] for ex in sample_data]
    tokenizer.build_vocab(all_texts)
    
    # 데이터셋 생성
    dataset = GECDataset(sample_data, tokenizer, max_len=20)
    
    print(f"\n데이터셋 크기: {len(dataset)}")
    
    # 샘플 데이터 확인
    sample = dataset[0]
    print(f"\n샘플 데이터:")
    print(f"  - src shape: {sample['src'].shape}")
    print(f"  - tgt shape: {sample['tgt'].shape}")
    print(f"  - src_text: {sample['src_text']}")
    print(f"  - tgt_text: {sample['tgt_text']}")
    print(f"  - src IDs: {sample['src'].tolist()[:10]}...")
    print(f"  - tgt IDs: {sample['tgt'].tolist()[:10]}...")
    
    # DataLoader 테스트
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"\nDataLoader 테스트:")
    for i, batch in enumerate(dataloader):
        print(f"\n배치 {i+1}:")
        print(f"  - src shape: {batch['src'].shape}")
        print(f"  - tgt shape: {batch['tgt'].shape}")
        print(f"  - src_texts: {batch['src_text']}")
        if i == 0:  # 첫 번째 배치만 출력
            break
    
    print("\n GECDataset 테스트 완료!")

