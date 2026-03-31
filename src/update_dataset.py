"""
Script để sửa dataset Twitter16:
1. Ánh xạ label: non-rumor, true -> truth (0); false, unverified -> rumor (1)
2. Đổi tên cột: text -> content
"""

import pandas as pd
from pathlib import Path

def map_label(label):
    """
    Ánh xạ label từ raw label sang binary label
    - non-rumor, true -> truth (0)
    - false, unverified -> rumor (1)
    """
    if label.lower() in ('non-rumor', 'true'):
        return 'truth'
    elif label.lower() in ('false', 'unverified'):
        return 'rumor'
    else:
        raise ValueError(f"Unknown label: {label}")

def fix_dataset(input_path, output_path=None):
    """
    Sửa dataset
    
    Args:
        input_path: đường dẫn file CSV đầu vào
        output_path: đường dẫn file CSV đầu ra (mặc định ghi đè file gốc)
    """
    if output_path is None:
        output_path = input_path
    
    # Đọc CSV
    print(f"📖 Đọc file: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"   Dữ liệu: {len(df)} rows, {list(df.columns)}")
    
    # Đổi tên cột text -> content (nếu có)
    if 'text' in df.columns:
        df = df.rename(columns={'text': 'content'})
        print("   ✓ Đổi tên cột: text -> content")
    
    # Ánh xạ label (nếu có cột label)
    if 'label' in df.columns:
        print(f"   Label trước: {df['label'].unique()}")
        df['label'] = df['label'].apply(map_label)
        print(f"   ✓ Ánh xạ label: {df['label'].unique()}")
    
    # Lưu file
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Lưu file: {output_path}\n")
    
    return df

def process_twitter16_dataset():
    """
    Xử lý dataset Twitter16 từ thư mục data/Twitter16/time/
    """
    base_dir = Path('./data/ver_1')
    
    if not base_dir.exists():
        print(f"❌ Không tìm thấy thư mục: {base_dir}")
        return
    
    print("="*60)
    print("FIXING TWITTER16 DATASET")
    print("="*60 + "\n")
    
    # Xử lý từng file
    for split in ['train', 'test']:
        csv_file = base_dir / f'{split}.csv'
        
        if csv_file.exists():
            print(f"🔄 Xử lý {split}.csv:")
            df = fix_dataset(csv_file)
            
            # In thống kê
            print(f"   📊 Thống kê label:")
            print(f"   {df['label'].value_counts().to_dict()}\n")
        else:
            print(f"⚠️  Không tìm thấy: {csv_file}\n")

if __name__ == '__main__':
    process_twitter16_dataset()
    print("✨ Hoàn thành!")
