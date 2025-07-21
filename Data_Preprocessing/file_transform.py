import numpy as np
from pathlib import Path

# ==================== 配置部分 ====================
input_dir = "/root/autodl-tmp"          # 原始数据目录
output_dir = "/root/autodl-tmp/encoded_data"  # 输出目录
label_mapping = {                       # 标签映射表
    'background': 0,
    'cargo': 1,
    'tanker': 2,
    'tug': 3,
    'passengership': 4
}

# 自动创建输出目录
Path(output_dir).mkdir(parents=True, exist_ok=True)

def encode_npz_labels(npz_path, output_path):
    """核心转换函数：将字符串标签转换为整数标签"""
    try:
        # 加载数据
        with np.load(npz_path) as data:
            images = data['data']
            str_labels = data['labels']
            
            # 数据完整性检查
            assert len(images) == len(str_labels), "图像与标签数量不匹配"
            
            # 标签转换
            int_labels = np.array([label_mapping[label] for label in str_labels], dtype=np.int64)
            
            # 保存新数据（压缩格式）
            np.savez_compressed(
                output_path,
                data=images,
                labels=int_labels
            )
            
            # 打印转换统计
            unique_labels, counts = np.unique(int_labels, return_counts=True)
            print(f"\n✅ 转换完成: {Path(npz_path).name} → {Path(output_path).name}")
            print(f"标签分布: {dict(zip(unique_labels, counts))}")
            
    except Exception as e:
        print(f"\n❌ 转换失败: {Path(npz_path).name}")
        print(f"错误详情: {str(e)}")
        raise

# ==================== 批量处理 ====================
file_pairs = [
    ("train_data_labels.npz", "train_encoded.npz"),
    ("validation_data_labels.npz", "val_encoded.npz"),
    ("test_data_labels.npz", "test_encoded.npz")
]

for input_file, output_file in file_pairs:
    input_path = Path(input_dir) / input_file
    output_path = Path(output_dir) / output_file
    
    if not input_path.exists():
        print(f"⚠️ 文件不存在: {input_file}，已跳过")
        continue
        
    encode_npz_labels(input_path, output_path)

print("\n🎉 全部处理完成！输出目录:", output_dir)