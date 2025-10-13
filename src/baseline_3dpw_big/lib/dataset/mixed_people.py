#!/usr/bin/env python3
"""
创建混合人数数据集用于测试动态人数功能
"""

import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader

class MixedPeopleDataset(Dataset):
    """
    混合不同人数的数据集
    """
    def __init__(self, data_dir, t_his=16, t_pred=14, split='train', train_ratio=0.8):
        """
        Args:
            data_dir: 数据目录
            t_his: 输入帧数
            t_pred: 预测帧数
            split: 'train' 或 'test'
            train_ratio: 训练集比例（默认80%）
        """
        self.data_dir = data_dir
        self.t_his = t_his
        self.t_pred = t_pred
        self.split = split
        self.train_ratio = train_ratio
        self.samples = []
        
        # 加载不同人数的数据文件
        data_files = {
            'mix1_6persons.npy': 6,
            # 'mix2_10persons.npy': 10,  # 注释掉10人数据集，减小显存占用
            'mupots_120_3persons.npy': 3
        }
        
        print("Loading mixed people datasets...")
        for filename, expected_people in data_files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                try:
                    data = np.load(filepath)
                    print(f"Loaded {filename}: shape={data.shape}")
                    
                    # 处理数据格式
                    if len(data.shape) == 4:  # [N, P, T, JK]
                        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], -1, 3)
                    elif len(data.shape) == 5:  # [N, P, T, J, K]
                        pass
                    else:
                        print(f"Unexpected data shape for {filename}: {data.shape}")
                        continue
                    
                    # 只取足够长度的序列
                    total_frames = self.t_his + self.t_pred
                    if data.shape[2] >= total_frames:
                        # 划分训练集和测试集
                        num_sequences = data.shape[0]
                        train_size = int(num_sequences * self.train_ratio)
                        
                        if self.split == 'train':
                            sequence_indices = range(0, train_size)
                        else:  # test
                            sequence_indices = range(train_size, num_sequences)
                        
                        # 从选定的序列中抽取样本
                        for i in sequence_indices:
                            if data.shape[2] >= total_frames:
                                # 固定采样（使用序列索引作为种子确保可复现）
                                np.random.seed(i)  # 固定随机种子
                                start_idx = np.random.randint(0, max(1, data.shape[2] - total_frames + 1))
                                sample = data[i, :, start_idx:start_idx + total_frames]
                                
                                # 确保人数正确
                                actual_people = sample.shape[0]
                                if actual_people > 0:
                                    self.samples.append({
                                        'data': sample,
                                        'people_count': actual_people,
                                        'source': filename,
                                        'seq_idx': i  # 添加序列索引用于调试
                                    })
                    
                    print(f"Extracted {len([s for s in self.samples if s['source'] == filename])} samples from {filename}")
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"File not found: {filepath}")
        
        print(f"Total samples: {len(self.samples)}")
        
        # 统计人数分布
        people_counts = [s['people_count'] for s in self.samples]
        unique, counts = np.unique(people_counts, return_counts=True)
        print(f"People distribution: {dict(zip(unique, counts))}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = sample['data']  # [P, T, J, 3]
        
        # 确保数据格式: [P, T, J*3]
        if len(data.shape) == 4:  # [P, T, J, 3]
            data = data.reshape(data.shape[0], data.shape[1], -1)
        
        return torch.FloatTensor(data)


def collate_mixed_batch(batch):
    """
    处理不同人数的批次，模拟原始3DPW数据集格式
    """
    # 找到最大人数
    max_people = max(sample.shape[0] for sample in batch)
    batch_size = len(batch)
    time_steps = batch[0].shape[1]
    joint_features = batch[0].shape[2]  # 45 = 15 joints * 3 coords
    
    # 计算关节和坐标维度
    coords_dim = 3
    joints_dim = joint_features // coords_dim  # 15 joints
    
    # 创建5维张量匹配原始格式 [B, P, T, J, 3]
    padded_batch = torch.zeros(batch_size, max_people, time_steps, joints_dim, coords_dim)
    padding_mask = torch.zeros(batch_size, max_people, dtype=torch.bool)
    
    for i, sample in enumerate(batch):
        actual_people = sample.shape[0]
        # 重塑样本数据：[P, T, JK] -> [P, T, J, 3]
        sample_reshaped = sample.reshape(actual_people, time_steps, joints_dim, coords_dim)
        padded_batch[i, :actual_people] = sample_reshaped
        padding_mask[i, :actual_people] = True
    
    # 返回格式匹配原始collate函数
    # 原始函数期望：tensor_data, None, padding_mask
    return padded_batch, None, padding_mask


def test_mixed_people_dataset():
    """
    测试混合人数数据集
    """
    print("=" * 60)
    print("测试混合人数数据集（带 train/test 分割）")
    print("=" * 60)
    
    # 创建数据集
    data_dir = r"C:\Users\31564\Desktop\EMPMP\data"
    dataset_train = MixedPeopleDataset(data_dir, split='train')
    dataset_test = MixedPeopleDataset(data_dir, split='test')
    
    print(f"\n训练集样本数: {len(dataset_train)}")
    print(f"测试集样本数: {len(dataset_test)}")
    
    # 检查训练集和测试集的序列是否有重叠
    train_seqs = set((s['source'], s['seq_idx']) for s in dataset_train.samples)
    test_seqs = set((s['source'], s['seq_idx']) for s in dataset_test.samples)
    overlap = train_seqs & test_seqs
    
    if overlap:
        print(f"\n⚠️  警告：发现 {len(overlap)} 个重叠序列！")
    else:
        print("\n✅ 训练集和测试集无重叠")
    
    # 使用训练集继续测试
    dataset = dataset_train
    
    if len(dataset) == 0:
        print("❌ 数据集为空！")
        return False
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=collate_mixed_batch
    )
    
    print("\n检查数据加载器...")
    
    for i, (batch_data, _, padding_mask) in enumerate(dataloader):
        print(f"\nBatch {i + 1}:")
        print(f"  Batch shape: {batch_data.shape}")
        print(f"  Padding mask shape: {padding_mask.shape}")
        
        # 计算每个样本的人数
        people_counts = padding_mask.sum(dim=1).numpy()
        print(f"  People counts: {people_counts}")
        print(f"  People range: {people_counts.min()} - {people_counts.max()}")
        print(f"  Average people: {people_counts.mean():.1f}")
        
        # 只检查前3个批次
        if i >= 2:
            break
    
    print("\n✅ 混合人数数据集测试完成！")
    return True


if __name__ == "__main__":
    success = test_mixed_people_dataset()
    if success:
        print("\n🎉 混合人数数据集创建成功！现在可以测试真正的动态人数了")
    else:
        print("\n💥 混合人数数据集创建失败")