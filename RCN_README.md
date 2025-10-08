# RCN实现：支持多人及可变人数场景的轻量化方案

## 项目概述

本项目实现了关系卷积网络（Relation Convolutional Network, RCN）来改进现有的多人运动预测模型，使其能够：

1. **支持多人场景**：处理2-N人的多人交互
2. **处理可变人数**：通过动态掩码机制支持不同场景的人数变化
3. **保持轻量化**：严格控制模型参数量（<3M）

## 三阶段实现方案

### 第一阶段：基础RCN架构

**文件**: `src/models_dual_inter_traj_3dpw/rcn.py`

**核心功能**:
- ✅ `RelationEncoder`: 编码空间、时间、交互三种关系类型
- ✅ `RelationConvLayer`: 关系图卷积层
- ✅ `BasicRCN`: 基础RCN模型（~0.8M参数）
- ✅ `create_person_mask`: 动态掩码生成

**使用示例**:
```python
from src.models_dual_inter_traj_3dpw.rcn import BasicRCN

rcn = BasicRCN(input_dim=39, hidden_dim=64, num_layers=2, max_persons=5)
output = rcn(features, distances, mask)
```

### 第二阶段：添加注意力机制

**文件**: `src/models_dual_inter_traj_3dpw/rcn_attention.py`

**核心功能**:
- ✅ `MultiHeadRelationAttention`: 多头关系注意力
- ✅ `AdaptiveRelationWeighting`: 自适应关系权重
- ✅ `AttentionRCNLayer`: 带注意力的RCN层
- ✅ `RCNWithAttention`: 完整注意力RCN（~1.5M参数）
- ✅ `EfficientAttention`: 线性复杂度注意力（O(P·d²) vs O(P²·d)）

**使用示例**:
```python
from src.models_dual_inter_traj_3dpw.rcn_attention import RCNWithAttention

rcn = RCNWithAttention(
    input_dim=39, hidden_dim=64, num_layers=3, 
    num_heads=4, dropout=0.1
)
output, attention_weights = rcn(features, distances, mask, return_attention=True)
```

### 第三阶段：优化关系类型和特征提取

**文件**: `src/models_dual_inter_traj_3dpw/rcn_optimized.py`

**核心功能**:
- ✅ `HierarchicalRelationType`: 层次化关系类型（空间几何+时序+交互）
- ✅ `EfficientFeatureExtractor`: 深度可分离卷积（降低8-10倍参数）
- ✅ `LearnableRelationWeighting`: 可学习的关系权重
- ✅ `OptimizedRCN`: 优化的完整RCN（~1.2M轻量 / ~2.5M标准）

**使用示例**:
```python
from src.models_dual_inter_traj_3dpw.rcn_optimized import create_optimized_rcn

rcn, num_params = create_optimized_rcn(
    input_dim=39, hidden_dim=64, relation_dim=32,
    num_layers=3, lightweight=True
)
output, info = rcn(features, distances, positions, velocities, mask)
```

## 集成到现有模型

**文件**: `src/models_dual_inter_traj_3dpw/rcn_integration.py`

提供三种集成方式：

```python
from src.models_dual_inter_traj_3dpw.rcn_integration import (
    integrate_rcn_phase1,  # 基础RCN
    integrate_rcn_phase2,  # 带注意力
    integrate_rcn_phase3,  # 优化版本
    VariablePersonRCN      # 可变人数专用
)

# 方式1：直接集成到现有模型
model = integrate_rcn_phase3(config, lightweight=True)

# 方式2：可变人数模型
model = VariablePersonRCN(config, min_persons=2, max_persons=5)
output = model(motion_input, traj, num_persons=[2, 3, 4, 5])
```

## 文件结构

```
src/models_dual_inter_traj_3dpw/
├── rcn.py                  # 第一阶段：基础RCN
├── rcn_attention.py        # 第二阶段：注意力机制
├── rcn_optimized.py        # 第三阶段：优化实现
└── rcn_integration.py      # 集成模块

docs/
└── RCN_Implementation_Guide.md  # 详细使用指南

test_rcn_demo.py            # 演示和测试脚本
RCN_README.md              # 本文件
```

## 关键特性

### 1. 可变人数支持

通过动态掩码机制，支持在同一个batch中处理不同人数的样本：

```python
# 创建掩码 (2人、3人、4人、5人的混合batch)
num_persons = [2, 3, 4, 5]
mask = create_person_mask(num_persons, max_persons=5, device='cuda')

# RCN会自动处理
output = rcn(features, distances, mask)
# 填充位置的输出自动为0
```

### 2. 轻量化设计

| 技术 | 作用 | 参数节省 |
|------|------|---------|
| 深度可分离卷积 | 特征提取 | 8-10倍 |
| 关系类型共享 | 编码器 | 3倍 |
| 组卷积 | 全连接层 | 2-4倍 |
| 低秩分解 | 注意力 | 2倍 |

### 3. 层次化关系建模

**空间关系**：
- 欧式距离
- 水平角度
- 高度差异

**时序关系**：
- 运动趋势
- 速度模式

**交互关系**：
- 行为相似性（点积）
- 行为互补性（求和）

### 4. 高效计算

- **BasicRCN**: O(T·P²·D²)
- **EfficientAttention**: O(T·P·D²) - 线性复杂度
- **OptimizedRCN**: O(T·P·D²) - 深度可分离卷积

## 性能对比

| 模型 | 参数量 | 计算复杂度 | 适用场景 |
|------|--------|-----------|---------|
| BasicRCN | 0.8M | 中等 | 快速原型 |
| RCNWithAttention | 1.5M | 较高 | 性能优先 |
| OptimizedRCN (轻量) | 1.2M | 低 | 实时应用 |
| OptimizedRCN (标准) | 2.5M | 中等 | 最佳性能 |

## 使用步骤

### 步骤1：选择合适的RCN版本

- **资源受限** → `BasicRCN`
- **需要注意力** → `RCNWithAttention`
- **追求轻量** → `OptimizedRCN(lightweight=True)`
- **追求性能** → `OptimizedRCN(lightweight=False)`

### 步骤2：集成到训练流程

```python
# 在 train_rc.py 中
from src.models_dual_inter_traj_3dpw.rcn_integration import integrate_rcn_phase3

# 创建模型
model = integrate_rcn_phase3(config, lightweight=True)

# 训练循环
for batch in dataloader:
    motion_input, motion_target, traj = batch
    
    # 前向传播
    output = model(motion_input, traj)
    
    # 计算损失和反向传播
    loss = criterion(output, motion_target)
    loss.backward()
    optimizer.step()
```

### 步骤3：处理可变人数

```python
from src.models_dual_inter_traj_3dpw.rcn import create_person_mask

# 在数据加载器中添加人数信息
num_persons = [len(sample['persons']) for sample in batch]

# 创建掩码
mask = create_person_mask(num_persons, max_persons, device)

# 传递给模型
output = model(motion_input, traj, num_persons=num_persons)
```

## 测试

运行演示脚本（需要安装依赖）：

```bash
python test_rcn_demo.py
```

验证Python语法：

```bash
python -m py_compile src/models_dual_inter_traj_3dpw/rcn*.py
```

## 配置建议

### 基础配置（推荐）

```python
config = {
    'input_dim': 39,
    'hidden_dim': 64,
    'relation_dim': 32,
    'num_layers': 3,
    'num_heads': 4,
    'dropout': 0.1,
    'lightweight': True
}
```

### 高性能配置

```python
config = {
    'input_dim': 39,
    'hidden_dim': 128,
    'relation_dim': 64,
    'num_layers': 4,
    'num_heads': 8,
    'dropout': 0.1,
    'lightweight': False
}
```

### 轻量配置

```python
config = {
    'input_dim': 39,
    'hidden_dim': 32,
    'relation_dim': 16,
    'num_layers': 2,
    'num_heads': 4,
    'dropout': 0.05,
    'lightweight': True
}
```

## 训练技巧

1. **学习率策略**: Warmup + Cosine Decay
2. **梯度裁剪**: `max_norm=1.0`
3. **混合精度**: 使用AMP加速训练
4. **数据增强**: 随机改变人数
5. **正则化**: Weight Decay + Dropout

## 未来改进方向

- [ ] 添加语义关系类型
- [ ] 实现图神经网络变体
- [ ] 支持更大规模场景（>10人）
- [ ] 优化推理速度
- [ ] 添加预训练模型

## 参考

详细使用指南: `docs/RCN_Implementation_Guide.md`

## 总结

本实现提供了完整的三阶段RCN方案，从基础架构到优化版本，每个阶段都有清晰的目标和实现。通过模块化设计，可以根据实际需求灵活选择和集成，同时保持模型的轻量化特点。

**核心优势**：
- ✅ 支持多人场景（2-N人）
- ✅ 处理可变人数（动态掩码）
- ✅ 保持轻量化（<3M参数）
- ✅ 模块化设计（易于集成）
- ✅ 完整文档（使用指南）
