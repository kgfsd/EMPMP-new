"""
改进版: 保留Stylization机制的纯GCN模型
- 用GCN特征替代x_global进行调制
- 保留scale/shift的调制机制
- 完全消除P*T维度依赖
"""
import torch
from torch import nn
from einops.layers.torch import Rearrange
from .gcn import DynamicGCN


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        # B, P, D, T
        mean = x.mean(axis=-2, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-2, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = x.transpose(-2, -1)
        x = self.fc(x)
        x = x.transpose(-2, -1)
        return x


class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPblock(nn.Module):
    """单个MLP块:时序FC + 空间FC"""
    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial'):
        super().__init__()

        self.fc0 = Temporal_FC(seq)
        self.fc1 = Spatial_FC(dim)

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
                self.norm1 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
                self.norm1 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
                self.norm1 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc0.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc1.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc1.fc.bias, 0)

    def forward(self, x):
        # Temporal
        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_
        
        # Spatial
        x__ = self.fc1(x)
        x__ = self.norm1(x__)
        x = x + x__
        
        return x


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class GCNBlock(nn.Module):
    """
    GCN块 - 负责建模人与人之间的空间交互
    输入: [B, P, D, T]
    输出: [B, D, T] - 消除人数维度的全局特征
    
    通过聚合所有人的GCN特征来消除人数影响,使模型能处理可变人数
    """
    def __init__(self, dim, gcn_layers=2):
        super().__init__()
        self.gcn = DynamicGCN(
            dim=dim,
            num_layers=gcn_layers,
        )
        
    def forward(self, x, distances, padding_mask=None):
        """
        x: [B, P, D, T] - 输入特征
        distances: [B, T, P, P] - 人际距离矩阵
        padding_mask: [B, P] - 有效人员掩码
        返回: [B, D, T] - 消除人数维度的全局特征
        """
        # 直接返回 per-person 的 GCN 表征 [B, P, D, T]
        # 这样上层可以选择对每个 person 做独立的调制或再聚合。
        x_gcn = self.gcn(x, distances, padding_mask)  # [B, P, D, T]

        # 可选：在此处可以添加对每个 person 的小投影（例如 temporal projection），
        # 目前保持原样并直接返回 per-person 特征以供上层使用。
        return x_gcn


class StylizationBlock(nn.Module):
    """
    Stylization块 - 改进版,消除人数P的影响
    x_global维度: [B, D, T] (无P维度)
    
    交互逻辑:
    1. 用x_global生成scale/shift调制x (局部特征)
    2. 用x聚合后更新x_global
    3. 添加距离信息到x
    """
    def __init__(self, time_dim, dim, gcn_layers=2):
        super().__init__()
        self.time_dim = time_dim
        self.dim = dim
           
        
         # 改进的Stylization组件 - 适配 x_global 为 [B, P, D, T]
        self.emb_layers = nn.Sequential(
            nn.Linear(time_dim, 2 * time_dim),  # T -> 2T (无P依赖)
        )
        self.norm = LN(dim)
        self.norm_global = LN(dim)
        self.out_layers = nn.Sequential(
            zero_module(nn.Linear(time_dim, time_dim)),
        )
        # 对于 B D P T 设计：对每个 person 的时序维做投影以生成 per-person shift
        self.global_emb_layers = nn.Sequential(
            nn.Linear(time_dim, time_dim),  # T -> T
        )
        
        self.temp_linear = nn.Sequential(
            nn.Linear(time_dim, time_dim),
        )
        # 距离编码: 不依赖固定 num_p。
        # 我们对每个标量距离使用一个小 MLP 将其映射到 D 维，然后对目标人维度求和/池化，得到每个人的距离上下文特征。
        # 输入 distances: [B, T, P, P], 对最后一个维度 (targets) 进行元素级编码。
        self.dis_scalar_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
         # 用于对 person 维做加权池化的得分网络 (基于 per-person 的 D 维特征)
        self.pool_score_fc = nn.Sequential(
            nn.Linear(dim, 1),
        )
         # 对每个编码后的距离向量生成 attention score (用于 targets 维的加权)
        self.dis_attn = nn.Linear(dim, 1)

    def forward(self, x, x_global, distances, padding_mask=None):
        """
        x: [B, P, D, T] - 局部特征
        x_global: [B, P, D, T] - per-person 全局特征(由GCN生成)
        distances: [B, T, P, P] - 人际距离矩阵
        padding_mask: [B, P] - 有效人员掩码
        返回: (x_new, x_global_new) - 更新后的局部和全局特征
        """
        b, p, d, t = x.shape
        x_clone=x.clone()
        # 用 x_global 生成 scale 和 shift 调制 x
        # 我们先在 person 维度上做 masked pooling -> 得到 [B, 1, D, T] 的共享全局
          # 用 attention-based masked pooling 在 person 维上产生共享全局 [B,1,D,T]
        # 先对 time 维做平均以获得每人 summary [B,P,D]
        person_summary = x_global.mean(dim=-1)  # [B, P, D]
        scores = self.pool_score_fc(person_summary)  # [B, P, 1]
        scores = scores.squeeze(-1)  # [B, P]
        if padding_mask is not None:
            # mask out invalid persons
            scores = scores.masked_fill(padding_mask == 0, float('-1e9'))
        attn = torch.softmax(scores, dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1]
        xg_pooled = (x_global * attn).sum(dim=1, keepdim=True)  # [B,1,D,T]

        # emb_layers 对 time 维进行变换，输入形状应为 [..., T]
        xg_pooled = self.emb_layers(xg_pooled)  # [B,1,D,2*T]
        scale, shift = torch.chunk(xg_pooled, 2, dim=-1)  # 每个: B,1,D,T
        x = x * (1 + scale) + shift  # B, P, D, T
        x = self.out_layers(x)
        
        # 添加距离信息 - 使用动态编码，无需固定 num_p
        # distances: [B, T, P, P] - 对每个人 i, distances[:, :, i, :] 表示他到所有人的距离
        # 我们先重排到 [B, P, T, P] (sources, targets)，然后对每个标量执行相同的 MLP，得到 [B, P, T, P, D]
        # 最后在 targets 维度上求和或做 masked sum，得到每个 source 的距离上下文 [B, P, T, D]
        # 重排
        distances_sp = distances.permute(0, 2, 1, 3)  # [B, P, T, P]
        # 为 scalar MLP 准备输入: 添加最后一个单通道维度
        # distances_sp.unsqueeze(-1): [B, P, T, P, 1]
        d_in = distances_sp.unsqueeze(-1)
        # 将最后一个维度送入共享 MLP -> 输出 [B, P, T, P, D]
        d_encoded = self.dis_scalar_mlp(d_in)  # broadcasting linear over last dim
        # 对 targets(P) 维度做加权/普通求和, 使用 padding_mask 屏蔽无效人员
         # 使用 attention 在 targets 维度上做加权聚合
        # d_encoded: [B, P, T, P, D]
        # 计算每个 target 的得分
        attn_logits = self.dis_attn(d_encoded)  # [B, P, T, P, 1]
        attn_logits = attn_logits.squeeze(-1)  # [B, P, T, P]
        if padding_mask is not None:
            # padding_mask: [B, P]  -> 用于 mask targets 维度
            mask_t = padding_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,P]
            attn_logits = attn_logits.masked_fill(mask_t == 0, float('-1e9'))
        # softmax over targets dim (dim=3)
        attn_weights = torch.softmax(attn_logits, dim=3)  # [B, P, T, P]
        # 加权求和 targets
        d_context = (d_encoded * attn_weights.unsqueeze(-1)).sum(dim=3)  # [B, P, T, D]

        # 转换到 [B, P, D, T]
        distances_encoded = d_context.permute(0, 1, 3, 2)
        distances_encoded = self.temp_linear(distances_encoded)  # [B, P, D, T]
        x = x + distances_encoded
        x = self.norm(x)  # [B, P, D, T]
        
        # 用 x 的 per-person 特征更新 x_global（都为 [B, P, D, T]）
        # 直接对每个 person 的时序维做投影得到 shift2: [B, P, D, T]
        shift2 = self.global_emb_layers(x_clone)  # [B, P, D, T]
        x_global = x_global + shift2
        x_global = self.norm_global(x_global)
        
        return x, x_global

    
class TransMLPWithGCNStylization(nn.Module):
    """

    -  x_global = [B, D, P,T] (包含人数P,固定人数
    
    流程: 多层局部MLP + 定期插入GCN和Stylization交互
    """
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis,
                 interaction_interval=2, num_p=3, gcn_layers=2):
        super().__init__()
        
        # 局部MLP层
        self.local_mlps = nn.Sequential(*[
            MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers)
        ])
        
        # GCN块 - 生成x_global [B, D, T] (无P维度)
        num_interactions = num_layers // interaction_interval
        self.gcn_blocks = nn.ModuleList([
            GCNBlock(dim, gcn_layers=gcn_layers)
            for _ in range(num_interactions)
        ])
        
        # Stylization块 - 适配无P维度的x_global
        self.stylization_blocks = nn.ModuleList([
            StylizationBlock(seq, dim, gcn_layers=gcn_layers)
            for _ in range(num_interactions)
        ])
        
        self.interaction_interval = interaction_interval
        self.num_layers = num_layers
        
    def forward(self, x, distances, padding_mask=None):
        """
        x: [B, P, D, T] - 输入特征 (P可变)
        distances: [B, T, P, P] - 人际距离
        padding_mask: [B, P] - 有效人员掩码
        返回: [B, P, D, T] - 输出特征
        """
        b, p, d, t = x.shape
        
        # 初始化x_global: 用GCN从初始x生成 [B, D, T]
        x_global = self.gcn_blocks[0](x, distances, padding_mask)  # [B, D, P,T] 
        
        global_step = 0
        
        # 逐层处理
        for i, local_layer in enumerate(self.local_mlps):
            # 局部时序+空间建模 (MLPblock)
            x = local_layer(x)
            
            # 每隔 interaction_interval 层,进行全局交互
            if (i + 1) % self.interaction_interval == 0 and global_step < len(self.gcn_blocks):
                # 用GCN更新x_global(如果不是第一次)
                if global_step > 0:
                    x_global_gcn = self.gcn_blocks[global_step](x, distances, padding_mask)  # [B, D, T]
                    x_global = x_global + x_global_gcn  # 残差连接
                
                # Stylization交互: x和x_global相互作用
                x_new, x_global_new = self.stylization_blocks[global_step](
                    x, x_global, distances, padding_mask
                )
                
                # 残差连接
                x = x + x_new
                x_global = x_global + x_global_new
                
                global_step += 1
        
        return x


def build_mlps_gcn_stylization(args):
    """构建带Stylization的GCN版本模型"""
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    
    gcn_layers = getattr(args, 'gcn_layers', 2)
    interaction_interval = getattr(args, 'interaction_interval', 2)
    return TransMLPWithGCNStylization(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
        interaction_interval=interaction_interval,
        gcn_layers=gcn_layers,
    )
