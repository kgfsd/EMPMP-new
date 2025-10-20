"""
GCN+Stylization多人运动预测核心模块
集成IPLM交互先验学习，支持可变人数场景的运动建模
"""
import torch
from torch import nn
from einops.layers.torch import Rearrange
from .gcn import DynamicGCN


class LN(nn.Module):
    """
    Layer Normalization for [B, P, D, T] format
    在特征维度D上进行归一化
    """
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        # x: [B, P, D, T]
        mean = x.mean(axis=-2, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-2, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class LN_v2(nn.Module):
    """
    Layer Normalization for temporal dimension
    在时序维度T上进行归一化
    """
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        # x: [B, P, T, D] or similar
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class Spatial_FC(nn.Module):
    """
    空间全连接层，处理特征维度D
    输入输出: [B, P, D, T]
    """
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, P, D, T] -> [B, P, T, D] -> FC -> [B, P, T, D] -> [B, P, D, T]
        x = x.transpose(-2, -1)
        x = self.fc(x)
        x = x.transpose(-2, -1)
        return x


class Temporal_FC(nn.Module):
    """
    时序全连接层，处理时间维度T
    输入输出: [B, P, D, T]
    """
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, P, D, T] -> FC on T dimension
        x = self.fc(x)
        return x


class MLPblock(nn.Module):
    """
    单个MLP块: 时序FC + 空间FC + IPLM先验知识学习
    
    Args:
        dim: 特征维度D
        seq: 序列长度T
        use_iplm: 是否使用IPLM模块
        iplm_config: IPLM配置参数
    
    输入输出: [B, P, D, T]
    """
    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial', 
                 use_iplm=False, iplm_config=None):
        super().__init__()

        self.fc0 = Temporal_FC(seq)
        self.fc1 = Spatial_FC(dim)
        self.use_iplm = use_iplm

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

        # IPLM模块集成
        if self.use_iplm and iplm_config is not None:
            from .iplm import InteractionFeatureExtractor, IPLM
            self.ipm_feature_extractor = InteractionFeatureExtractor(
                feature_dim=iplm_config.feature_dim,
                hidden_dim=iplm_config.hidden_dim
            )
            self.iplm = IPLM(
                knowledge_space_size=iplm_config.knowledge_space_size,
                feature_dim=iplm_config.feature_dim,
                lr=iplm_config.lr
            )
            # 将IPLM特征投影到MLP特征维度
            self.ipm_to_mlp_proj = nn.Linear(iplm_config.feature_dim, dim)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化网络参数"""
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc0.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc1.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc1.fc.bias, 0)
        
        if self.use_iplm and hasattr(self, 'ipm_to_mlp_proj'):
            nn.init.xavier_uniform_(self.ipm_to_mlp_proj.weight, gain=1.0)
            nn.init.constant_(self.ipm_to_mlp_proj.bias, 0)

    def forward(self, x, distances=None, padding_mask=None):
        """
        前向传播
        
        Args:
            x: [B, P, D, T] - 输入特征
            distances: [B, T, P, P] - 人际距离矩阵 (仅在使用IPLM时需要)
            padding_mask: [B, P] - 有效人员掩码 (仅在使用IPLM时需要)
        
        Returns:
            x: [B, P, D, T] - 输出特征
            loss_lk: float - IPLM损失 (如果启用)
        """
        loss_lk = None
        
        # IPLM先验知识学习 (在局部特征处理前应用)
        if self.use_iplm and distances is not None:
            # 提取交互特征
            f_ipm = self.ipm_feature_extractor(distances, padding_mask)  # [B, T, D_ipm]
            f_ipmlm, loss_lk = self.iplm(f_ipm)  # [B, T, D_ipm], loss
            
            # 投影到MLP特征维度并融合
            ipm_features = self.ipm_to_mlp_proj(f_ipmlm)  # [B, T, D]
            # 广播到 [B, P, D, T] 并与输入特征融合
            imp_features = ipm_features.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, D, T]
            x = x + imp_features  # 残差连接融合先验知识
        
        # 时序处理
        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_
        
        # 空间处理
        x__ = self.fc1(x)
        x__ = self.norm1(x__)
        x = x + x__
        
        if self.use_iplm:
            return x, loss_lk
        else:
            return x


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class GCNBlock(nn.Module):
    """
    GCN块 - 建模人与人之间的空间交互
    
    功能:
    - 通过图卷积网络学习人际关系
    - 输出per-person的全局特征表示
    - 支持可变人数场景
    
    Args:
        dim: 特征维度D
        gcn_layers: GCN层数
    
    输入输出: [B, P, D, T]
    """
    def __init__(self, dim, gcn_layers=2):
        super().__init__()
        self.gcn = DynamicGCN(
            dim=dim,
            num_layers=gcn_layers,
        )
        
    def forward(self, x, distances, padding_mask=None):
        """
        前向传播
        
        Args:
            x: [B, P, D, T] - 输入特征
            distances: [B, T, P, P] - 人际距离矩阵
            padding_mask: [B, P] - 有效人员掩码
        
        Returns:
            x_gcn: [B, P, D, T] - per-person的GCN特征表示
        """
        x_gcn = self.gcn(x, distances, padding_mask)  # [B, P, D, T]
        return x_gcn


class StylizationBlock(nn.Module):
    """
    Stylization块 - 特征调制与交互模块
    
    功能:
    1. 使用x_global生成scale/shift参数调制局部特征x
    2. 融合距离信息增强空间感知
    3. 更新全局特征x_global
    
    Args:
        time_dim: 时序维度T
        dim: 特征维度D
        gcn_layers: GCN层数
    
    输入输出: [B, P, D, T]
    """
    def __init__(self, time_dim, dim, gcn_layers=2):
        super().__init__()
        self.time_dim = time_dim
        self.dim = dim
           
        # Stylization组件 - 生成调制参数
        self.emb_layers = nn.Sequential(
            nn.Linear(time_dim, 2 * time_dim),  # T -> 2T (scale + shift)
        )
        self.norm = LN(dim)
        self.norm_global = LN(dim)
        self.out_layers = nn.Sequential(
            zero_module(nn.Linear(time_dim, time_dim)),
        )
        
        # 全局特征处理
        self.global_emb_layers = nn.Sequential(
            nn.Linear(time_dim, time_dim),  # T -> T
        )
        
        self.temp_linear = nn.Sequential(
            nn.Linear(time_dim, time_dim),
        )
        
        # 距离编码模块 - 支持可变人数
        self.dis_scalar_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        
        # 人员维度加权池化
        self.pool_score_fc = nn.Sequential(
            nn.Linear(dim, 1),
        )
        
        # 距离注意力机制
        self.dis_attn = nn.Linear(dim, 1)

    def forward(self, x, x_global, distances, padding_mask=None):
        """
        前向传播
        
        Args:
            x: [B, P, D, T] - 局部特征
            x_global: [B, P, D, T] - per-person全局特征(由GCN生成)
            distances: [B, T, P, P] - 人际距离矩阵
            padding_mask: [B, P] - 有效人员掩码
        
        Returns:
            x_new: [B, P, D, T] - 调制后的局部特征
            x_global_new: [B, P, D, T] - 更新后的全局特征
        """
        b, p, d, t = x.shape
        x_clone = x.clone()
        
        # 1. 使用x_global生成scale和shift调制参数
        # 通过attention-based masked pooling在person维上产生共享全局特征 [B,1,D,T]
        person_summary = x_global.mean(dim=-1)  # [B, P, D]
        scores = self.pool_score_fc(person_summary)  # [B, P, 1]
        scores = scores.squeeze(-1)  # [B, P]
        
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask == 0, float('-1e9'))
        
        attn = torch.softmax(scores, dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1]
        xg_pooled = (x_global * attn).sum(dim=1, keepdim=True)  # [B,1,D,T]

        # 生成调制参数
        xg_pooled = self.emb_layers(xg_pooled)  # [B,1,D,2*T]
        scale, shift = torch.chunk(xg_pooled, 2, dim=-1)  # 每个: [B,1,D,T]
        x = x * (1 + scale) + shift  # [B, P, D, T]
        x = self.out_layers(x)
        
        # 2. 添加距离信息 - 使用动态编码支持可变人数
        distances_sp = distances.permute(0, 2, 1, 3)  # [B, P, T, P]
        d_in = distances_sp.unsqueeze(-1)  # [B, P, T, P, 1]
        d_encoded = self.dis_scalar_mlp(d_in)  # [B, P, T, P, D]
        
        # 使用attention在targets维度上做加权聚合
        attn_logits = self.dis_attn(d_encoded)  # [B, P, T, P, 1]
        attn_logits = attn_logits.squeeze(-1)  # [B, P, T, P]
        
        if padding_mask is not None:
            mask_t = padding_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,P]
            attn_logits = attn_logits.masked_fill(mask_t == 0, float('-1e9'))
        
        attn_weights = torch.softmax(attn_logits, dim=3)  # [B, P, T, P]
        d_context = (d_encoded * attn_weights.unsqueeze(-1)).sum(dim=3)  # [B, P, T, D]

        # 转换到 [B, P, D, T] 格式
        distances_encoded = d_context.permute(0, 1, 3, 2)  # [B, P, D, T]
        distances_encoded = self.temp_linear(distances_encoded)
        x = x + distances_encoded
        x = self.norm(x)
        
        # 3. 更新全局特征x_global
        shift2 = self.global_emb_layers(x_clone)  # [B, P, D, T]
        x_global = x_global + shift2
        x_global = self.norm_global(x_global)
        
        return x, x_global

    
class TransMLPWithGCNStylization(nn.Module):
    """
    GCN+Stylization集成的多层MLP模块
    
    功能:
    - 多层局部MLP处理
    - 定期插入GCN和Stylization交互
    - 支持共享IPLM减少参数量
    - 可变人数场景适配
    
    Args:
        dim: 特征维度D
        seq: 序列长度T
        num_layers: MLP层数
        interaction_interval: GCN/Stylization交互间隔
        use_iplm: 是否使用IPLM模块
        iplm_interval: IPLM应用间隔
    
    输入输出: [B, P, D, T]
    """
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis,
                 interaction_interval=2,  gcn_layers=2, use_iplm=True, iplm_config=None, iplm_interval=4):
        super().__init__()
        
        self.use_iplm = use_iplm
        self.iplm_interval = iplm_interval  # 每隔多少层使用一次IPLM
        
        # 共享IPLM模块 - 所有MLPblock共享同一个实例
        if self.use_iplm and iplm_config is not None:
            from .iplm import InteractionFeatureExtractor, IPLM
            self.shared_ipm_feature_extractor = InteractionFeatureExtractor(
                feature_dim=iplm_config.feature_dim,
                hidden_dim=iplm_config.hidden_dim
            )
            self.shared_iplm = IPLM(
                knowledge_space_size=iplm_config.knowledge_space_size,
                feature_dim=iplm_config.feature_dim,
                lr=iplm_config.lr
            )
            # 将IPLM特征投影到MLP特征维度
            self.shared_ipm_to_mlp_proj = nn.Linear(iplm_config.feature_dim, dim)
        
        # 局部MLP层 - 不再包含独立的IPLM
        self.local_mlps = nn.Sequential(*[
            MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis, 
                    use_iplm=False, iplm_config=None)  # 完全禁用独立IPLM
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

    def forward(self, x, x_global=None, distances=None, padding_mask=None):
        """
        前向传播
        
        Args:
            x: [B, P, D, T] - 输入特征
            x_global: [B, P, D, T] - 全局特征(可选)
            distances: [B, T, P, P] - 人际距离矩阵
            padding_mask: [B, P] - 有效人员掩码
        
        Returns:
            x: [B, P, D, T] - 输出特征
            total_loss_lk: float - IPLM总损失 (如果启用)
        """
        b, p, d, t = x.shape
        total_loss_lk = 0.0
        global_step = 0
        
        # 初始化全局特征
        x_global = self.gcn_blocks[0](x, distances, padding_mask)  # [B, P, D, T]
        
        # 共享IPLM处理 - 只计算一次
        imp_features = None
        if self.use_iplm and distances is not None:
            # 提取交互特征
            f_ipm = self.shared_ipm_feature_extractor(distances, padding_mask)  # [B, T, D_ipm]
            f_ipmlm, loss_lk = self.shared_iplm(f_ipm)  # [B, T, D_ipm], loss
            
            if loss_lk is not None:
                total_loss_lk += loss_lk
            
            # 投影到MLP特征维度
            ipm_features = self.shared_ipm_to_mlp_proj(f_ipmlm)  # [B, T, D]
            # 广播到 [B, P, D, T] 格式
            imp_features = ipm_features.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, D, T]
        
        # 多层MLP处理与定期交互
        for i, local_layer in enumerate(self.local_mlps):
            # 选择性应用共享IPLM特征
            if imp_features is not None and (i + 1) % self.iplm_interval == 0:
                x = x + imp_features  # 残差连接融合先验知识
            
            # 局部MLP处理
            x = local_layer(x)
            
            # 定期进行GCN和Stylization交互
            if (i + 1) % self.interaction_interval == 0 and global_step < len(self.gcn_blocks):
                # GCN处理: 更新x_global (除第一次外)
                if global_step > 0:
                    x_global_gcn = self.gcn_blocks[global_step](x, distances, padding_mask)
                    x_global = x_global + x_global_gcn  # 残差连接
                
                # Stylization交互: x和x_global相互作用
                x_new, x_global_new = self.stylization_blocks[global_step](
                    x, x_global, distances, padding_mask
                )
                
                # 残差连接
                x = x + x_new
                x_global = x_global + x_global_new
                
                global_step += 1
        
        if self.use_iplm:
            return x, total_loss_lk
        else:
            return x


def build_mlps_gcn_stylization(args):
    """
    构建GCN+Stylization集成模型
    
    Args:
        args: 配置参数对象，包含以下属性:
            - hidden_dim: 特征维度
            - seq_len: 序列长度
            - num_layers: MLP层数
            - gcn_layers: GCN层数 (默认2)
            - interaction_interval: 交互间隔 (默认2)
            - use_iplm: 是否使用IPLM (默认False)
            - iplm_config: IPLM配置
            - iplm_interval: IPLM应用间隔 (默认4)
    
    Returns:
        TransMLPWithGCNStylization: 构建的模型实例
    """
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    
    gcn_layers = getattr(args, 'gcn_layers', 2)
    interaction_interval = getattr(args, 'interaction_interval', 2)
    
    # IPLM相关参数
    use_iplm = getattr(args, 'use_iplm', False)
    iplm_config = getattr(args, 'iplm_config', None) if use_iplm else None
    iplm_interval = getattr(args, 'iplm_interval', 4)
    
    return TransMLPWithGCNStylization(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
        interaction_interval=interaction_interval,
        gcn_layers=gcn_layers,
        use_iplm=use_iplm,
        iplm_config=iplm_config,
        iplm_interval=iplm_interval,
    )
