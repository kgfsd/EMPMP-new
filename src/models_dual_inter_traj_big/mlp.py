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
        #B,P,D,T
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
        # self.arr0 = Rearrange('b n d -> b d n')
        # self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        # x = self.arr0(x)
        x = x.transpose(-2, -1)  # 转置最后两个维度
        x = self.fc(x)
        x = x.transpose(-2, -1)  # 恢复最后两个维度的顺序
        # x = self.arr1(x)
        return x

class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class MLPblock(nn.Module):

    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial'):
        super().__init__()

        if not use_spatial_fc:
            self.fc0 = Temporal_FC(seq)
        else:
            self.fc0 = Spatial_FC(dim)

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        # nn.init.xavier_uniform_(self.fc0.fc.weight)
        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x):

        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_

        return x
    
class MLPblock(nn.Module):

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
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc0.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc1.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc1.fc.bias, 0)
        
    def forward(self, x):
        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_

        x__=self.fc1(x)
        x__=self.norm1(x__)
        x=x+x__
        return x
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class StylizationBlock(nn.Module):

    def __init__(self, time_dim, num_p,dim):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.Linear(time_dim*num_p, 2 * time_dim),#pt->2t
        )
        self.norm = LN(dim)
        self.norm_global=LN(dim)
        self.out_layers = nn.Sequential(
            zero_module(nn.Linear(time_dim, time_dim)),
        )
        self.global_emb_layers = nn.Sequential(
            nn.Linear(time_dim*num_p, time_dim*num_p),#pt->pt
        )
        
        self.temp_linear=nn.Sequential(
            nn.Linear(time_dim, time_dim),
        )
        self.dis_linear=nn.Linear(num_p,dim)
        # self.temp_linear2=nn.Sequential(
        #     zero_module(nn.Linear(dim+1, dim)),
        # )
        # self.out_layers_temp = nn.Sequential(
        #     zero_module(nn.Linear(time_dim, time_dim)),
        # )
        # self.temp_linear2=nn.Sequential(
        #     nn.Linear(time_dim*num_p,time_dim*num_p)
        # )
        # self.temp_norm=LN(dim)
        
        # self.temp_linear2=nn.Sequential(
        #     zero_module(nn.Linear(time_dim, time_dim)),
        # )
        # self.temp_norm2=LN(dim)
    def forward(self, x, x_global,distances):
        """
        x: B, P,D,T
        x_global: B,D,PT
        distances: B,P,1,T
        """
        post_x=False
        if post_x:#先更新x，用更新后的x更新x_global
            x_global_clone=x_global.clone().unsqueeze(1)#B,1,D,PT
            x_global_clone = self.emb_layers(x_global_clone)#b,1,d,2t
            # scale: B,1, d, t / shift: B,1, d, t
            scale, shift = torch.chunk(x_global_clone, 2, dim=-1)
            x = x* (1 + scale) + shift#B,P,D,T
            x = self.out_layers(x)
            x=self.norm(x)#B,P,D,T
            
            x_clone=x.clone().transpose(1,2).flatten(-2)#B,D,PT
            shift2=self.global_emb_layers(x_clone)#B,D,PT
            x_global=x_global+shift2
            x_global=self.norm_global(x_global)
        else:#同步更新，用更新前的x更新x_global
            x_clone=x.clone().transpose(1,2).flatten(-2)#B,D,PT
            x_global_clone=x_global.clone().unsqueeze(1)#B,1,D,PT
            
            x_global_clone = self.emb_layers(x_global_clone)#b,1,d,2t
            # scale: B,1, d, t / shift: B,1, d, t
            scale, shift = torch.chunk(x_global_clone, 2, dim=-1)
            x = x* (1 + scale) + shift#B,P,D,T
            x = self.out_layers(x)
            
            distances=self.dis_linear(distances)#B,T,P,D
            distances1=distances.permute(0,2,3,1)#B,P,D,T
            # distances1=self.temp_norm(distances1)
            distances1=self.temp_linear(distances1)#B,P,D,T
            # scale_temp,shift_temp=torch.chunk(distances1,2,dim=-1)
            # x=x*(1+scale_temp)+shift_temp
            # x=self.out_layers_temp(x)   
            x=x+distances1
            x=self.norm(x)#B,P,D,T
            
            shift2=self.global_emb_layers(x_clone)#B,D,PT
            x_global=x_global+shift2
            
            # distances_for_global=distances.transpose(1,2).flatten(-2)#B,1,P*T
            # distances2=self.temp_linear2(distances_for_global)#B,1,P*T
            # x_global=x_global+distances2#B,D,PT
            
            x_global=self.norm_global(x_global)
            
        return x,x_global



    """
    Enhanced StylizationBlock with GCN for handling variable numbers of people.
    Uses dynamic graph construction based on Euclidean distances.
    """
    def __init__(self, time_dim, num_p, dim,gcn_layers=2):
        super().__init__()
        # Original stylization components
        self.time_dim = time_dim
        self.num_p = num_p
        self.dim = dim
        
        self.norm = LN(dim)
        self.norm_global = LN(dim)
        self.out_layers = nn.Sequential(
            zero_module(nn.Linear(time_dim, time_dim)),
        )
        
        # GCN for modeling spatial relationships
        self.gcn = DynamicGCN(
            dim=dim,
            num_layers=gcn_layers,
        )
        
        # Linear layer to blend GCN output
        self.gcn_blend = nn.Linear(dim, dim)
        
        # Cache for dynamic layers
        self._emb_layer_cache = {}
        self._global_emb_layer_cache = {}
        
    def _get_emb_layer(self, seq_len):
        """动态创建或获取嵌入层"""
        if seq_len not in self._emb_layer_cache:
            self._emb_layer_cache[seq_len] = nn.Sequential(
                nn.Linear(seq_len, 2 * self.time_dim),
            ).to(next(self.parameters()).device)
        return self._emb_layer_cache[seq_len]
    
    def _get_global_emb_layer(self, seq_len):
        """动态创建或获取全局嵌入层"""
        if seq_len not in self._global_emb_layer_cache:
            self._global_emb_layer_cache[seq_len] = nn.Sequential(
                nn.Linear(seq_len, seq_len),
            ).to(next(self.parameters()).device)
        return self._global_emb_layer_cache[seq_len]
        
    def forward(self, x, x_global, distances, padding_mask=None):
        """
        x: B, P, D, T - 只包含有效的人
        x_global: B, D, PT - 对应有效人数的全局特征
        distances: B, T, P_total, P_total - 完整的距离矩阵
        padding_mask: B, P_total - 完整的 mask
        """
        b, p_actual, d, t = x.shape  # p_actual 是实际的人数
        
        # Apply GCN to capture spatial relationships between people
        # 我们需要从完整的距离矩阵中提取有效部分
        if padding_mask is not None:
            # 构建一个只包含有效人员的 mask
            valid_mask = padding_mask[:, :p_actual]  # [B, P_actual]
        else:
            valid_mask = None
            
        # 提取有效距离矩阵
        distances_valid = distances[:, :, :p_actual, :p_actual]  # [B, T, P_actual, P_actual]
        
        x_gcn = self.gcn(x, distances_valid, valid_mask)  # [B, P_actual, D, T]
        x_gcn = self.gcn_blend(x_gcn.transpose(-2, -1)).transpose(-2, -1)  # Blend GCN output
        
        # Original stylization logic
        x_clone = x.clone().transpose(1, 2).flatten(-2)  # B, D, P_actual*T
        seq_len = p_actual * t
        
        # 动态获取嵌入层
        emb_layer = self._get_emb_layer(seq_len)
        global_emb_layer = self._get_global_emb_layer(seq_len)
        
        x_global_clone = x_global.clone().unsqueeze(1)  # B, 1, D, P_actual*T
        
        x_global_clone = emb_layer(x_global_clone)  # b, 1, d, 2*time_dim
        scale, shift = torch.chunk(x_global_clone, 2, dim=-1)
        x = x * (1 + scale) + shift  # B, P_actual, D, T
        x = self.out_layers(x)
        
        # Integrate GCN output
        x = x + x_gcn
        x = self.norm(x)  # B, P_actual, D, T
        
        # Update global features
        shift2 = global_emb_layer(x_clone)  # B, D, P_actual*T
        x_global = x_global + shift2
        x_global = self.norm_global(x_global)
        
        return x, x_global

    
class TransMLP(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis,interaction_interval=2,p=3):
        super().__init__()
        self.local_mlps = nn.Sequential(*[
            MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers)])
        # 动态全局 MLP - 保持原有固定维度的行为
        self.global_mlps=nn.Sequential(*[
            MLPblock(dim, seq*p, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers//interaction_interval)])
        self.stylization_blocks = nn.ModuleList([
            StylizationBlock(seq, p,dim) for _ in range(num_layers//interaction_interval)
        ])
        self.interaction_interval=interaction_interval
    def forward(self, x, distances, padding_mask=None):#distances:B,T,P,P
        b,p,d,t=x.shape
        # 初始化 x_global 与 x 一样
        x_global = x.clone().transpose(1,2).flatten(-2)#B,D,PT
        global_step = 0
        
        # distances=torch.mean(distances,dim=-1,keepdim=True)#B,T,P,1
        # distances=distances.permute(0,2,3,1)#B,P,1,T
        # 逐层进行local和global的交互
        for i, local_layer in enumerate(self.local_mlps):
            x = local_layer(x)  # 计算 local MLP

            # 每经过 interaction_interval 层 local_mlp，执行一次 global_mlp 的更新和交互
            if (i + 1) % self.interaction_interval == 0 and global_step < len(self.global_mlps):
                x_global = self.global_mlps[global_step](x_global)  # 动态计算 global MLP
                x_new, x_global_new = self.stylization_blocks[global_step](x, x_global, distances)  # 使用 StylizationBlock 进行交互
                
                x=x+x_new
                x_global=x_global+x_global_new
                
                global_step += 1
        
        return x



    """
    Enhanced TransMLP with GCN-based global flow module.
    Supports variable numbers of people through dynamic graph construction.
    """
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis, 
                 interaction_interval=2, p=3,gcn_layers=2):
        super().__init__()
        self.local_mlps = nn.Sequential(*[
            MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers)])
        
        # 动态全局 MLP - 不预定义维度
        self.num_global_layers = num_layers // interaction_interval
        self.global_layer_config = {
            'dim': dim,
            'use_norm': use_norm,
            'use_spatial_fc': use_spatial_fc,
            'layernorm_axis': layernorm_axis
        }
        
        # Use GCN-based stylization blocks
        self.stylization_blocks = nn.ModuleList([
            GCNStylizationBlock(seq, p, dim, gcn_layers=gcn_layers)
            for _ in range(num_layers//interaction_interval)
        ])
        self.interaction_interval = interaction_interval
        self.seq = seq
        
        # Cache for dynamic global MLPs
        self._global_mlp_cache = {}
        
    def _get_global_mlp(self, seq_len):
        """动态创建或获取全局 MLP"""
        if seq_len not in self._global_mlp_cache:
            self._global_mlp_cache[seq_len] = nn.Sequential(*[
                MLPblock(
                    self.global_layer_config['dim'], 
                    seq_len, 
                    self.global_layer_config['use_norm'], 
                    self.global_layer_config['use_spatial_fc'], 
                    self.global_layer_config['layernorm_axis']
                )
                for _ in range(self.num_global_layers)
            ]).to(next(self.parameters()).device)
        return self._global_mlp_cache[seq_len]
        
    def forward(self, x, distances, padding_mask=None):
        """
        x: [B, P, D, T] - local features
        distances: [B, T, P, P] - pairwise distances
        padding_mask: [B, P] - bool mask indicating real people
        """
        b, p, d, t = x.shape
        
        # 计算有效人数
        if padding_mask is not None:
            actual_people = padding_mask.sum(dim=1).max().item()  # 批次中的最大人数
        else:
            actual_people = p
        
        # Initialize x_global with actual dimensions
        x_global = x[:, :actual_people].clone().transpose(1, 2).flatten(-2)  # B, D, actual_P*T
        actual_seq_len = actual_people * t
        
        # Get or create global MLP for current sequence length
        global_mlps = self._get_global_mlp(actual_seq_len)
        global_step = 0
        
        # Process layers with periodic global interactions
        for i, local_layer in enumerate(self.local_mlps):
            x = local_layer(x)
            
            if (i + 1) % self.interaction_interval == 0 and global_step < len(global_mlps):
                x_global = global_mlps[global_step](x_global)
                
                # 对于 stylization，我们需要处理维度匹配
                x_current = x[:, :actual_people]  # 只处理有效的人
                x_new, x_global_new = self.stylization_blocks[global_step](x_current, x_global, distances, padding_mask)
                
                # 更新有效部分 - 避免就地操作
                x_updated = x.clone()
                x_updated[:, :actual_people] = x_current + x_new
                x = x_updated
                x_global = x_global + x_global_new
                
                global_step += 1
        
        return x


def build_mlps(args):
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    
    # Check if GCN-based approach should be used
    use_gcn = getattr(args, 'use_gcn', False)
    return TransMLP(
            dim=args.hidden_dim,
            seq=seq_len,
            use_norm=args.with_normalization,
            use_spatial_fc=args.spatial_fc_only,
            num_layers=args.num_layers,
            layernorm_axis=args.norm_axis,
            interaction_interval=args.interaction_interval,
            p=args.n_p,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    if activation == 'silu':
        return nn.SiLU
    #if activation == 'swish':
    #    return nn.Hardswish
    if activation == 'softplus':
        return nn.Softplus
    if activation == 'tanh':
        return nn.Tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_norm_fn(norm):
    if norm == "batchnorm":
        return nn.BatchNorm1d
    if norm == "layernorm":
        return nn.LayerNorm
    if norm == 'instancenorm':
        return nn.InstanceNorm1d
    raise RuntimeError(F"norm should be batchnorm/layernorm, not {norm}.")


