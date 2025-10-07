import torch
from torch import nn
from einops.layers.torch import Rearrange

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

    def __init__(self, time_dim, num_p,dim, max_p=None):
        super().__init__()
        # Use max_p if provided, otherwise use num_p for backward compatibility
        effective_p = max_p if max_p is not None else num_p
        self.num_p = num_p
        self.max_p = effective_p
        
        self.emb_layers = nn.Sequential(
            nn.Linear(time_dim*effective_p, 2 * time_dim),#pt->2t
        )
        self.norm = LN(dim)
        self.norm_global=LN(dim)
        self.out_layers = nn.Sequential(
            zero_module(nn.Linear(time_dim, time_dim)),
        )
        self.global_emb_layers = nn.Sequential(
            nn.Linear(time_dim*effective_p, time_dim*effective_p),#pt->pt
        )
        
        self.temp_linear=nn.Sequential(
            nn.Linear(time_dim, time_dim),
        )
        self.dis_linear=nn.Linear(effective_p,dim)
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
    def forward(self, x, x_global,distances, padding_mask=None):
        """
        x: B, P,D,T
        x_global: B,D,PT
        distances: B,P,1,T or B,T,P,P
        padding_mask: B,P (optional, 1 for valid persons, 0 for padded)
        """
        b, p, d, t = x.shape
        # Pad x to max_p if needed
        if p < self.max_p:
            pad_size = self.max_p - p
            x_padded = torch.cat([x, torch.zeros(b, pad_size, d, t, device=x.device)], dim=1)
            if padding_mask is None:
                padding_mask = torch.cat([torch.ones(b, p, device=x.device), torch.zeros(b, pad_size, device=x.device)], dim=1)
        else:
            x_padded = x
            if padding_mask is None:
                padding_mask = torch.ones(b, p, device=x.device)
        
        # Pad distances to max_p if needed
        if distances.dim() == 4 and distances.shape[2] == 1:  # B,P,1,T
            if p < self.max_p:
                distances_padded = torch.cat([distances, torch.zeros(b, self.max_p - p, 1, t, device=distances.device)], dim=1)
            else:
                distances_padded = distances
        elif distances.dim() == 4:  # B,T,P,P
            if p < self.max_p:
                pad_size = self.max_p - p
                distances_padded = torch.cat([
                    torch.cat([distances, torch.zeros(b, t, pad_size, p, device=distances.device)], dim=2),
                    torch.zeros(b, t, self.max_p, pad_size, device=distances.device)
                ], dim=3)
            else:
                distances_padded = distances
        else:
            distances_padded = distances
            
        post_x=False
        if post_x:#先更新x，用更新后的x更新x_global
            x_global_clone=x_global.clone().unsqueeze(1)#B,1,D,PT_max
            x_global_clone = self.emb_layers(x_global_clone)#b,1,d,2t
            # scale: B,1, d, t / shift: B,1, d, t
            scale, shift = torch.chunk(x_global_clone, 2, dim=-1)
            x_padded = x_padded* (1 + scale) + shift#B,P_max,D,T
            x_padded = self.out_layers(x_padded)
            x_padded=self.norm(x_padded)#B,P_max,D,T
            
            # Apply padding mask
            if padding_mask is not None:
                mask = padding_mask.view(b, self.max_p, 1, 1)
                x_padded = x_padded * mask
            
            x_clone=x_padded.clone().transpose(1,2).flatten(-2)#B,D,P_max*T
            shift2=self.global_emb_layers(x_clone)#B,D,P_max*T
            x_global=x_global+shift2
            x_global=self.norm_global(x_global)
        else:#同步更新，用更新前的x更新x_global
            x_clone=x_padded.clone().transpose(1,2).flatten(-2)#B,D,P_max*T
            x_global_clone=x_global.clone().unsqueeze(1)#B,1,D,P_max*T
            
            x_global_clone = self.emb_layers(x_global_clone)#b,1,d,2t
            # scale: B,1, d, t / shift: B,1, d, t
            scale, shift = torch.chunk(x_global_clone, 2, dim=-1)
            x_padded = x_padded* (1 + scale) + shift#B,P_max,D,T
            x_padded = self.out_layers(x_padded)
            
            if distances_padded.dim() == 4 and distances_padded.shape[1] == t:  # B,T,P,D
                distances_processed=self.dis_linear(distances_padded.transpose(2, 3))#B,T,D,P_max
                distances1=distances_processed.permute(0,3,2,1)#B,P_max,D,T
            else:  # B,P,1,T
                distances_processed=self.dis_linear(distances_padded.transpose(1,3))#B,T,D,P_max
                distances1=distances_processed.permute(0,3,2,1)#B,P_max,D,T
            
            distances1=self.temp_linear(distances1)#B,P_max,D,T
            x_padded=x_padded+distances1
            
            # Apply padding mask
            if padding_mask is not None:
                mask = padding_mask.view(b, self.max_p, 1, 1)
                x_padded = x_padded * mask
                
            x_padded=self.norm(x_padded)#B,P_max,D,T
            
            shift2=self.global_emb_layers(x_clone)#B,D,P_max*T
            x_global=x_global+shift2
            x_global=self.norm_global(x_global)
            
        # Extract only valid persons
        x_out = x_padded[:, :p, :, :]
        return x_out, x_global
    
class TransMLP(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis,interaction_interval=2,p=3,max_p=None):
        super().__init__()
        # Use max_p if provided, otherwise use p for backward compatibility
        effective_p = max_p if max_p is not None else p
        self.p = p
        self.max_p = effective_p
        
        self.local_mlps = nn.Sequential(*[
            MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers)])
        self.global_mlps=nn.Sequential(*[
            MLPblock(dim, seq*effective_p, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers//interaction_interval)])
        self.stylization_blocks = nn.ModuleList([
            StylizationBlock(seq, p, dim, max_p=effective_p) for _ in range(num_layers//interaction_interval)
        ])
        self.interaction_interval=interaction_interval
    def forward(self, x,distances, padding_mask=None):#distances:B,T,P,P, padding_mask: B,P
        b,p,d,t=x.shape
        
        # Create padding mask if not provided
        if padding_mask is None:
            padding_mask = torch.ones(b, p, device=x.device)
        
        # Pad x and padding_mask to max_p if needed
        if p < self.max_p:
            pad_size = self.max_p - p
            x_padded = torch.cat([x, torch.zeros(b, pad_size, d, t, device=x.device)], dim=1)
            padding_mask_padded = torch.cat([padding_mask, torch.zeros(b, pad_size, device=x.device)], dim=1)
        else:
            x_padded = x
            padding_mask_padded = padding_mask
            
        # 初始化 x_global 与 x 一样
        x_global = x_padded.clone().transpose(1,2).flatten(-2)#B,D,P_max*T
        global_step = 0
        
        # 逐层进行local和global的交互
        for i, local_layer in enumerate(self.local_mlps):
            # Only apply local layer to actual persons
            x_temp = x_padded[:, :p, :, :]
            x_temp = local_layer(x_temp)  # 计算 local MLP
            x_padded = torch.cat([x_temp, x_padded[:, p:, :, :]], dim=1)

            # 每经过 interaction_interval 层 local_mlp，执行一次 global_mlp 的更新和交互
            if (i + 1) % self.interaction_interval == 0 and global_step < len(self.global_mlps):
                x_global = self.global_mlps[global_step](x_global)  # 动态计算 global MLP
                x_new, x_global_new = self.stylization_blocks[global_step](x_padded[:, :p, :, :], x_global, distances, padding_mask)  # 使用 StylizationBlock 进行交互
                
                # Update only actual persons
                x_padded_temp = x_padded.clone()
                x_padded_temp[:, :p, :, :] = x_padded[:, :p, :, :] + x_new
                x_padded = x_padded_temp
                x_global=x_global+x_global_new
                
                global_step += 1
        
        # Return only valid persons
        return x_padded[:, :p, :, :]

def build_mlps(args):
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    
    # Support variable person count with max_p parameter
    max_p = getattr(args, 'max_p', None)
    
    return TransMLP(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
        interaction_interval=args.interaction_interval,
        p=args.n_p,
        max_p=max_p,
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


