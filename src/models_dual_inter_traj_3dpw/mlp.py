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
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
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
        self.max_p = num_p  # Store max number of people
        self.time_dim = time_dim
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
        x: B, P,D,T where P can be <= max_p
        x_global: B,D,PT where P can be <= max_p
        distances: B,P,1,T where P can be <= max_p
        """
        B, P, D, T = x.shape
        post_x=False
        
        # Handle variable person count by padding to max_p if needed
        if P < self.max_p:
            # Pad x with zeros to match max_p
            padding = torch.zeros(B, self.max_p - P, D, T, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            
            # Pad x_global accordingly
            x_global_padded = torch.zeros(B, D, self.max_p * T, device=x_global.device, dtype=x_global.dtype)
            x_global_padded[:, :, :P*T] = x_global
            
            # Pad distances
            distances_padding = torch.zeros(B, self.max_p - P, 1, T, device=distances.device, dtype=distances.dtype)
            distances_padded = torch.cat([distances, distances_padding], dim=1)
        else:
            x_padded = x
            x_global_padded = x_global
            distances_padded = distances
            
        if post_x:#先更新x，用更新后的x更新x_global
            x_global_clone=x_global_padded.clone().unsqueeze(1)#B,1,D,max_p*T
            x_global_clone = self.emb_layers(x_global_clone)#b,1,d,2t
            # scale: B,1, d, t / shift: B,1, d, t
            scale, shift = torch.chunk(x_global_clone, 2, dim=-1)
            x_padded = x_padded* (1 + scale) + shift#B,max_p,D,T
            x_padded = self.out_layers(x_padded)
            x_padded=self.norm(x_padded)#B,max_p,D,T
            
            x_clone=x_padded.clone().transpose(1,2).flatten(-2)#B,D,max_p*T
            shift2=self.global_emb_layers(x_clone)#B,D,max_p*T
            x_global_padded=x_global_padded+shift2
            x_global_padded=self.norm_global(x_global_padded)
        else:#同步更新，用更新前的x更新x_global
            x_clone=x_padded.clone().transpose(1,2).flatten(-2)#B,D,max_p*T
            x_global_clone=x_global_padded.clone().unsqueeze(1)#B,1,D,max_p*T
            
            x_global_clone = self.emb_layers(x_global_clone)#b,1,d,2t
            # scale: B,1, d, t / shift: B,1, d, t
            scale, shift = torch.chunk(x_global_clone, 2, dim=-1)
            x_padded = x_padded* (1 + scale) + shift#B,max_p,D,T
            x_padded = self.out_layers(x_padded)
            
            distances_padded=self.dis_linear(distances_padded)#B,max_p,1,D
            distances1=distances_padded.permute(0,2,3,1)#B,1,D,max_p -> should be B,max_p,D,T
            # distances1=self.temp_norm(distances1)
            distances1=self.temp_linear(distances1)#B,max_p,D,T
            # scale_temp,shift_temp=torch.chunk(distances1,2,dim=-1)
            # x=x*(1+scale_temp)+shift_temp
            # x=self.out_layers_temp(x)   
            x_padded=x_padded+distances1
            x_padded=self.norm(x_padded)#B,max_p,D,T
            
            shift2=self.global_emb_layers(x_clone)#B,D,max_p*T
            x_global_padded=x_global_padded+shift2
            
            # distances_for_global=distances.transpose(1,2).flatten(-2)#B,1,P*T
            # distances2=self.temp_linear2(distances_for_global)#B,1,P*T
            # x_global=x_global+distances2#B,D,PT
            
            x_global_padded=self.norm_global(x_global_padded)
        
        # Extract only the valid person dimensions
        x_out = x_padded[:, :P, :, :]
        x_global_out = x_global_padded[:, :, :P*T]
        
        return x_out, x_global_out
    
class TransMLP(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis,num_global_layers=4,interaction_interval=4,p=3):
        super().__init__()
        self.max_p = p  # Store max number of people
        self.seq = seq
        self.local_mlps = nn.Sequential(*[
            MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers)])
        self.global_mlps=nn.Sequential(*[
            MLPblock(dim, seq*p, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_global_layers)])
        self.stylization_blocks = nn.ModuleList([
            StylizationBlock(seq, p,dim) for _ in range(interaction_interval)
        ])
        self.interaction_interval=interaction_interval
        self.num_layers=num_layers
        self.a=self.num_layers//self.interaction_interval
        self.b=num_global_layers//interaction_interval
    def forward(self, x,distances):#distances:B,T,P,P
        # x: B, P, D, T where P can be <= max_p
        B, P, D, T = x.shape
        # 初始化 x_global 与 x 一样
        x_global = x.clone().transpose(1,2).flatten(-2)#B,D,P*T
        
        for j in range(self.interaction_interval):
            # 逐层进行local和global的交互
            for i, local_layer in enumerate(self.local_mlps[self.a*j:self.a*j+self.a]):
                x = local_layer(x)  # 计算 local MLP
            for i, global_layer in enumerate(self.global_mlps[self.b*j:self.b*j+self.b]):
                # Handle variable person count for global_mlp
                if P < self.max_p:
                    # Pad x_global to max_p*T for global_mlp processing
                    x_global_padded = torch.zeros(B, D, self.max_p * T, device=x_global.device, dtype=x_global.dtype)
                    x_global_padded[:, :, :P*T] = x_global
                    x_global_padded = global_layer(x_global_padded)
                    # Extract only valid part
                    x_global = x_global_padded[:, :, :P*T]
                else:
                    x_global = global_layer(x_global)  # 计算 global MLP
                            
            x_new, x_global_new = self.stylization_blocks[j](x, x_global,distances)  # 使用 StylizationBlock 进行交互
            x=x+x_new
            x_global=x_global+x_global_new
                    
        # return x+0.2*x_global.reshape(b,d,p,t).transpose(1,2)
        return x

def build_mlps(args):
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    
    # Support both n_p (fixed) and max_p (maximum for variable person count)
    max_persons = getattr(args, 'max_p', getattr(args, 'n_p', 3))
    
    return TransMLP(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
        num_global_layers=args.num_global_layers,
        interaction_interval=args.interaction_interval,
        p=max_persons,
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


