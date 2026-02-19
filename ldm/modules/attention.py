import math
from inspect import isfunction

import torch
import torch.backends
import torch.backends.cuda
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
from torch.utils import checkpoint

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("Xformers is not available. Install via ")


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def fill_inf_from_mask(self, sim, mask):
        if mask is not None:
            B, M = mask.shape
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1).reshape(B * self.heads, 1, -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim

    def forward_plain(self, x, key, value, mask=None):
        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(key)  # B*M*(H*C)
        v = self.to_v(value)  # B*M*(H*C)

        B, N, HC = q.shape
        _, M, _ = key.shape
        H = self.heads
        C = HC // H

        q = q.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        k = k.view(B, M, H, C).permute(0, 2, 1, 3).reshape(B * H, M, C)  # (B*H)*M*C
        v = v.view(B, M, H, C).permute(0, 2, 1, 3).reshape(B * H, M, C)  # (B*H)*M*C

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale  # (B*H)*N*M
        self.fill_inf_from_mask(sim, mask)
        attn = sim.softmax(dim=-1)  # (B*H)*N*M

        out = torch.einsum('b i j, b j d -> b i d', attn, v)  # (B*H)*N*C
        out = out.view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, (H * C))  # B*N*(H*C)

        return self.to_out(out)

    def forward(self, x, key, value, mask=None):
        if not XFORMERS_IS_AVAILABLE:
            return self.forward_plain(x, key, value, mask)

        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(key)  # B*M*(H*C)
        v = self.to_v(value)  # B*M*(H*C)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward_plain(self, x):
        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(x)  # B*N*(H*C)
        v = self.to_v(x)  # B*N*(H*C)

        B, N, HC = q.shape
        H = self.heads
        C = HC // H

        q = q.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        k = k.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        v = v.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1)  # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v)  # (B*H)*N*C
        out = out.view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, (H * C))  # B*N*(H*C)

        return self.to_out(out)

    def forward(self, x, context=None, mask=None):
        if not XFORMERS_IS_AVAILABLE:
            return self.forward_plain(x)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        if mask is not None:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
        super().__init__()

        self.attn = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads,
                                   dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1

    def forward(self, x, objs):
        x = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(self.norm1(x), objs, objs)
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x


class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        self.linear = nn.Linear(context_dim, query_dim)
        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        self.scale = 1

    def forward(self, x, objs, masks=None):
        N = objs.shape[1]
        B, HW, C = x.shape
        H = W = int(math.sqrt(HW))
        
        objs = self.linear(objs)
        
        M = masks.shape[1]
        tokens_per_instance = N // M
        
        masks = self._downsample_masks_robust(masks, (H, W))
        
        # masks
        masks_expanded = torch.repeat_interleave(masks, repeats=tokens_per_instance, dim=1)
        masks_expanded = masks_expanded.view(B, N, HW).bool()
        
        # attention mask
        attn_masks = torch.ones(B, HW + N, HW + N, dtype=torch.bool, device=x.device)
        
        # Visual ↔ Object
        attn_masks[:, HW:, :HW] = masks_expanded
        attn_masks[:, :HW, HW:] = masks_expanded.transpose(1, 2)
        
        # Object ↔ Object: block diagonal
        obj_mask = torch.zeros(N, N, dtype=torch.bool, device=x.device)
        for inst_id in range(M):
            start = inst_id * tokens_per_instance
            end = (inst_id + 1) * tokens_per_instance
            obj_mask[start:end, start:end] = True
        
        attn_masks[:, HW:, HW:] = obj_mask.unsqueeze(0)
        
        attn_masks = attn_masks.unsqueeze(1).repeat(1, self.attn.heads, 1, 1)
        attn_masks = attn_masks.view(B * self.attn.heads, HW + N, HW + N)
        
        combined = torch.cat([x, objs], dim=1)
        attn_out = self.attn(self.norm1(combined), mask=attn_masks)
        
        x = x + self.scale * torch.tanh(self.alpha_attn) * attn_out[:, :HW, :]
        # ===============================================
        
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))
        return x
    
    def _downsample_masks_robust(self, masks, target_size):
        H, W = masks.shape[2:]
        target_h, target_w = target_size
        
        masks = (masks > 0.5).float()
        
        ratio_h = H // target_h
        ratio_w = W // target_w
        
        if ratio_h > 1 or ratio_w > 1:
            masks = F.max_pool2d(masks, kernel_size=(ratio_h, ratio_w), stride=(ratio_h, ratio_w))
            
            if masks.shape[2:] != target_size:
                masks = F.interpolate(masks, size=target_size, mode='nearest')
            
            if target_h <= 16:
                kernel = 5 if target_h <= 8 else 3
                masks = F.max_pool2d(masks, kernel_size=kernel, stride=1, padding=kernel//2)
        else:
            masks = F.interpolate(masks, size=target_size, mode='nearest')
        
        return (masks > 0.0).float()


class DFMSpatialTransformer(nn.Module):
    def __init__(self, in_channels, key_dim, value_dim, n_heads, d_head, depth=1, 
                 fuser_type=None, use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        query_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels, query_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(query_dim, key_dim, value_dim, n_heads, d_head, fuser_type,
                                   use_checkpoint=use_checkpoint)
             for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(query_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context, objs, masks):
        """
        Args:
            x: (B, C, H, W)
            context: text embeddings
            objs: object tokens (B, N, C)
            masks: spatial masks (B, M, 512, 512)
        """
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        for block in self.transformer_blocks:
            x = block(x, context, objs, masks)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in


class BasicTransformerBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=True):
        super().__init__()
        self.attn1 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)
        self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, 
                                    heads=n_heads, dim_head=d_head)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.use_checkpoint = use_checkpoint

        if fuser_type == "gatedSA":
            self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head)
        elif fuser_type == "gatedSA2":
            self.fuser = GatedSelfAttentionDense2(query_dim, key_dim, n_heads, d_head)
        elif fuser_type == "gatedCA":
            self.fuser = GatedCrossAttentionDense(query_dim, key_dim, value_dim, n_heads, d_head)
        else:
            assert False

    def forward(self, x, context, objs, masks):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, objs, masks)
        else:
            return self._forward(x, context, objs, masks)

    def _forward(self, x, context, objs, masks):
        x = self.attn1(self.norm1(x)) + x
        x = self.fuser(x, objs, masks)
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x



class GatedSelfAttentionDense2(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1

    def forward(self, x, objs):
        B, N_visual, _ = x.shape
        B, N_ground, _ = objs.shape

        objs = self.linear(objs)

        # sanity check 
        size_v = math.sqrt(N_visual)
        size_g = math.sqrt(N_ground)
        assert int(size_v) == size_v, "Visual tokens must be square rootable"
        assert int(size_g) == size_g, "Grounding tokens must be square rootable"
        size_v = int(size_v)
        size_g = int(size_g)

        # select grounding token and resize it to visual token size as residual 
        out = self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, N_visual:, :]
        out = out.permute(0, 2, 1).reshape(B, -1, size_g, size_g)
        out = torch.nn.functional.interpolate(out, (size_v, size_v), mode='bicubic')
        residual = out.reshape(B, -1, N_visual).permute(0, 2, 1)

        # add residual to visual feature 
        x = x + self.scale * torch.tanh(self.alpha_attn) * residual
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x

