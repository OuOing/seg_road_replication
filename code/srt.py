import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    # Fallback placeholders for static code analysis
    class Dummy:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return self

    class nn:
        Module = Dummy
        ModuleList = list
        Linear = Dummy
        Conv2d = Dummy
        LayerNorm = Dummy
        Dropout = Dummy
        GELU = Dummy

    HAS_TORCH = False

class SpatialReductionAttention(nn.Module):
    """
    Spatial Reduction Attention (SRA) block from SegFormer & Seg-Road.
    Reduces self-attention complexity from O(N^2) to O(N^2 / r^2) by downsampling Key and Value.
    """
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # Paper's Reshape(x, r) * W formulation can be efficiently implemented 
            # using a 2D convolution with kernel size and stride equal to sr_ratio.
            # This extracts features and reduces resolution simultaneously.
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # q shape: (B, num_heads, N, head_dim)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # Convert sequence back to 2D image shape: (B, C, H, W)
            x_2d = x.transpose(1, 2).reshape(B, C, H, W)
            # Apply Spatial Reduction convolution: (B, C, H/r, W/r)
            x_reduced = self.sr(x_2d)
            # Reshape back to sequence: (B, N_reduced, C)
            x_reduced = x_reduced.flatten(2).transpose(1, 2)
            # Apply LayerNorm
            x_reduced = self.norm(x_reduced)
            kv = self.kv(x_reduced).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # k, v shapes: (B, num_heads, N_reduced, head_dim)
        k, v = kv[0], kv[1]

        # Scaled dot-product attention
        # attn shape: (B, num_heads, N, N_reduced)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output shape: (B, num_heads, N, head_dim) -> transpose -> reshape to (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MixFFN(nn.Module):
    """
    Mix Feed-Forward Network (MixFFN) with a 3x3 depthwise convolution
    to inject local positional information (eliminates explicit positional encoding).
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # Depthwise convolution: group=hidden_features
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        
        # Reshape to 2D for depthwise convolution
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).view(B, C, H, W)
        x_2d = self.dwconv(x_2d)
        
        # Reshape back to 1D sequence
        x = x_2d.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SRTBlock(nn.Module):
    """
    A single block of Spatial Reduction Transformer (SRT).
    Composed of SRA followed by MixFFN with residual connections.
    """
    def __init__(self, dim, num_heads, sr_ratio=1, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(
            dim, num_heads=num_heads, sr_ratio=sr_ratio, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(
            in_features=dim, hidden_features=int(dim * mlp_ratio), 
            act_layer=nn.GELU, drop=drop
        )

    def forward(self, x, H, W):
        # Attention + Residual Connection
        x = x + self.attn(self.norm1(x), H, W)
        # MLP + Residual Connection
        x = x + self.mlp(self.norm2(x), H, W)
        return x

class SRTEncoderStage(nn.Module):
    """
    A stage of the Hierarchical Transformer Encoder.
    Includes Patch Merging (represented by overlapping Conv2d patch embedding) 
    followed by multiple SRTBlocks.
    """
    def __init__(self, in_chans, embed_dim, num_blocks, num_heads, sr_ratio, patch_size=4, stride=4):
        super().__init__()
        # Patch embedding: shrinks H and W by `stride` factor, projects channels to `embed_dim`
        # Seg-Road uses a convolutional structure instead of hard patch partition (as detailed on page 6)
        padding = patch_size // 2  # To keep size consistent with division
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.blocks = nn.ModuleList([
            SRTBlock(dim=embed_dim, num_heads=num_heads, sr_ratio=sr_ratio)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x input shape: (B, C_in, H_in, W_in)
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        # Flatten to sequence: (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        for block in self.blocks:
            x = block(x, H, W)
            
        # Reshape back to 2D feature map format: (B, C, H, W)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x, H, W

if __name__ == "__main__":
    print("SRT script loaded.")
    if HAS_TORCH:
        print("Running PyTorch SRT module sanity check...")
        # Create a mock input tensor: Batch=2, Channels=3, H=512, W=512
        x = torch.randn(2, 3, 512, 512)
        
        # Define an encoder stage that does a 4x reduction (patch_size=4, stride=4)
        # Input channels: 3, output embed_dim: 32, blocks: 2, heads: 4, spatial reduction ratio: 8
        stage = SRTEncoderStage(in_chans=3, embed_dim=32, num_blocks=2, num_heads=4, sr_ratio=8, patch_size=7, stride=4)
        
        out, H, W = stage(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape} (H={H}, W={W})")
        
        # Verify resolution reduction: 512 / 4 = 128 (approx depending on padding)
        assert out.shape == (2, 32, 128, 128), f"Unexpected output shape: {out.shape}"
        print("SRT stage verification successful!")
    else:
        print("PyTorch not installed. Skipping live verification tests.")
