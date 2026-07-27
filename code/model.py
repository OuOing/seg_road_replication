try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from srt import SRTEncoderStage
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
        BatchNorm2d = Dummy
        GroupNorm = Dummy
        LayerNorm = Dummy
        Dropout = Dummy
        GELU = Dummy
        Sequential = Dummy
        ReLU = Dummy
    SRTEncoderStage = Dummy
    HAS_TORCH = False


class SegRoadDecoder(nn.Module):
    """
    CNN-based Decoder for Seg-Road.
    Fuses multi-scale feature maps from 4 stages of the Encoder by:
    1. Upsampling each feature map to 1/4 of the input image size.
    2. Concatenating them channel-wise.
    3. Applying Conv-BN-ReLU fusion.
    4. Predicting Segmentation (1 or 2 channels) and Pixel Connectivity (8 channels).
    """
    def __init__(self, encoder_dims, decoder_dim=128, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        
        # 1x1 convolutions to project different stage dimensions to decoder_dim
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, decoder_dim, kernel_size=1) for dim in encoder_dims
        ])
        
        # Fusion conv: processes concatenated projected features (decoder_dim * 4 channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, decoder_dim),
            nn.ReLU(inplace=True)
        )
        
        # Segmentation head
        self.seg_head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        
        # Pixel Connectivity Structure (PCS) head - 8 orientations
        self.pcs_head = nn.Conv2d(decoder_dim, 8, kernel_size=1)

    def forward(self, features):
        # features list: [Stage1_feat, Stage2_feat, Stage3_feat, Stage4_feat]
        # Target shape for upsampling is the shape of Stage 1 (1/4 of input size)
        target_h, target_w = features[0].shape[2], features[0].shape[3]
        
        projected = []
        for i, feat in enumerate(features):
            # Project to decoder_dim
            x = self.proj[i](feat)
            if i > 0:
                # Bilinear upsampling to match Stage 1 spatial dimensions
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            projected.append(x)
            
        # Concatenate channel-wise: (B, decoder_dim * 4, target_h, target_w)
        x_concat = torch.cat(projected, dim=1)
        
        # Fuse channels: (B, decoder_dim, target_h, target_w)
        x_fused = self.fusion_conv(x_concat)
        
        # Outputs
        seg_out = self.seg_head(x_fused)
        pcs_out = self.pcs_head(x_fused)
        
        # Upsample both outputs back to the input image size (4x upsampling)
        # Note: The calling model class will handle this upsampling to the exact input size.
        return seg_out, pcs_out

class SegRoad(nn.Module):
    """
    Seg-Road model composed of Hierarchical SRT Encoder stages and CNN Decoder.
    """
    def __init__(self, in_chans=3, num_classes=1, model_size='s'):
        super().__init__()
        
        # Model config variants (s, m, l) based on parameter budget
        configs = {
            's': {
                'dims': [32, 64, 160, 256],
                'blocks': [2, 2, 2, 2],
                'heads': [1, 2, 5, 8],
                'sr_ratios': [8, 4, 2, 1]
            },
            'm': {
                'dims': [64, 128, 320, 512],
                'blocks': [3, 4, 6, 3],
                'heads': [1, 2, 5, 8],
                'sr_ratios': [8, 4, 2, 1]
            },
            'l': {
                'dims': [64, 128, 320, 512],
                'blocks': [3, 4, 18, 3],
                'heads': [1, 2, 5, 8],
                'sr_ratios': [8, 4, 2, 1]
            }
        }
        
        cfg = configs.get(model_size, configs['s'])
        dims = cfg['dims']
        blocks = cfg['blocks']
        heads = cfg['heads']
        sr_ratios = cfg['sr_ratios']
        
        # Stage 1: Conv kernel_size=11, stride=4, output size: H/4, W/4
        self.stage1 = SRTEncoderStage(in_chans, dims[0], blocks[0], heads[0], sr_ratios[0], patch_size=11, stride=4)
        # Stage 2: patch_size=3, stride=2, output size: H/8, W/8
        self.stage2 = SRTEncoderStage(dims[0], dims[1], blocks[1], heads[1], sr_ratios[1], patch_size=3, stride=2)
        # Stage 3: patch_size=3, stride=2, output size: H/16, W/16
        self.stage3 = SRTEncoderStage(dims[1], dims[2], blocks[2], heads[2], sr_ratios[2], patch_size=3, stride=2)
        # Stage 4: patch_size=3, stride=2, output size: H/32, W/32
        self.stage4 = SRTEncoderStage(dims[2], dims[3], blocks[3], heads[3], sr_ratios[3], patch_size=3, stride=2)
        
        # Decoder
        self.decoder = SegRoadDecoder(encoder_dims=dims, decoder_dim=128, num_classes=num_classes)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        
        # Encoder passes
        f1, _, _ = self.stage1(x)
        f2, _, _ = self.stage2(f1)
        f3, _, _ = self.stage3(f2)
        f4, _, _ = self.stage4(f3)
        
        # Decoder pass (returns 1/4 size predictions)
        seg_out, pcs_out = self.decoder([f1, f2, f3, f4])
        
        # Upsample back to original resolution
        seg_out = F.interpolate(seg_out, size=(H, W), mode='bilinear', align_corners=False)
        pcs_out = F.interpolate(pcs_out, size=(H, W), mode='bilinear', align_corners=False)
        
        return seg_out, pcs_out

if __name__ == "__main__":
    print("Seg-Road model script loaded.")
    if HAS_TORCH:
        print("Running Seg-Road model check...")
        # Create a mock input tensor: Batch=2, Channels=3, H=512, W=512
        x = torch.randn(2, 3, 512, 512)
        
        # Instantiate small Seg-Road-s model
        model = SegRoad(in_chans=3, num_classes=1, model_size='s')
        
        seg, pcs = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Seg output shape: {seg.shape} (should be (2, 1, 512, 512))")
        print(f"PCS output shape: {pcs.shape} (should be (2, 8, 512, 512))")
        
        assert seg.shape == (2, 1, 512, 512), "Segmentation output shape mismatch!"
        assert pcs.shape == (2, 8, 512, 512), "PCS output shape mismatch!"
        print("Model verification successful!")
    else:
        print("PyTorch not installed. Skipping live verification tests.")
