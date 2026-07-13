import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def generate_pcs_labels_numpy(road_mask: np.ndarray, r: int = 2) -> np.ndarray:
    """
    Generate Pixel Connectivity Structure (PCS) labels from a binary road mask (NumPy).
    
    Args:
        road_mask: Binary mask of shape (H, W) where 1 indicates road, 0 indicates background.
        r: Distance parameter (pixel spacing).
        
    Returns:
        pcs_label: Array of shape (8, H, W) representing connectivity in 8 directions.
    """
    H, W = road_mask.shape
    pcs_label = np.zeros((8, H, W), dtype=np.uint8)
    
    # 8 directions offsets: (dy, dx)
    # 0: Top-Left, 1: Top, 2: Top-Right, 3: Left, 4: Right, 5: Bottom-Left, 6: Bottom, 7: Bottom-Right
    directions = [
        (-r, -r), (-r, 0), (-r, r),
        (0, -r),           (0, r),
        (r, -r),  (r, 0),  (r, r)
    ]
    
    for idx, (dy, dx) in enumerate(directions):
        shifted = np.zeros_like(road_mask)
        
        y_src_start = max(0, -dy)
        y_src_end = H + min(0, -dy)
        x_src_start = max(0, -dx)
        x_src_end = W + min(0, -dx)
        
        y_dst_start = max(0, dy)
        y_dst_end = H + min(0, dy)
        x_dst_start = max(0, dx)
        x_dst_end = W + min(0, dx)
        
        shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
            road_mask[y_src_start:y_src_end, x_src_start:x_src_end]
            
        pcs_label[idx] = road_mask & shifted
        
    return pcs_label

def generate_pcs_labels_pytorch(road_mask: "torch.Tensor", r: int = 2) -> "torch.Tensor":
    """
    Generate Pixel Connectivity Structure (PCS) labels from a binary road mask (PyTorch).
    Useful inside a PyTorch Dataset or Custom Transform.
    
    Args:
        road_mask: PyTorch tensor of shape (H, W) or (1, H, W)
        r: Distance parameter.
        
    Returns:
        pcs_label: Tensor of shape (8, H, W)
    """
    if road_mask.dim() == 3:
        road_mask = road_mask.squeeze(0)
    H, W = road_mask.shape
    device = road_mask.device
    
    pcs_label = torch.zeros((8, H, W), dtype=torch.uint8, device=device)
    
    directions = [
        (-r, -r), (-r, 0), (-r, r),
        (0, -r),           (0, r),
        (r, -r),  (r, 0),  (r, r)
    ]
    
    for idx, (dy, dx) in enumerate(directions):
        shifted = torch.zeros_like(road_mask)
        
        y_src_start = max(0, -dy)
        y_src_end = H + min(0, -dy)
        x_src_start = max(0, -dx)
        x_src_end = W + min(0, -dx)
        
        y_dst_start = max(0, dy)
        y_dst_end = H + min(0, dy)
        x_dst_start = max(0, dx)
        x_dst_end = W + min(0, dx)
        
        shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
            road_mask[y_src_start:y_src_end, x_src_start:x_src_end]
            
        pcs_label[idx] = road_mask & shifted
        
    return pcs_label

def reverse_mapping_numpy(pcs_pred: np.ndarray, r: int = 2, threshold: float = 0.5) -> np.ndarray:
    """
    Reverse map the predicted 8-channel connectivity map to a 2D segmentation map (NumPy).
    
    Args:
        pcs_pred: Array of shape (8, H, W) with values in range [0, 1] (or binary).
        r: Distance parameter.
        threshold: Binarization threshold.
        
    Returns:
        seg_out: Converted binary segmentation map of shape (H, W).
    """
    H, W = pcs_pred.shape[1], pcs_pred.shape[2]
    seg_out = np.zeros((H, W), dtype=np.uint8)
    
    directions = [
        (-r, -r), (-r, 0), (-r, r),
        (0, -r),           (0, r),
        (r, -r),  (r, 0),  (r, r)
    ]
    
    for idx, (dy, dx) in enumerate(directions):
        active_y, active_x = np.where(pcs_pred[idx] >= threshold)
        
        # If connection exists, both current pixel and neighbor pixel are road pixels
        seg_out[active_y, active_x] = 1
        
        target_y = active_y + dy
        target_x = active_x + dx
        
        valid = (target_y >= 0) & (target_y < H) & (target_x >= 0) & (target_x < W)
        seg_out[target_y[valid], target_x[valid]] = 1
        
    return seg_out

def reverse_mapping_pytorch(pcs_pred: "torch.Tensor", r: int = 2, threshold: float = 0.5) -> "torch.Tensor":
    """
    Reverse map the predicted 8-channel connectivity map to a 2D segmentation map (PyTorch).
    
    Args:
        pcs_pred: Tensor of shape (8, H, W) or (B, 8, H, W) with probabilities.
        r: Distance parameter.
        threshold: Binarization threshold.
        
    Returns:
        seg_out: Binary segmentation map of shape (H, W) or (B, H, W).
    """
    is_batched = pcs_pred.dim() == 4
    if not is_batched:
        pcs_pred = pcs_pred.unsqueeze(0)  # Add batch dim: (1, 8, H, W)
        
    B, _, H, W = pcs_pred.shape
    device = pcs_pred.device
    seg_out = torch.zeros((B, H, W), dtype=torch.uint8, device=device)
    
    directions = [
        (-r, -r), (-r, 0), (-r, r),
        (0, -r),           (0, r),
        (r, -r),  (r, 0),  (r, r)
    ]
    
    for idx, (dy, dx) in enumerate(directions):
        mask = pcs_pred[:, idx] >= threshold  # (B, H, W) boolean mask
        
        # 1. Current pixels where connectivity is true
        seg_out[mask] = 1
        
        # 2. Target pixels (shifted by dy, dx)
        # Shift mask to target coordinates
        shifted_mask = torch.zeros_like(mask)
        
        y_src_start = max(0, -dy)
        y_src_end = H + min(0, -dy)
        x_src_start = max(0, -dx)
        x_src_end = W + min(0, -dx)
        
        y_dst_start = max(0, dy)
        y_dst_end = H + min(0, dy)
        x_dst_start = max(0, dx)
        x_dst_end = W + min(0, dx)
        
        shifted_mask[:, y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
            mask[:, y_src_start:y_src_end, x_src_start:x_src_end]
            
        seg_out[shifted_mask] = 1
        
    if not is_batched:
        seg_out = seg_out.squeeze(0)
        
    return seg_out

if __name__ == "__main__":
    print("Running PCS logic verification test...")
    # Create a simple mock road mask (H=10, W=10) with a diagonal road
    mock_road = np.zeros((10, 10), dtype=np.uint8)
    for i in range(10):
        mock_road[i, i] = 1  # Main diagonal
    
    print("\nOriginal road mask:")
    print(mock_road)
    
    # Generate PCS labels (distance r=2)
    pcs_labels = generate_pcs_labels_numpy(mock_road, r=2)
    print(f"\nGenerated PCS labels shape: {pcs_labels.shape}")
    
    # Let's inspect the Top-Left channel (direction idx = 0: dy=-2, dx=-2)
    print("\nTop-Left connectivity channel (should have 1s along diagonal except top-left limits):")
    print(pcs_labels[0])
    
    # Reverse map back to segmentation
    recon_road = reverse_mapping_numpy(pcs_labels, r=2)
    print("\nReconstructed road mask from PCS (should match original):")
    print(recon_road)
    
    assert np.array_equal(mock_road, recon_road), "Reconstruction failed!"
    print("\nVerification successful! NumPy implementations match.")
    if HAS_TORCH:
        # PyTorch verification
        mock_road_tensor = torch.tensor(mock_road, dtype=torch.uint8)
        pcs_labels_tensor = generate_pcs_labels_pytorch(mock_road_tensor, r=2)
        recon_road_tensor = reverse_mapping_pytorch(pcs_labels_tensor.float(), r=2)
        
        assert torch.equal(mock_road_tensor, recon_road_tensor), "PyTorch reconstruction failed!"
        print("Verification successful! PyTorch implementations match.")
    else:
        print("\nPyTorch not available. Skipping PyTorch verification tests.")

