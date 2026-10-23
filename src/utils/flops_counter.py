# utils/flops_counter.py
import torch
import torch.nn as nn


def count_flops(model, input_shape, device='cuda'):
    """
    Count FLOPs for a model.
    
    Args:
        model: PyTorch model
        input_shape: tuple (batch_size, channels, height, width)
        device: 'cuda' or 'cpu'
    
    Returns:
        flops: total FLOPs
        params: total parameters
    """
    try:
        from thop import profile
        
        model = model.to(device)
        model.eval()
        
        input_tensor = torch.randn(input_shape).to(device)
        
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        return flops, params
        
    except ImportError:
        print("thop not installed. Install with: pip install thop")
        # Fallback: count parameters only
        params = sum(p.numel() for p in model.parameters())
        return 0, params


def print_model_summary(model, input_shape, device='cuda'):
    """Print model summary with FLOPs."""
    flops, params = count_flops(model, input_shape, device)
    
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {params:,}")
    print(f"FLOPs: {flops:,}")
    if flops > 0:
        print(f"FLOPs (G): {flops / 1e9:.2f} G")
    print("=" * 60)