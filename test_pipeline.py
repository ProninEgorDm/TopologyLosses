#!/usr/bin/env python3
"""
Test script for the enhanced training pipeline.
"""

import os
import sys
import torch
import yaml

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Train.pipeline import SegmentationPipeline

def test_pipeline():
    """Test the pipeline with a small number of epochs."""
    
    # Load config
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Reduce epochs for testing
    config['training']['epochs'] = 2
    config['training']['device'] = 'cpu'  # Use CPU for testing
    
    # Save test config
    test_config_path = 'configs/test_config.yaml'
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print("Testing SegmentationPipeline...")
    
    try:
        # Create pipeline
        pipeline = SegmentationPipeline(test_config_path)
        print("✓ Pipeline initialized successfully")
        
        # Check FLOPs
        print(f"Model FLOPs: {pipeline.flops}")
        print(f"Model parameters: {pipeline.params}")
        
        # Run training
        pipeline.train()
        print("✓ Training completed successfully")
        
        # Test evaluation
        test_results = pipeline.evaluate(verbose=True)
        print("✓ Evaluation completed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_pipeline()
    sys.exit(0 if success else 1)