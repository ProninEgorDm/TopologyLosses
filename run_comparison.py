#!/usr/bin/env python3
"""
Script to run loss function comparison for VessMAP segmentation.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Train.pipeline import MultiLossComparator

def main():
    parser = argparse.ArgumentParser(description='Run loss function comparison for VessMAP segmentation')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to base configuration file')
    parser.add_argument('--loss-configs', type=str, default=None,
                       help='Path to YAML file with loss configurations to compare')
    parser.add_argument('--output-dir', type=str, default='logs/comparisons',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Define loss configurations to compare
    if args.loss_configs and os.path.exists(args.loss_configs):
        with open(args.loss_configs, 'r') as f:
            loss_configs = yaml.safe_load(f)
    else:
        # Default loss configurations
        loss_configs = {
            'dice': {
                'segmentation_loss': 'dice'
            },
            'bce': {
                'segmentation_loss': 'bce'
            },
            'focal': {
                'segmentation_loss': 'focal'
            },
            'dice_bce': {
                'segmentation': [
                    {'dice': {'weight': 0.5}},
                    {'bce': {'weight': 0.5}}
                ]
            },
            'dice_topo_betti': {
                'segmentation': [
                    {'dice': {'weight': 0.95}}
                ],
                'topological': [
                    {'betti': {'weight': 0.05}}
                ]
            },
            'combined_topo': {
                'segmentation': [
                    {'dice': {'weight': 0.5}},
                    {'bce': {'weight': 0.5}}
                ],
                'topological': [
                    {'betti': {'weight': 0.1}},
                    {'voi': {'tloss': 0.1}}
                ]
            },
            'SAT_combined': {
                'segmentation': [
                    {'dice': {'weight': 0.5}},
                    {'bce': {'weight': 0.5}}
                ],
                'topological': [
                    {'sat': {'weight': 0.05}}
                ]
            },
        }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Update config with output directory
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    base_config['training']['log_dir'] = args.output_dir
    
    # Save updated config
    config_path = os.path.join(args.output_dir, 'comparison_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(base_config, f)
    
    # Run comparison
    comparator = MultiLossComparator(config_path, loss_configs)
    comparator.run_comparison()
    
    print(f"\nComparison completed! Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()