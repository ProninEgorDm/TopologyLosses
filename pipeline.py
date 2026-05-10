# pipeline.py
import argparse
import yaml
import json
import os
import torch
from train import Trainer


class VessMapPipeline:
    
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def run(self):
        """Run the complete pipeline."""
        print("=" * 70)
        print("VessMAP Segmentation Pipeline")
        print("=" * 70)
        
        # Create trainer and train
        trainer = Trainer(self.config_path)
        results = trainer.train()
        
        print("\n" + "=" * 70)
        print("Pipeline completed successfully!")
        print("=" * 70)
        
        return results
    
    def evaluate_only(self, checkpoint_path):
        """Run evaluation only with a trained model."""
        # Create trainer
        trainer = Trainer(self.config_path)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load test data
        _, _, test_loader = trainer._load_data()
        
        # Evaluate
        test_loss, test_seg, test_top = trainer.validate(test_loader, 0)
        
        # Print results
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Seg Metrics: {test_seg}")
        print(f"Test Top Metrics: {test_top}")
        
        return {
            'test_loss': test_loss,
            'test_seg_metrics': test_seg,
            'test_top_metrics': test_top
        }


def main():
    parser = argparse.ArgumentParser(description='VessMAP Segmentation Pipeline')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train',
                        help='Pipeline mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for evaluation mode')
    
    args = parser.parse_args()
    
    pipeline = VessMapPipeline(args.config)
    
    if args.mode == 'train':
        pipeline.run()
    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path required for evaluation mode")
        pipeline.evaluate_only(args.checkpoint)


if __name__ == '__main__':
    main()