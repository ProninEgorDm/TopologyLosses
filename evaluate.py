"""
Comprehensive evaluation script for VessMAP semantic segmentation models.
Provides detailed metrics, visualizations, and comparisons.
"""

import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from Train.pipeline import SegmentationPipeline
from metrics.segmentation_metrics import SegmentationMetrics


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to configuration file (extracted from checkpoint if not provided)
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if config_path is None:
            self.config = self.checkpoint['config']
        else:
            import yaml
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Load pipeline
        self.pipeline = SegmentationPipeline.__new__(SegmentationPipeline)
        self.pipeline.config = self.config
        self.pipeline.device = torch.device(self.config['training'].get('device', 'cpu'))
        self.pipeline.epochs = self.config['training']['epochs']
        self.pipeline._setup_directories()
        
        self.pipeline.model = self.pipeline._build_model()
        self.pipeline.train_loader, self.pipeline.val_loader, self.pipeline.test_loader = self.pipeline._build_dataloaders()
        self.pipeline.seg_loss = self.pipeline._build_loss()
        self.pipeline.top_loss = self.pipeline._build_topological_loss()
        self.pipeline.optimizer = self.pipeline._build_optimizer()
        self.pipeline.scheduler = self.pipeline._build_scheduler()
        self.pipeline.metrics = SegmentationMetrics()
        
        self.pipeline.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.pipeline.model.eval()
        
        self.metrics = SegmentationMetrics()
    
    def evaluate_full(self, save_report=True):
        """
        Perform full evaluation on all splits.
        
        Args:
            save_report: Whether to save a detailed report
        
        Returns:
            Dictionary with complete evaluation results
        """
        print("=" * 80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 80)
        
        results = {}
        
        # Evaluate on all splits
        for split_name, loader in [
            ('train', self.pipeline.train_loader),
            ('val', self.pipeline.val_loader),
            ('test', self.pipeline.test_loader)
        ]:
            print(f"\nEvaluating on {split_name.upper()} set...")
            split_results = self._evaluate_split(loader)
            results[split_name] = split_results
            
            # Print results
            print(f"\n{split_name.upper()} Results:")
            print(f"  Loss: {split_results['loss']:.4f}")
            print(f"  Metrics:")
            for metric_name, metric_value in split_results['metrics'].items():
                print(f"    {metric_name}: {metric_value:.4f}")
        
        if save_report:
            self._save_evaluation_report(results)
        
        return results
    
    def _evaluate_split(self, loader):
        """Evaluate on a single split."""
        epoch_loss = 0.0
        preds_all = []
        targets_all = []
        
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.pipeline.device)
                labels = batch['label'].to(self.pipeline.device)
                
                outputs = self.pipeline.model(images)
                
                seg_loss = self.pipeline.seg_loss(outputs, labels)
                loss = seg_loss
                
                if self.pipeline.top_loss is not None:
                    top_loss = self.pipeline.top_loss(outputs, labels)
                    loss += top_loss
                
                epoch_loss += loss.item()
                preds_all.append(torch.sigmoid(outputs).detach().cpu())
                targets_all.append(labels.detach().cpu())
        
        # Compute metrics
        preds_all = torch.cat(preds_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        metrics = self.pipeline._compute_metrics(preds_all, targets_all)
        
        return {
            'loss': epoch_loss / len(loader),
            'metrics': metrics,
            'predictions': preds_all,
            'targets': targets_all
        }
    
    def generate_prediction_samples(self, num_samples=5, split='val', save_dir='predictions'):
        """
        Generate sample predictions with visualizations.
        
        Args:
            num_samples: Number of samples to visualize
            split: Which split to sample from ('train', 'val', 'test')
            save_dir: Directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if split == 'train':
            loader = self.pipeline.train_loader
        elif split == 'val':
            loader = self.pipeline.val_loader
        else:
            loader = self.pipeline.test_loader
        
        self.pipeline.model.eval()
        sample_count = 0
        
        with torch.no_grad():
            for batch in loader:
                if sample_count >= num_samples:
                    break
                
                images = batch['image'].to(self.pipeline.device)
                labels = batch['label'].to(self.pipeline.device)
                
                outputs = self.pipeline.model(images)
                preds = torch.sigmoid(outputs).detach().cpu()
                
                for i in range(images.size(0)):
                    if sample_count >= num_samples:
                        break
                    
                    image = images[i].cpu().squeeze().numpy()
                    target = labels[i].cpu().squeeze().numpy()
                    pred = preds[i].squeeze().numpy()
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axes[0].imshow(image, cmap='gray')
                    axes[0].set_title('Input Image')
                    axes[0].axis('off')
                    
                    axes[1].imshow(target, cmap='gray')
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    
                    im = axes[2].imshow(pred, cmap='gray')
                    axes[2].set_title('Prediction')
                    axes[2].axis('off')
                    plt.colorbar(im, ax=axes[2])
                    
                    plt.tight_layout()
                    save_path = os.path.join(save_dir, f'{split}_sample_{sample_count:03d}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    sample_count += 1
    
    def compute_per_image_metrics(self, split='val'):
        """
        Compute metrics for each image separately.
        
        Args:
            split: Which split to evaluate ('train', 'val', 'test')
        
        Returns:
            DataFrame with per-image metrics
        """
        if split == 'train':
            loader = self.pipeline.train_loader
        elif split == 'val':
            loader = self.pipeline.val_loader
        else:
            loader = self.pipeline.test_loader
        
        per_image_metrics = []
        
        self.pipeline.model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.pipeline.device)
                labels = batch['label'].to(self.pipeline.device)
                image_ids = batch['image_id']
                
                outputs = self.pipeline.model(images)
                preds = torch.sigmoid(outputs).detach().cpu()
                labels = labels.cpu()
                
                for i in range(images.size(0)):
                    pred = preds[i].squeeze().numpy()
                    target = labels[i].squeeze().numpy()
                    
                    # Compute per-image metrics
                    metrics = SegmentationMetrics()
                    metrics.update(
                        torch.from_numpy(pred).unsqueeze(0),
                        torch.from_numpy(target).unsqueeze(0)
                    )
                    image_metrics = metrics.compute()
                    image_metrics['image_id'] = image_ids[i]
                    
                    per_image_metrics.append(image_metrics)
        
        return per_image_metrics
    
    def _save_evaluation_report(self, results):
        """Save detailed evaluation report."""
        report_path = os.path.join(
            os.path.dirname(self.checkpoint_path),
            'evaluation_report.json'
        )
        
        # Prepare results for JSON serialization
        report_data = {}
        for split_name, split_results in results.items():
            report_data[split_name] = {
                'loss': float(split_results['loss']),
                'metrics': {
                    k: float(v) for k, v in split_results['metrics'].items()
                }
            }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nEvaluation report saved to {report_path}")


class ComparisonAnalyzer:
    """Analyze and visualize loss function comparisons."""
    
    def __init__(self, comparison_results_path: str):
        """
        Initialize analyzer.
        
        Args:
            comparison_results_path: Path to loss comparison results JSON
        """
        with open(comparison_results_path, 'r') as f:
            self.results = json.load(f)
    
    def plot_comparison(self, save_path='comparison_plot.png'):
        """Create comparison plots."""
        loss_functions = list(self.results.keys())
        
        # Extract metrics
        test_losses = [self.results[lf]['test_loss'] for lf in loss_functions]
        test_dices = [self.results[lf]['test_metrics']['f1'] for lf in loss_functions]
        test_ious = [self.results[lf]['test_metrics']['iou'] for lf in loss_functions]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss comparison
        axes[0].bar(loss_functions, test_losses, color='steelblue')
        axes[0].set_ylabel('Test Loss')
        axes[0].set_title('Loss Function Comparison - Loss')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Dice comparison
        axes[1].bar(loss_functions, test_dices, color='coral')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Loss Function Comparison - Dice')
        axes[1].set_ylim([0, 1])
        axes[1].tick_params(axis='x', rotation=45)
        
        # IoU comparison
        axes[2].bar(loss_functions, test_ious, color='lightgreen')
        axes[2].set_ylabel('IoU Score')
        axes[2].set_title('Loss Function Comparison - IoU')
        axes[2].set_ylim([0, 1])
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        plt.close()
    
    def print_summary(self):
        """Print comparison summary."""
        print("\n" + "=" * 80)
        print("LOSS FUNCTION COMPARISON SUMMARY")
        print("=" * 80)
        
        print(f"\n{'Loss Function':<20} {'Test Loss':<15} {'Test Dice':<15} {'Test IoU':<15}")
        print("-" * 65)
        
        for loss_name, data in self.results.items():
            test_loss = data['test_loss']
            test_dice = data['test_metrics']['f1']
            test_iou = data['test_metrics']['iou']
            
            print(f"{loss_name:<20} {test_loss:<15.4f} {test_dice:<15.4f} {test_iou:<15.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VessMAP segmentation model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (optional)')
    parser.add_argument('--generate-samples', type=int, default=0,
                        help='Number of prediction samples to generate')
    parser.add_argument('--sample-split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Which split to sample from')
    parser.add_argument('--per-image-metrics', action='store_true',
                        help='Compute per-image metrics')
    parser.add_argument('--compare-losses', type=str, default=None,
                        help='Path to loss comparison results JSON')
    
    args = parser.parse_args()
    
    if args.compare_losses:
        # Analyze loss comparison
        analyzer = ComparisonAnalyzer(args.compare_losses)
        analyzer.print_summary()
        analyzer.plot_comparison(save_path='loss_comparison.png')
    else:
        # Evaluate model
        evaluator = ModelEvaluator(args.checkpoint, args.config)
        
        # Full evaluation
        results = evaluator.evaluate_full()
        
        # Generate samples if requested
        if args.generate_samples > 0:
            evaluator.generate_prediction_samples(
                num_samples=args.generate_samples,
                split=args.sample_split
            )
        
        # Per-image metrics if requested
        if args.per_image_metrics:
            per_image = evaluator.compute_per_image_metrics(split=args.sample_split)
            print(f"\nComputed metrics for {len(per_image)} images")


if __name__ == '__main__':
    main()
