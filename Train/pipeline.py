"""
Comprehensive training and validation pipeline for VessMAP semantic segmentation.
Supports multiple loss functions and metrics collection.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from data.dataset import VessMapDataset, create_data_splits
from models import TransUNet
# from losses.segmentation_losses import get_segmentation_loss
# from losses.topological_losses import get_topological_loss
from losses.segmentation_losses.losses import get_segmentation_loss
from losses.topological_losses.losses import get_topological_loss
from metrics.segmentation_metrics import SegmentationMetrics
from metrics.topological_metrcis import TopologicalMetrics
from utils.flops_counter import count_flops, print_model_summary
import time


class WeightedLoss(nn.Module):
    """Wrap a loss module with an explicit weight."""

    def __init__(self, loss_module: nn.Module, weight: float = 1.0):
        super().__init__()
        self.loss_module = loss_module
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return self.weight * self.loss_module(pred, target)


class LossComposer(nn.Module):
    """Combine multiple weighted loss modules into a single loss."""

    def __init__(self, loss_modules: dict[str, nn.Module]):
        super().__init__()
        self.losses = nn.ModuleDict(loss_modules)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        total_loss = 0.0
        for loss_module in self.losses.values():
            total_loss = total_loss + loss_module(pred, target)
        return total_loss


class SegmentationPipeline:
    """Complete training and validation pipeline for semantic segmentation."""
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['training'].get('device', 'cpu'))
        self.epochs = self.config['training']['epochs']
        
        # Setup directories
        self._setup_directories()
        
        # Initialize components
        self.model = self._build_model()
        self._compute_model_flops()
        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()
        self.loss_fn = self._build_loss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.metrics = SegmentationMetrics(threshold=self.config['training'].get('metric_threshold', 0.5))
        self.topological_metrics = TopologicalMetrics(threshold=self.config['training'].get('metric_threshold', 0.5))
        
        # Setup logging
        self.writer = SummaryWriter(self.config['training']['log_dir'])
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        
        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': defaultdict(list),
            'val_metrics': defaultdict(list),
            'epoch_times': [],  # Training time per epoch
            'flops': self.flops,
            'params': self.params,
        }
        
        # Loss comparison tracking
        self.loss_comparison = defaultdict(lambda: {
            'train_losses': [],
            'val_losses': [],
            'val_metrics': defaultdict(list)
        })
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self):
        """Create necessary directories."""
        dirs_to_create = [
            self.config['training']['checkpoint_dir'],
            self.config['training']['log_dir'],
            os.path.join(self.config['training']['log_dir'], 'comparisons'),
        ]
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
    
    def _build_model(self) -> nn.Module:
        """Build the model."""
        model_name = self.config['model']['name'].lower()
        
        if model_name == 'transunet':
            model = TransUNet(
                in_channels=self.config['model']['in_channels'],
                out_channels=self.config['model']['out_channels'],
                img_dim=self.config['data']['image_size'],
                block_num=self.config['model']['block_num'],
                mlp_dim=self.config['model']['mlp_dim'],
                head_num=self.config['model']['head_num'],
                patch_dim=self.config['model']['patch_dim'],
                class_num=1
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(self.device)
        return model
    
    def _compute_model_flops(self):
        """Compute and store model FLOPs and parameters."""
        input_shape = (1, self.config['model']['in_channels'], 
                      self.config['data']['image_size'], self.config['data']['image_size'])
        self.flops, self.params = count_flops(self.model, input_shape, self.device.type)
        print_model_summary(self.model, input_shape, self.device.type)
    
    def _build_dataloaders(self):
        """Build train, validation, and test dataloaders."""
        data_dir = self.config['data']['dataset_path']
        
        # Get all image IDs
        image_dir = os.path.join(data_dir, 'images')
        all_items = [f.replace('.tiff', '') for f in os.listdir(image_dir) if f.endswith('.tiff')]
        
        # Create splits
        train_items, val_items, test_items = create_data_splits(
            all_items,
            test_size=self.config['data']['test_size'],
            val_size_from_test=self.config['data']['val_size_from_test'],
        )
        
        # Create datasets
        train_dataset = VessMapDataset(train_items, data_dir=data_dir, train=True)
        val_dataset = VessMapDataset(val_items, data_dir=data_dir, train=False)
        test_dataset = VessMapDataset(test_items, data_dir=data_dir, train=False)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")
        
        return train_loader, val_loader, test_loader
    
    def _build_topological_loss(self):
        """Build topological loss (if specified)."""
        loss_name = self.config['loss'].get('topological_loss')
        weight = self.config['loss'].get('topological_weight', 0.1)
        return get_topological_loss(loss_name, weight)

    def _normalize_loss_name(self, loss_name: str, loss_type: str) -> str:
        """Normalize loss name aliases for segmentation or topological modules."""
        if not isinstance(loss_name, str):
            raise ValueError(f"Loss name must be a string, got: {type(loss_name)}")

        key = loss_name.strip().lower()
        if loss_type == 'segmentation':
            aliases = {
                'ce': 'bce',
                'cross_entropy': 'bce',
                'bce': 'bce',
                'dice': 'dice',
                'focal': 'focal',
                'combined': 'combined',
            }
            return aliases.get(key, key)
        elif loss_type == 'topological':
            aliases = {
                'betti': 'betti',
                'betti_matching': 'betti_matching',
                'voi': 'voi',
                'hausdorff': 'hausdorff',
                'composite': 'composite',
                'cldice': 'cldice',
                'gdice': 'gdice',
                'sat': 'sat',
                'tloss': 'tloss',
                'bm': 'bm'
            }
            if key in aliases:
                return aliases[key]
            raise ValueError(
                f"Unknown topological loss '{loss_name}'. Supported: {list(aliases.values())}"
            )

    def _build_loss(self) -> nn.Module:
        """Build the final training loss function."""
        loss_config = self.config['loss']

        if isinstance(loss_config, dict) and 'segmentation' in loss_config:
            return self._build_composite_loss(loss_config)

        segment_loss_name = loss_config.get('segmentation_loss', 'bce')
        segmentation = get_segmentation_loss(segment_loss_name)
        topological = get_topological_loss(
            loss_config.get('topological_loss'),
            loss_config.get('topological_weight', 0.1)
        )

        if topological is not None:
            return LossComposer({'segmentation': segmentation, 'topological': topological})

        return segmentation

    def _build_composite_loss(self, loss_config: dict) -> nn.Module:
        """Build a composite loss from structured loss configuration."""
        loss_modules = {}

        if 'segmentation' in loss_config:
            for entry in loss_config['segmentation']:
                if not isinstance(entry, dict) or len(entry) != 1:
                    raise ValueError(
                        'Each segmentation loss entry must be a single-key dict.'
                    )
                name, params = next(iter(entry.items()))
                params = params or {}
                weight = float(params.pop('weight', 1.0))
                loss_name = self._normalize_loss_name(name, 'segmentation')
                module = get_segmentation_loss(loss_name, **params)
                weighted = WeightedLoss(module, weight)
                module_key = f'segmentation_{loss_name}'
                loss_modules[module_key] = weighted

        if 'topological' in loss_config:
            for entry in loss_config['topological']:
                if not isinstance(entry, dict) or len(entry) != 1:
                    raise ValueError(
                        'Each topological loss entry must be a single-key dict.'
                    )
                name, params = next(iter(entry.items()))
                params = params or {}
                weight = float(params.pop('weight', 1.0))
                loss_name = self._normalize_loss_name(name, 'topological')
                module = get_topological_loss(loss_name, weight, **params)
                module_key = f'topological_{loss_name}'
                loss_modules[module_key] = module

        if not loss_modules:
            raise ValueError('No valid losses were found in composite loss configuration.')

        return LossComposer(loss_modules)

    def _loss_description(self) -> str:
        """Create a readable loss description for logging."""
        loss_config = self.config['loss']
        if isinstance(loss_config, dict) and 'segmentation' in loss_config:
            segment_names = [next(iter(entry.keys())) for entry in loss_config.get('segmentation', [])]
            topo_names = [next(iter(entry.keys())) for entry in loss_config.get('topological', [])]
            return f"segmentation={segment_names}, topological={topo_names}"

        return (
            f"segmentation={loss_config.get('segmentation_loss', 'unknown')}, "
            f"topological={loss_config.get('topological_loss', 'none')}"
        )

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer."""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )
    
    def _compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute segmentation and topological metrics."""
        self.metrics.update(predictions, targets)
        self.topological_metrics.update(predictions, targets)

        seg_metrics = self.metrics.compute()
        topo_metrics = self.topological_metrics.compute()

        self.metrics.reset()
        self.topological_metrics.reset()

        flattened_topo = {
            'betti_pred_b0': topo_metrics['betti_pred']['b0'],
            'betti_pred_b1': topo_metrics['betti_pred']['b1'],
            'betti_target_b0': topo_metrics['betti_target']['b0'],
            'betti_target_b1': topo_metrics['betti_target']['b1'],
            'connectivity_match': topo_metrics['connectivity_match'],
            'skeleton_accuracy': topo_metrics['skeleton_accuracy'],
        }

        metrics = {**seg_metrics, **flattened_topo}
        return metrics

    def _train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        preds_all = []
        targets_all = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device).squeeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze(1)
            
            loss = self.loss_fn(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            preds_all.append(torch.sigmoid(outputs).detach().cpu())
            targets_all.append(labels.cpu())
            
            pbar.set_postfix({'loss': f'{epoch_loss / (batch_idx + 1):.4f}'})
        
        # Compute epoch metrics
        preds_all = torch.cat(preds_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        metrics = self._compute_metrics(preds_all, targets_all)
        
        return {
            'loss': epoch_loss / len(self.train_loader),
            'metrics': metrics
        }
    
    def _val_epoch(self, epoch: int) -> dict:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        epoch_loss = 0.0
        preds_all = []
        targets_all = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Val]')
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device).squeeze(1)

                # Forward pass
                outputs = self.model(images).squeeze(1)

                # Compute loss
                loss = self.loss_fn(outputs, labels)
                
                epoch_loss += loss.item()
                preds_all.append(torch.sigmoid(outputs))
                targets_all.append(labels)

        preds_all = torch.cat(preds_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        metrics = self._compute_metrics(preds_all, targets_all)
        
        return {
            'loss': epoch_loss / len(self.val_loader),
            'metrics': metrics
        }
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': dict(self.history),
            'flops': self.flops,
            'params': self.params,
        }
        
        if is_best:
            loss_desc = self._loss_description().replace(' ', '_').replace(',', '_').replace('=', '_')
            best_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                f'best_model_{loss_desc}.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def train(self):
        """Run full training pipeline."""
        print("=" * 80)
        print(f"Starting training - Loss: {self._loss_description()}")
        print("=" * 80)
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Train
            train_result = self._train_epoch(epoch)
            
            # Validate
            val_result = self._val_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_result['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_result['loss'], epoch)
            self.writer.add_scalar('Time/epoch', epoch_time, epoch)
            
            for metric_name, metric_value in train_result['metrics'].items():
                self.writer.add_scalar(f'Train/{metric_name}', metric_value, epoch)
            
            for metric_name, metric_value in val_result['metrics'].items():
                self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            # Update history
            self.history['train_loss'].append(train_result['loss'])
            self.history['val_loss'].append(val_result['loss'])
            self.history['epoch_times'].append(epoch_time)
            
            for metric_name, metric_value in train_result['metrics'].items():
                self.history['train_metrics'][metric_name].append(metric_value)
            
            for metric_name, metric_value in val_result['metrics'].items():
                self.history['val_metrics'][metric_name].append(metric_value)
            
            # Save checkpoints
            if self.config['training']['save_checkpoints']:
                self._save_checkpoint(epoch)
                
                # Save best model
                if val_result['loss'] < self.best_val_loss:
                    self.best_val_loss = val_result['loss']
                    self._save_checkpoint(epoch, is_best=True)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"Train Loss: {train_result['loss']:.4f} | Val Loss: {val_result['loss']:.4f}")
            print(f"Train Dice: {train_result['metrics']['f1']:.4f} | Val Dice: {val_result['metrics']['f1']:.4f}")
            print(f"Train IoU: {train_result['metrics']['iou']:.4f} | Val IoU: {val_result['metrics']['iou']:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s")
        
        self.writer.close()
        self._save_history()
        print("\nTraining completed!")
    
    def evaluate(self, loader=None, verbose=True):
        """
        Evaluate model on given dataloader.
        
        Args:
            loader: DataLoader to evaluate on. If None, uses val_loader.
            verbose: Whether to print results.
        
        Returns:
            Dictionary with metrics
        """
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        epoch_loss = 0.0
        preds_all = []
        targets_all = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc='Evaluating')
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)

                labels = labels.squeeze(1)
                loss = self.loss_fn(outputs, labels)
                epoch_loss += loss.item()
                preds_all.append(torch.sigmoid(outputs))
                targets_all.append(labels)
        
        # Compute metrics
        preds_all = torch.cat(preds_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        metrics = self._compute_metrics(preds_all, targets_all)
        
        results = {
            'loss': epoch_loss / len(loader),
            'metrics': metrics
        }
        
        if verbose:
            print("\nEvaluation Results:")
            print(f"Loss: {results['loss']:.4f}")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")
        
        return results
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(
            self.config['training']['log_dir'],
            'training_history.json'
        )
        
        # Convert defaultdict to regular dict for JSON serialization
        history_to_save = {
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'train_metrics': dict(self.history['train_metrics']),
            'val_metrics': dict(self.history['val_metrics']),
            'epoch_times': self.history['epoch_times'],
            'flops': self.flops,
            'params': self.params,
            'total_training_time': sum(self.history['epoch_times']),
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")


class MultiLossComparator:
    """Compare performance of different loss functions."""
    
    def __init__(self, config_path: str, loss_configs: dict):
        """
        Initialize comparator.
        
        Args:
            config_path: Path to configuration file
            loss_configs: Dict of loss config names to loss configurations
        """
        self.config_path = config_path
        self.loss_configs = loss_configs
        self.results = {}
    
    def run_comparison(self):
        """Run training with all loss configurations and collect results."""
        print("=" * 80)
        print("LOSS FUNCTION COMPARISON")
        print("=" * 80)
        
        for loss_name, loss_config in self.loss_configs.items():
            print(f"\n{'='*80}")
            print(f"Training with {loss_name.upper()} configuration")
            print(f"{'='*80}\n")
            
            # Load base config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update loss config
            config['loss'] = loss_config
            
            # Create pipeline
            pipeline = SegmentationPipeline.__new__(SegmentationPipeline)
            pipeline.config = config
            pipeline.device = torch.device(config['training'].get('device', 'cpu'))
            pipeline.epochs = config['training']['epochs']
            pipeline._setup_directories()
            
            pipeline.model = pipeline._build_model()
            pipeline._compute_model_flops()
            pipeline.train_loader, pipeline.val_loader, pipeline.test_loader = pipeline._build_dataloaders()
            pipeline.loss_fn = pipeline._build_loss()
            pipeline.optimizer = pipeline._build_optimizer()
            pipeline.scheduler = pipeline._build_scheduler()
            pipeline.metrics = SegmentationMetrics(threshold=config['training'].get('metric_threshold', 0.5))
            pipeline.topological_metrics = TopologicalMetrics(threshold=config['training'].get('metric_threshold', 0.5))
            
            pipeline.writer = SummaryWriter(
                os.path.join(pipeline.config['training']['log_dir'], f'config_{loss_name}')
            )
            pipeline.best_val_loss = float('inf')
            pipeline.best_val_dice = 0.0
            
            pipeline.history = {
                'train_loss': [],
                'val_loss': [],
                'train_metrics': defaultdict(list),
                'val_metrics': defaultdict(list),
                'epoch_times': [],
                'flops': pipeline.flops,
                'params': pipeline.params,
            }
            
            # Train
            pipeline.train()
            
            # Evaluate on test set
            test_results = pipeline.evaluate(pipeline.test_loader)
            
            self.results[loss_name] = {
                'train_history': pipeline.history,
                'test_results': test_results,
                'best_val_loss': pipeline.best_val_loss,
                'flops': pipeline.flops,
                'params': pipeline.params,
                'total_training_time': sum(pipeline.history['epoch_times']),
            }
            
            pipeline.writer.close()
        
        self._save_comparison_results()
    
    def _save_comparison_results(self):
        """Save comparison results to JSON."""
        results_path = os.path.join(
            os.path.dirname(self.config_path),
            '..',
            'logs',
            'loss_comparison.json'
        )
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        # Prepare results for JSON serialization
        results_to_save = {}
        for loss_name, data in self.results.items():
            results_to_save[loss_name] = {
                'test_metrics': data['test_results']['metrics'],
                'test_loss': data['test_results']['loss'],
                'best_val_loss': data['best_val_loss'],
                'flops': data['flops'],
                'params': data['params'],
                'total_training_time': data['total_training_time'],
            }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nComparison results saved to {results_path}")
        self._print_comparison_summary()
    
    def _print_comparison_summary(self):
        """Print comparison summary."""
        print("\n" + "=" * 100)
        print("COMPARISON SUMMARY")
        print("=" * 100)
        
        print(f"\n{'Loss Function':<20} {'Test Loss':<12} {'Test Dice':<12} {'Test IoU':<12} {'Time (s)':<12} {'FLOPs (G)':<12}")
        print("-" * 80)
        
        for loss_name, data in self.results.items():
            test_loss = data['test_results']['loss']
            test_dice = data['test_results']['metrics']['f1']
            test_iou = data['test_results']['metrics']['iou']
            total_time = data['total_training_time']
            flops_g = data['flops'] / 1e9 if data['flops'] > 0 else 0
            
            print(f"{loss_name:<20} {test_loss:<12.4f} {test_dice:<12.4f} {test_iou:<12.4f} {total_time:<12.2f} {flops_g:<12.2f}")
