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
        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()
        self.seg_loss = self._build_loss()
        self.top_loss = self._build_topological_loss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.metrics = SegmentationMetrics()
        
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
    
    def _build_loss(self) -> nn.Module:
        """Build segmentation loss."""
        loss_name = self.config['loss']['segmentation_loss']
        return get_segmentation_loss(loss_name)
    
    def _build_topological_loss(self):
        """Build topological loss (if specified)."""
        loss_name = self.config['loss'].get('topological_loss')
        weight = self.config['loss'].get('topological_weight', 0.1)
        return get_topological_loss(loss_name, weight)
    
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
        """Compute segmentation metrics."""
        self.metrics.update(predictions, targets)
        metrics = self.metrics.compute()
        self.metrics.reset()
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
            
            # Compute loss
            seg_loss = self.seg_loss(outputs, labels)
            loss = seg_loss
            
            # Add topological loss if available
            if self.top_loss is not None:
                top_loss = self.top_loss(outputs, labels)
                loss += top_loss
            
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
                seg_loss = self.seg_loss(outputs, labels)
                loss = seg_loss
                
                if self.top_loss is not None:
                    top_loss = self.top_loss(outputs, labels)
                    loss += top_loss
                
                epoch_loss += loss.item()
                preds_all.append(torch.sigmoid(outputs))
                targets_all.append(labels)
                
                pbar.set_postfix({'loss': f'{epoch_loss / (batch_idx + 1):.4f}'})
        
        # Compute epoch metrics
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
        }
        
        if is_best:
            best_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def train(self):
        """Run full training pipeline."""
        print("=" * 80)
        print(f"Starting training - Loss: {self.config['loss']['segmentation_loss']}")
        print("=" * 80)
        
        for epoch in range(self.epochs):
            # Train
            train_result = self._train_epoch(epoch)
            
            # Validate
            val_result = self._val_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_result['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_result['loss'], epoch)
            
            for metric_name, metric_value in train_result['metrics'].items():
                self.writer.add_scalar(f'Train/{metric_name}', metric_value, epoch)
            
            for metric_name, metric_value in val_result['metrics'].items():
                self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            # Update history
            self.history['train_loss'].append(train_result['loss'])
            self.history['val_loss'].append(val_result['loss'])
            
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
                
                seg_loss = self.seg_loss(outputs, labels)
                loss = seg_loss
                
                if self.top_loss is not None:
                    top_loss = self.top_loss(outputs, labels)
                    loss += top_loss
                
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
    
    def __init__(self, config_path: str, loss_functions: list):
        """
        Initialize comparator.
        
        Args:
            config_path: Path to configuration file
            loss_functions: List of loss function names to compare
        """
        self.config_path = config_path
        self.loss_functions = loss_functions
        self.results = {}
    
    def run_comparison(self):
        """Run training with all loss functions and collect results."""
        print("=" * 80)
        print("LOSS FUNCTION COMPARISON")
        print("=" * 80)
        
        for loss_name in self.loss_functions:
            print(f"\n{'='*80}")
            print(f"Training with {loss_name.upper()} loss")
            print(f"{'='*80}\n")
            
            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update loss
            config['loss']['segmentation_loss'] = loss_name
            
            # Create pipeline
            pipeline = SegmentationPipeline.__new__(SegmentationPipeline)
            pipeline.config = config
            pipeline.device = torch.device(config['training'].get('device', 'cpu'))
            pipeline.epochs = config['training']['epochs']
            pipeline._setup_directories()
            
            pipeline.model = pipeline._build_model()
            pipeline.train_loader, pipeline.val_loader, pipeline.test_loader = pipeline._build_dataloaders()
            pipeline.seg_loss = pipeline._build_loss()
            pipeline.top_loss = pipeline._build_topological_loss()
            pipeline.optimizer = pipeline._build_optimizer()
            pipeline.scheduler = pipeline._build_scheduler()
            pipeline.metrics = SegmentationMetrics()
            
            pipeline.writer = SummaryWriter(
                os.path.join(pipeline.config['training']['log_dir'], f'loss_{loss_name}')
            )
            pipeline.best_val_loss = float('inf')
            pipeline.best_val_dice = 0.0
            
            pipeline.history = {
                'train_loss': [],
                'val_loss': [],
                'train_metrics': defaultdict(list),
                'val_metrics': defaultdict(list),
            }
            
            # Train
            pipeline.train()
            
            # Evaluate on test set
            test_results = pipeline.evaluate(pipeline.test_loader)
            
            self.results[loss_name] = {
                'train_history': pipeline.history,
                'test_results': test_results,
                'best_val_loss': pipeline.best_val_loss,
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
            }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nComparison results saved to {results_path}")
        self._print_comparison_summary()
    
    def _print_comparison_summary(self):
        """Print comparison summary."""
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        
        print(f"\n{'Loss Function':<20} {'Test Loss':<15} {'Test Dice':<15} {'Test IoU':<15}")
        print("-" * 65)
        
        for loss_name, data in self.results.items():
            test_loss = data['test_results']['loss']
            test_dice = data['test_results']['metrics']['f1']
            test_iou = data['test_results']['metrics']['iou']
            
            print(f"{loss_name:<20} {test_loss:<15.4f} {test_dice:<15.4f} {test_iou:<15.4f}")
