"""
Training script for VessMAP semantic segmentation with multiple loss functions.
Usage:
    Single training:
        python train.py --config configs/default_config.yaml
    
    Compare losses:
        python run_comparison.py --config configs/default_config.yaml
"""

import argparse
import sys
from Train.pipeline import SegmentationPipeline


def main():
    parser = argparse.ArgumentParser(description='Train VessMAP segmentation model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Single training
    pipeline = SegmentationPipeline(args.config)
    
    if args.checkpoint:
        pipeline.load_checkpoint(args.checkpoint)
    
    pipeline.train()


if __name__ == '__main__':
    main()
        model_config = self.config['model']
        
        if model_config['name'] == 'transunet':
            return TransUNet(
                in_channels=model_config['model']['in_channels'],
                out_channels=model_config['model']['out_channels'],
                img_dim=self.config['data']['image_size'],
                block_num=model_config['model']['block_num'],
                mlp_dim=model_config['model']['mlp_dim'],
                head_num=model_config['model']['head_num'],
                patch_dim=model_config['model']['patch_dim'],
                class_num=1
            )
        else:
            raise ValueError(f"Unknown model: {model_config['name']}")
    
    def _get_transforms(self):
        """Get transforms from config."""
        from torchvision.transforms import v2
        
        transforms = []
        
        # Basic transforms
        transforms.extend([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        # Normalization
        if 'normalize' in self.config['transforms']:
            norm = self.config['transforms']['normalize']
            transforms.append(
                v2.Normalize(mean=norm['mean'], std=norm['std'])
            )
        
        return v2.Compose(transforms)
    
    def _get_augmentations(self):
        """Get augmentations from config."""
        if not self.config['transforms'].get('augmentations', False):
            return None
        
        from torchvision.transforms import v2
        
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(10),
        ])
    
    def _load_data(self):
        """Load and split data."""
        # Load measures to get item list
        measures_path = os.path.join(
            self.config['data']['dataset_path'], 
            'annotator1', 'measures.json'
        )
        measures = load_measures(measures_path)
        items = list(measures.keys())
        
        # Create splits
        train_items, val_items, test_items = create_data_splits(
            items,
            self.config['data']['test_size'],
            self.config['data']['val_size_from_test']
        )
        
        # Get transforms
        transforms = self._get_transforms()
        augmentations = self._get_augmentations()
        
        # Create datasets
        train_dataset = VessMapDataset(
            train_items, transforms, 
            augmentations if self.config['transforms']['augmentations'] else None
        )
        val_dataset = VessMapDataset(val_items, transforms)
        test_dataset = VessMapDataset(test_items, transforms)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Ensure output shape matches label
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = nn.functional.interpolate(
                    outputs, size=labels.shape[-2:], mode='bilinear'
                )
            
            # Compute loss
            seg_loss_val = self.seg_loss(outputs.squeeze(1), labels)
            loss = seg_loss_val
            
            if self.top_loss is not None:
                top_loss_val = self.top_loss(outputs.squeeze(1), labels)
                loss = loss + top_loss_val
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = epoch * len(loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)
        
        return total_loss / len(loader)
    
    def validate(self, loader, epoch):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        self.seg_metrics.reset()
        self.top_metrics.reset()
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Ensure output shape matches label
                if outputs.shape[-2:] != labels.shape[-2:]:
                    outputs = nn.functional.interpolate(
                        outputs, size=labels.shape[-2:], mode='bilinear'
                    )
                
                # Compute loss
                seg_loss_val = self.seg_loss(outputs.squeeze(1), labels)
                loss = seg_loss_val
                
                if self.top_loss is not None:
                    top_loss_val = self.top_loss(outputs.squeeze(1), labels)
                    loss = loss + top_loss_val
                
                total_loss += loss.item()
                
                # Update metrics
                probs = torch.sigmoid(outputs).squeeze(1)
                self.seg_metrics.update(probs.cpu(), labels.cpu())
                self.top_metrics.update((probs > 0.5).cpu(), labels.cpu())
        
        # Compute metrics
        seg_results = self.seg_metrics.compute()
        top_results = self.top_metrics.compute()
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', total_loss / len(loader), epoch)
        for k, v in seg_results.items():
            self.writer.add_scalar(f'val/{k}', v, epoch)
        for k, v in top_results.items():
            self.writer.add_scalar(f'val/{k}', v, epoch)
        
        return total_loss / len(loader), seg_results, top_results
    
    def save_checkpoint(self, epoch, metrics, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        path = os.path.join(self.config['training']['checkpoint_dir'], filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def train(self):
        """Main training loop."""
        print("=" * 60)
        print("Starting training...")
        print("=" * 60)
        
        # Load data
        train_loader, val_loader, test_loader = self._load_data()
        
        # Count FLOPs
        print_model_summary(
            self.model, 
            (1, self.config['model']['in_channels'], 
             self.config['data']['image_size'], 
             self.config['data']['image_size']),
            self.device
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, seg_metrics, top_metrics = self.validate(val_loader, epoch)
            
            # Print results
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Seg Metrics: {seg_metrics}")
            print(f"  Val Top Metrics: {top_metrics}")
            
            # Save checkpoint if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    epoch, 
                    {'loss': val_loss, **seg_metrics, **top_metrics},
                    'best_model.pth'
                )
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    epoch,
                    {'loss': val_loss, **seg_metrics, **top_metrics},
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
        
        # Final test evaluation
        print("\n" + "=" * 60)
        print("Final Test Evaluation")
        print("=" * 60)
        
        # Load best model
        best_checkpoint = torch.load(
            os.path.join(self.config['training']['checkpoint_dir'], 'best_model.pth')
        )
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Evaluate on test set
        test_loss, test_seg, test_top = self.validate(test_loader, self.epochs)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Seg Metrics: {test_seg}")
        print(f"Test Top Metrics: {test_top}")
        
        # Save final results
        results = {
            'test_loss': test_loss,
            'test_seg_metrics': test_seg,
            'test_top_metrics': test_top,
            'config': self.config
        }
        
        with open(os.path.join(self.config['training']['log_dir'], 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        self.writer.close()
        return results


if __name__ == '__main__':
    trainer = Trainer('configs/default_config.yaml')
    trainer.train()