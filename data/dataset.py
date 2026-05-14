import torch
import tifffile
import cv2
import numpy as np
from torch.utils.data import Dataset
import json

from .transforms import get_train_transforms, get_val_transforms, RandomOcclusions


class VessMapDataset(Dataset):
    """Dataset for VessMAP vessel segmentation."""
    
    def __init__(
        self,
        items,
        data_dir: str = '/Users/egor/VS_GIT_repositories/TopologyLosses/VessMAP/',
        train: bool = False,
        use_occlusion: bool = False,
        return_skeleton: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            items: List of image IDs
            data_dir: Path to VessMAP dataset
            train: Whether to use training transforms
            use_occlusion: Whether to apply occlusion augmentation
            return_skeleton: Whether to return skeleton annotations
        """
        self.items = items
        self.train = train
        self.transforms = get_train_transforms() if train else get_val_transforms(256)
        self.data_dir = data_dir
        self.return_skeleton = return_skeleton
        self.use_occlusion = use_occlusion and train  # Only apply during training
        self.occlusion = RandomOcclusions()
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image_id = self.items[idx]
        
        # Load image
        image = tifffile.imread(f'{self.data_dir}images/{image_id}.tiff')
        if image.ndim == 2:
            image = image[..., np.newaxis]
        image = image.astype(np.float32)
        
        # Load label
        label = cv2.imread(f'{self.data_dir}annotator1/labels/{image_id}.png', cv2.IMREAD_GRAYSCALE)
        label = label.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Load skeleton if needed
        if self.return_skeleton:
            skeleton = cv2.imread(
                f'{self.data_dir}annotator1/skeletons/{image_id}.png',
                cv2.IMREAD_GRAYSCALE
            )
            skeleton = skeleton.astype(np.float32) / 255.0
        else:
            skeleton = np.zeros_like(label)
        
        data_dict = {
            'image': np.expand_dims(image[:,:, 0], axis=0),  # Use only the first channel if multiples
            'label': np.expand_dims(label, axis=0),
            'skeleton': np.expand_dims(skeleton, axis=0)
        }
        # Apply transforms if available
        if self.transforms:
            transformed = self.transforms(data_dict)
            if self.use_occlusion:
                transformed = self.occlusion(transformed)
            
            # Handle different transform outputs
            if isinstance(transformed.get('image'), torch.Tensor):
                image = transformed['image']
                label = transformed['label']
                skeleton = transformed['skeleton']
            else:
                # Fallback if transforms return dict-like objects
                image = torch.from_numpy(transformed['image']).float()
                label = torch.from_numpy(transformed['label']).float()
                skeleton = torch.from_numpy(transformed['skeleton']).float()
        else:
            # Convert to tensors without transforms
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()
            skeleton = torch.from_numpy(skeleton).float()
        
        # Ensure correct shapes
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if label.ndim == 3:
            label = label.squeeze(0)
        if skeleton.ndim == 3:
            skeleton = skeleton.squeeze(0)
        
        return {
            'image': image.as_tensor(),
            'label': label.unsqueeze(0).as_tensor() ,  # (1, H, W)
            'skeleton': skeleton.unsqueeze(0).as_tensor(),  # (1, H, W)
            'image_id': image_id
        }


def load_measures(json_path):
    """Load measures from JSON file."""
    with open(json_path, 'r') as f:
        measures = json.load(f)
    return measures


def create_data_splits(items, test_size=0.25, val_size_from_test=0.2, random_seed=42):
    """
    Create train/val/test splits.
    
    Args:
        items: List of all image IDs
        test_size: Fraction of data for testing
        val_size_from_test: Fraction of test split to use for validation
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_items, val_items, test_items)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train vs (val+test)
    train_items, val_test_items = train_test_split(
        items,
        test_size=test_size,
        shuffle=True,
        random_state=random_seed
    )
    
    # Second split: val vs test
    val_items, test_items = train_test_split(
        val_test_items,
        test_size=val_size_from_test,
        shuffle=True,
        random_state=random_seed
    )
    
    return train_items, val_items, test_items