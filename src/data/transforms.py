import monai
from random import random
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRanged,
    RandSpatialCropd, RandFlipd, RandRotate90d, RandShiftIntensityd,
    EnsureTyped, AsDiscreted, ToTensord, RandGaussianNoised, RandAdjustContrastd
)


def get_train_transforms(input_shape = (256, 256)):
    """Get training transforms using MONAI"""
    return Compose([
        ScaleIntensityRanged(
            keys=['image'],
            a_min=0.0, a_max=255.0,  # Assuming 8-bit images
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        RandFlipd(keys=['image', 'label', 'skeleton'], spatial_axis=0, prob=0.5),
        RandFlipd(keys=['image', 'label', 'skeleton'], spatial_axis=1, prob=0.5),
        RandRotate90d(
            keys=['image', 'label', 'skeleton'],
            prob=0.4,
            max_k=3,
            spatial_axes=(0, 1)
        ),
        RandShiftIntensityd(keys=['image'], offsets=0.15, prob=0.5),
        RandGaussianNoised(keys=['image'], prob=0.1, mean=0.0, std=0.01),
        RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.8, 1.2)),
    ])

def get_val_transforms(input_shape):
    """Get validation transforms using MONAI"""
    return Compose([
        ScaleIntensityRanged(
            keys=['image'],
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        RandSpatialCropd(
            keys=['image', 'label', 'skeleton'],
            roi_size=input_shape,
            random_size=False,
            random_center=True
        ),
    ])

# ====================
# Custom 3D Occlusion Augmentation
# ====================

class RandomOcclusions:
    def __init__(self, occ_prob=0.8, max_blocks=6, min_size=2, max_size=8):
        self.occ_prob = occ_prob
        self.max_blocks = max_blocks
        self.min_size = min_size
        self.max_size = max_size
    
    def __call__(self, image_dict: dict) -> dict:
        if 'image' not in image_dict:
            return image_dict
        
        if random.random() > self.occ_prob:
            return image_dict
        
        image = image_dict['image']
        C, D, H = image.shape
        
        num_blocks = random.randint(1, self.max_blocks)
        for _ in range(num_blocks):
            block_d = random.randint(self.min_size, self.max_size)
            block_h = random.randint(self.min_size, self.max_size)
            
            d0 = random.randint(0, max(D - block_d, 1))
            h0 = random.randint(0, max(H - block_h, 1))
            
            d1 = min(d0 + block_d, D)
            h1 = min(h0 + block_h, H)
            
            # Set block to zero
            image[:, d0:d1, h0:h1] = 0.0
        
        image_dict['image'] = image
        return image_dict
