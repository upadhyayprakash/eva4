# data_transform_albumentation.py
from albumentations import (
    RandomGamma, ElasticTransform, IAAPerspective, HueSaturationValue,
    
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, CLAHE, OneOf,HorizontalFlip, ShiftScaleRotate, Compose
)

from albumentations.augmentations.functional import (add_shadow, add_fog, add_sun_flare, iso_noise)

from albumentations.pytorch import (ToTensor)
import albumentations as A
import numpy as np

def augment(aug, image):
    return aug(image=image)['image']

def basic_aug(p=1.0):
    return Compose([
        # HorizontalFlip(p=0.5),
        ToTensor(normalize={'mean':(0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)})
    ], p=p)

def strong_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=0.5),
        # RandomGamma(p=0.5),
        # ElasticTransform(value=10),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=1.),
        
        # Below are Required
        
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        # HueSaturationValue(p=0.3),
        ToTensor(normalize={'mean':(0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)})
    ], p=p)

augStrong = strong_aug(p=1)
basicAug = basic_aug(p=1)

# Functional implementation
def basic_aug_transform(image):
    image = np.array(image)
    return augment(basicAug, image)

def aug(image):
    image = np.array(image)
    return augment(augStrong, image)


# Following is a classical implementation.
class CustomAlbumentation():
    """
    Custom albumentation class
    """
    
    def __init__(self):
        self.transform = A.Compose([
                A.HorizontalFlip(),
                A.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ),
                A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=[0.4914*255, 0.4822*255, 0.4465*255], p=0.5),
                ToTensor(),
            ])
        
    def __call__(self, image):
        # return augmented image
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image = augmented['image']
        return image