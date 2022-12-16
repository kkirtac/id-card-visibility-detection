"""Defines the class to be used to perform spatial transforms on images."""
from typing import List
import albumentations


class TransformationsBase:
    """Data augmentation and preprocessing for images."""

    def __init__(self, scale_size: int, norm_mean: List, norm_std: List = None):
        """Initialize the parameters and transforms.
        
        Args:
            scale_size (int): scale the image to square size of this given int.
            norm_mean (List): list of 3 float numbers in range (0, 1]. Used to describe the normalization mean pixel 
                value for R,G,B -> [R_mean, G_mean, B_mean].
            norm_std (List, optional): list of 3 float numbers each in range (0, 1]. Defaults to None. Used to 
                describe the normalization std pixel value for R,G,B -> [R_std, G_std, B_std].
        """
        self.scale_size = scale_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        if self.norm_std is None:
            self.norm_std = [1., 1., 1.]

        # Data augmentations to apply to training data
        self.train_transform = albumentations.Compose([
            albumentations.Resize(height=self.scale_size, width=self.scale_size),
            albumentations.ShiftScaleRotate(shift_limit=0.1,
                                            scale_limit=(-0.2, 0.5),
                                            rotate_limit=15,
                                            border_mode=0,
                                            value=0,
                                            p=0.7),
        ])

        # Transformations for evaluation (validation/test dataset)
        self.eval_transform = albumentations.Compose([
            albumentations.Resize(height=self.scale_size, width=self.scale_size),
        ])

        self.norm_transform = albumentations.Normalize(self.norm_mean, self.norm_std)
