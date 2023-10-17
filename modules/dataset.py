import os
from typing import Tuple

import albumentations
import cv2
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2

LABEL_INT_MAPPING = {
    'FULL_VISIBILITY': 0,
    'PARTIAL_VISIBILITY': 1,
    'NO_VISIBILITY': 2,
}


class VisibilityDataset(torch.utils.data.Dataset):
    """Torch dataset class to access individual images and visibility labels as samples.

    Attributes:
        images_root_dir (str): Path to the images root directory.
        dataframe (pd.DataFrame): Pandas DataFrame providing access to image names and associated visibility labels.
        spatial_transform (albumentations.Compose, optional): Sequential composition of
            spatial transforms that will be applied to each image. Defaults to None.
        norm_transform (albumentations.Normalize, optional): Normalization transform. Defaults to None.
    """

    def __init__(
        self,
        images_root_dir: str,
        dataframe: pd.DataFrame,
        spatial_transform: albumentations.Compose = None,
        norm_transform: albumentations.Normalize = None,
    ) -> None:
        """Initializes the SingleFrameDataset class.

        Args:
            frames_root_dir (str): Directory path to the video frames root.
            dataframe (pd.DataFrame, optional): Pandas DataFrame providing access to video frames and associated phase labels.
                The input dataframe should simply have "video" and "frame" as columns.
                It is also fine if video and frame are provided as MultiIndex. Defaults to None.
            sampling_step (int, optional): The frame step rate for subsampling. Defaults to None.
            spatial_transform (albumentations.Compose, optional): Sequential composition of
                spatial transforms that will be applied to each frame. Defaults to None.
            norm_transform (albumentations.Normalize, optional): Normalization transform. Defaults to None.
        """
        self.images_root_dir = images_root_dir
        self.spatial_transform = spatial_transform
        self.norm_transform = norm_transform
        self.dataframe = dataframe

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: number of samples
        """
        return len(self.dataframe)

    @staticmethod
    def _load_and_transform_img(image_path: str,
                                spatial_transform: albumentations.Compose = None,
                                norm_transform: albumentations.Compose = None) -> torch.tensor:
        """Loads the image from path and applies the given image transforms. Finally,
        converts it to a torch tensor and returns.

        Args:
            image_path (str): Absolute path to the image.
            spatial_transform (albumentations.Compose): Sequential composition of
                spatial transforms that will be applied to each frame.
            norm_transform (albumentations.Normalize): Pixel normalization transform.

        Returns:
            torch.tensor: preprocessed and normalized image.
        """
        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if spatial_transform:
            image = spatial_transform(image=image)["image"]

        if norm_transform:
            image = norm_transform(image=image)["image"]

        image = ToTensorV2()(image=image)["image"]

        return image

    def __getitem__(self, index: int) -> Tuple:
        """Returns a batch of images and target labels..

        Args:
            index (int): Data point index.

        Returns:
            Tuple[torch.tensor, int]: 4-channel (N,C,H,W) torch tensor representation of batch of images
             and the target labels.
        """
        image_name = self.dataframe.loc[index, 'IMAGE_FILENAME']

        image_path = os.path.join(self.images_root_dir, image_name)

        img = self._load_and_transform_img(image_path, self.spatial_transform, self.norm_transform)
        target = LABEL_INT_MAPPING[self.dataframe.loc[index, 'LABEL']]

        return img, target
