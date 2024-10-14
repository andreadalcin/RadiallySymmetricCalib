import json
import os
import random
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import default_collate
from torchvision import transforms
from PIL import Image  # using pillow-simd for increased speed
from pathlib import Path
from typing import Dict
import cv2 as cv
import warnings

from data_loader.generic_loader import GenericDataset
from projections.cameras import ImageDescription

class FixedSizeDataset(GenericDataset):
    """Fisheye Woodscape Raw dataloader"""

    def __init__(self, data_path=None, path_file=None, is_train=False, config=None):
        super().__init__(data_path, path_file, is_train, config)
        
        self.do_resize = False
        self.data_folder_vangles = self.data_path / "va_vec"
        assert self.data_folder_vangles.exists()


    def get_description(self, file_name: str) -> ImageDescription:
        if not self.is_train:
            return super().get_description(file_name)
        return None
        # Avoind description loading at training time

    def get_va_vec(self, inputs) -> np.ndarray:
        file_name = inputs['file']
        file = f"{file_name}.npy"
        warnings.warn('Va_vec gt data may be old!')
        va_vec = np.load(self.data_folder_vangles / file).astype(np.float32)[::-1].copy() # Rearrange angles according to polar image convention.
        assert va_vec.shape[0] == self.network_input_height // 2
        return va_vec

    def get_image(self, file_name: str) -> np.ndarray:
        image = super().get_image(file_name)
        assert image.width == self.network_input_width and \
             image.height == self.network_input_height
