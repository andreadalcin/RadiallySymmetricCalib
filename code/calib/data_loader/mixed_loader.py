import json
import warnings
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

import sys
base_path = Path(__file__).parent / '../../../'
sys.path.append(str(base_path / 'modules'))
sys.path.append(str(base_path / 'code/calib'))

from misc import misc, tools
from projections.cameras import ImageDescription

class MixedDataset(data.Dataset):
    """Fisheye Woodscape Raw dataloader"""

    def __init__(self, data_path=None, path_file=None, is_train=False, config=None):
        super().__init__()

        # Base data folder
        self.data_path = Path(data_path)
        self.data_folder_imgs = self.data_path / "rgb_images"
        self.data_folder_calib = self.data_path / "calibration"
        self.data_folder_kps = self.data_path / "kps"

        self.kps_size = config.kps_size

        assert self.data_path.exists()
        assert self.data_folder_imgs.exists()
        assert self.data_folder_calib.exists()

        self.load_kps = not config.no_gt and 'calib' not in config.task
        self.load_va_vec = not config.no_gt and 'des' not in config.task
        if self.load_kps:
            assert self.data_folder_kps.exists()

        self.file_names = [line.rstrip('\n') for line in open(path_file)]
        self.is_train = is_train

        # Current training task (Calibration, keypoint detection, ...)
        self.batch_size = config.batch_size

        self.network_input_width = config.input_width
        self.network_input_height = config.input_height
        self.polar_image_width = config.polar_width

        # Kps
        self.valid_border_margin = config.valid_border_margin
        self.grayscale = config.grayscale

        # Resize should not be performed
        # Rolar mapping should not be performed
        assert config.do_resize == True
        assert config.do_polar_mapping == True

        self.color_aug = None
        self.load_edges = config.load_edges

        self.to_grayscale = transforms.Grayscale()
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((self.network_input_height, self.network_input_width))
        
    def get_image(self, file_name:str) -> Image.Image:
        file = f"{file_name}.png"
        image = Image.open(self.data_folder_imgs / file).convert("RGB")
        return image

    def get_description(self, file_name:str) -> ImageDescription:
        file = f"{file_name}.json"
        with open(self.data_folder_calib / file , "r") as f:
            des_dict = json.load(f)

        des = ImageDescription.from_des_dict(des_dict=des_dict)
        return des

    def get_va_vec(self, file_name:str) -> np.ndarray:
        file = f"{file_name}.npy"
        va_vec = np.load(self.data_folder_vangles / file).astype(np.float32)[::-1].copy() # Rearrange angles according to polar image convention.
        return va_vec
    
    def get_kps(self, file_name:str) -> np.ndarray:
        file = f"{file_name}.npy"
        kps = np.load(self.data_folder_kps / file)
        return kps
    
    def resize_description(self, description:ImageDescription, scale:float) -> ImageDescription:
        """Scales the intrinsics from original res to the network's initial input res"""
        description.f = description.f * scale
        description.width = self.network_input_width
        description.height = self.network_input_height
        return description
    

    def add_keypoint_map(self, kps, size) -> np.ndarray:
        image_shape = np.asarray(size)
        kp = np.minimum(np.round(kps).astype(np.int32), image_shape-1)
        kmap = np.zeros(image_shape)
        if len(kp.shape) != 2 or kp.shape[0] ==0:
            return kmap
        kmap[kp[:,1],kp[:,0]] = 1
        return kmap

    def preprocess(self, inputs: Dict[str,object]):
        """Resize color images to the required scales and augment if required.
        Create the color_aug object in advance and apply the same augmentation to all images in this item.
        This ensures that all images input to the network receive the same augmentation.
        """

        # Save grayscale
        if self.grayscale:
            inputs['img']:Image.Image = self.to_grayscale(inputs['color'])
        else:
            inputs['img']:Image.Image = inputs['color']
        
        # Resize
        in_size = inputs['img'].size
        inputs['img'] = self.resize(inputs['img'])
        out_size = inputs['img'].size
        scale = out_size[0] / in_size[0] # TODO assertion
        inputs['des'] = self.resize_description(inputs['des'], scale=scale)


        # Photometric augmentation
        inputs['img'] = self.color_aug(inputs['img'])

        inputs['valid_mask'] = misc.get_radial_mask(out_size, self.valid_border_margin).astype(np.int32)
        
        if self.load_kps:
            # KPS
            inputs['kps'] = inputs['kps'].astype(np.float32) / self.kps_size * out_size[0]
            inputs['kps_map'] = self.add_keypoint_map(inputs['kps'], out_size)

        if self.load_va_vec:
            # Angles
            inputs['va_vec'] =  inputs['des'].get_va_vector() # remove the first angle, useless


    def inputs_to_tensors(self, inputs):
        metadata = []
        for k in inputs:

            if k in ['img']:
                inputs[k] = self.to_tensor(inputs[k])

            elif k in ['valid_mask', 'kps_map','va_vec']:
                inputs[k] = torch.from_numpy(inputs[k]).type(torch.float32)

            else:
                metadata.append(k)
            #    warnings.warn(f"Key \'{k}\' has not a valid conversion specified from {type(inputs[k])} to Tensor.")
        return metadata
        

    def create_and_process_training_items(self, index):
        inputs = dict()
        do_color_aug = self.is_train and random.random() > 0.5
        file_name = self.file_names[index]

        inputs["file"] = file_name
        
        inputs["des"] = self.get_description(file_name=file_name)
        inputs["color"] = self.get_image(file_name=file_name)

        if self.load_kps:
            inputs["kps"] = self.get_kps(file_name=file_name)
        
        if do_color_aug:
            self.color_aug = transforms.ColorJitter(brightness=(0.8, 1.2),
                                                    contrast=(0.8, 1.2),
                                                    saturation=(0.8, 1.2),
                                                    hue=(-0.1, 0.1))
        else:
            self.color_aug = (lambda x: x)

        self.preprocess(inputs)

        metadata = self.inputs_to_tensors(inputs)
        inputs['metadata'] = metadata + ['metadata']
        if self.is_train:
            _ = tools.extract_subset(inputs, inputs['metadata'])

        return inputs

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        Keys in the dictionary are strings:
            Tensor values:
                "color"                 color images, optionally resized and color augmented.
                "va_vec"                vector of incident angles in the image plane.
                "semantic_labels"       semantic mask of the image.
                "pad"                   added in the collate phase, padding added to inputs to reach 
                                        the same input sized in the same batch.
                                        
            Other values:
                "file"       [no train] filename: file_name, string.
                "des"        [no train] image description instance, original camera information.
        """
        return self.create_and_process_training_items(index)

    def collate_fn(self, batch):
        
        metadata = batch[0].pop('metadata')

        # Collate elements that are not tensors
        manually_collated = dict()
        for key in metadata:
            if key in batch[0].keys():
                values = [inputs.pop(key) for inputs in batch]
                manually_collated[key] = values

        collated = default_collate(batch)
        # Add manually collated key-values
        collated.update(manually_collated)
        collated['metadata'] = metadata
        return collated

if __name__ == '__main__':
    import sys
    base_path = Path(__file__).parent / '../../../'
    sys.path.append(str(base_path / 'modules'))
    sys.path.append(str(base_path / 'code/calib'))

    from ruamel import yaml
    from misc.misc import Tupperware
    from misc import pipeline

    config = 'params/caldet.yaml'

    params = yaml.safe_load(open(config))
    args = Tupperware(params)

    ds = MixedDataset(
        data_path=args.dataset_dir,
        path_file=args.train_file,
        is_train=True,
        config=args
    )

    inputs = ds.__getitem__(0)
    print(inputs['valid_mask'].shape)