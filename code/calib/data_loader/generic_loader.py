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

from projections.cameras import ImageDescription
from misc import tools
from misc.misc import get_radial_mask

class GenericDataset(data.Dataset):

    def __init__(self, data_path=None, path_file=None, is_train=False, config=None):
        super().__init__()

        # Base woodscape folder
        self.data_path = Path(data_path)
        self.data_folder_imgs = self.data_path / "rgb_images"
        self.data_folder_calib = self.data_path / "calibration"

        assert self.data_path.exists()
        assert self.data_folder_imgs.exists()
        assert self.data_folder_calib.exists()

        self.file_names = [line.rstrip('\n') for line in open(path_file)]
        self.is_train = is_train

        self.batch_size = config.batch_size

        self.do_rotate = config.do_rotate
        self.do_flip = config.do_flip
        
        self.va_downsample = config.va_downsample

        # If resize should be performed
        self.do_resize = config.do_resize
        self.network_input_width = config.input_width
        self.network_input_height = config.input_height
        self.polar_image_width = config.polar_width

        self.mask = self.get_mask((self.network_input_height,self.network_input_width))

        
        self.load_edges = config.load_edges
        
        self.to_grayscale = transforms.Grayscale()
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((self.network_input_height, self.network_input_width),
                                        interpolation=transforms.InterpolationMode.BICUBIC)
        self.color_aug = transforms.ColorJitter( brightness=(.5, 2),
                                                    contrast=(.5, 2),
                                                    saturation=(.5, 3),
                                                    hue=(-.05, .05))
        self.rotation_aug = transforms.RandomRotation(degrees=(15), interpolation=transforms.InterpolationMode.BICUBIC)
        self.mirror_aug = transforms.RandomHorizontalFlip(p=.3)

    def get_mask(self, size):
        mask = ~get_radial_mask(size, 0 )

        lat = 220j
        mid = 320j

        x, y, z = 0+lat, 199.5+mid, 399+lat
        w = z-x
        w /= y-x
        c = (x-y)*(w-abs(w)**2)/2j/w.imag-x

        xs = np.arange(0,mask.shape[1])
        ys = np.sqrt( np.abs(c+x)**2 - (xs+c.real)**2 ) - c.imag

        mask[np.mgrid[:mask.shape[0],:mask.shape[1]][0] > ys] = True

        return mask
        
    def resize_description(self, description:ImageDescription, scale:float) -> ImageDescription:
        """Scales the intrinsics from original res to the network's initial input res"""
        description.f = description.f * scale
        description.width = self.network_input_width
        description.height = self.network_input_height
        return description

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

    def preprocess(self, inputs: Dict[str,object]):
        """Resize color images to the required scales and augment if required.
        Create the color_aug object in advance and apply the same augmentation to all images in this item.
        This ensures that all images input to the network receive the same augmentation.
        """

        # Input resizing
        if self.do_resize:
            img_aspect_ratio = inputs['color'].width/inputs['color'].height
            final_aspect_ratio = self.network_input_width / self.network_input_height
            assert np.isclose(img_aspect_ratio, final_aspect_ratio)
            inputs['des'] = self.resize_description(inputs['des'], scale = self.network_input_width/inputs['color'].width)
            inputs['color'] = self.resize(inputs['color'])


        # Color augmentation
        if self.is_train:
            inputs['color'] = self.color_aug(inputs['color'])

        # Rotation augmenation
        if self.is_train and self.do_rotate:
            inputs['color'] = self.rotation_aug(inputs['color'])
            
        if self.is_train and self.do_flip:
            inputs['color'] = self.mirror_aug(inputs['color'])

        img = np.array(inputs["color"]) 
        img[self.mask] = 0
        inputs['color'] = Image.fromarray(img)

        # Save grayscale
        inputs['grayscale'] = self.to_grayscale(inputs['color'])
        
        inputs["params"] = np.array(list(inputs['des'].get_intrinsics().values()))

    def get_va_vec(self, inputs):
        va_vec= inputs['des'].get_va_vector()

        if self.va_downsample is not None:
            step = va_vec.shape[0] // self.va_downsample
            va_vec = va_vec[(step-1)::step]

        return va_vec

    def create_and_process_training_items(self, index):
        inputs = dict()
        
        file_name = self.file_names[index]

        inputs["file"] = file_name
        inputs["des"] = self.get_description(file_name=file_name)
        inputs["color"] = self.get_image(file_name=file_name)
        
        self.preprocess(inputs)

        if self.load_edges:
            frame = np.asarray(inputs["color"])
            out = cv.Canny(frame,100,200)
            inputs['edges'] = Image.fromarray(out)
        inputs['va_vec'] = self.get_va_vec(inputs)

        return inputs
    
    def inputs_to_tensors(self, inputs):
        metadata = []
        for k in inputs:

            if k in ["color", 'grayscale', 'edges']:
                inputs[k] = self.to_tensor(inputs[k])

            elif k in ['va_vec','params']:
                inputs[k] = torch.from_numpy(inputs[k]).type(torch.float32)
            else:
                metadata.append(k)
            #    warnings.warn(f"Key \'{k}\' has not a valid conversion specified from {type(inputs[k])} to Tensor.")
        return metadata

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        Keys in the dictionary are strings:
            Tensor values:
                "color"                 color images, optionally resized and color augmented.
                "va_vec"                vector of incident angles in the image plane.
              
            Other values:
                "file"       [no train] filename: file_name, string.
                "des"                   image description instance, original camera information.
        """
        inputs = self.create_and_process_training_items(index)
        metadata = self.inputs_to_tensors(inputs)
        inputs['metadata'] = metadata + ['metadata']
        if self.is_train:
            _ = tools.extract_subset(inputs, inputs['metadata'])

        return inputs

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


