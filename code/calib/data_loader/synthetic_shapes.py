import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import default_collate
import warnings
from PIL import Image
import copy

import sys
from pathlib import Path
base_path = Path(__file__).parent / '../'
sys.path.append(str(base_path))

from misc import tools
from misc import pipeline
from data_loader import synthetic_dataset
from torchvision import transforms

class SyntheticShapes(data.Dataset):
    default_config = {
            'primitives': 'all',
            'truncate': {},
            'validation_size': -1,
            'test_size': -1,
            'on-the-fly': True,
            'cache_in_memory': False,
            'suffix': None,
            'add_augmentation_to_test_set': False,
            'num_parallel_calls': 10,
            'generation': {
                'split_sizes': {'training': 10000, 'validation': 200, 'test': 500},
                'image_size': [600, 600],
                'random_seed': 0,
                'params': {
                    'generate_background': {
                        'min_kernel_size': 150, 'max_kernel_size': 300,
                        'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
                    'draw_stripes': {'transform_params': (0.1, 0.1)},
                    'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
                },
            },
            'preprocessing': {
                'resize': [600, 600],
                'blur_size': 11,
            },
            'augmentation': {
                'photometric': {
                    'enable': True,
                    'primitives': 'all',
                    'params': {},
                    'random_order': True,
                },
                'warp': {
                    'enable': True,
                    'input_afov': 90,
                    'params': {
                        'out_size' : (400,400),
                        'range_f' : (250, 500),
                        'range_a' : (0, 1),
                        'range_xi' : (-0.5, 0.5),
                        'range_r' : (180,20,20),  # Roll, Pitch, Yaw
                    },
                    'valid_border_margin': 4,
                },
            }
    }
    drawing_primitives = [
            'draw_lines',
            'draw_polygon',
            'draw_multiple_polygons',
            'draw_ellipses',
            'draw_star',
            'draw_checkerboard',
            'draw_stripes',
            'draw_cube',
            'gaussian_noise'
    ]
    split_names = ['training', 'validation', 'test']

    def __init__(self, size:int = -1, is_train=False, config=default_config) -> None:
        super().__init__()
        self.config = tools.dict_update(copy.deepcopy(getattr(self, 'default_config', {})), config)

        self.size = size
        self.is_train = is_train

        self.photometric_aug = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1)
        self.homographic_aug = transforms.RandomPerspective()
        self.preprocessing = transforms.Compose([
            transforms.GaussianBlur(self.config['preprocessing']['blur_size']),
            transforms.Resize(self.config['preprocessing']['resize']),
        ])

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.size

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
        inputs = self.create_and_process_training_items()

        # Clean inputs and convert to tensors
        if self.is_train:
            inputs.pop('base_img', None)
            inputs.pop('base_kps', None)
            inputs.pop('kps', None)
            inputs.pop('warp', None)

        metadata = self.inputs_to_tensors(inputs)
        inputs['metadata'] = metadata + ['metadata']
        if self.is_train:
            _ = tools.extract_subset(inputs, inputs['metadata'])

        return inputs

    def inputs_to_tensors(self, inputs):
        metadata = []
        for k in inputs:

            if k in ['img']:
                inputs[k] = self.to_tensor(inputs[k].astype(np.uint8))

            elif k in ['valid_mask', 'kps_map']:
                inputs[k] = torch.from_numpy(inputs[k]).type(torch.float32)

            else:
                metadata.append(k)
                #warnings.warn(f"Key \'{k}\' has not a valid conversion specified from {type(inputs[k])} to Tensor.")
        return metadata

    def create_and_process_training_items(self):
        inputs = {}

        def _gen_shape():
            primitives = pipeline.parse_primitives(self.config['primitives'], self.drawing_primitives)
            
            primitive = np.random.choice(primitives)
            image = synthetic_dataset.generate_background(
                    self.config['generation']['image_size'],
                    **self.config['generation']['params']['generate_background'])
            points = np.array(getattr(synthetic_dataset, primitive)(
                    image, **self.config['generation']['params'].get(primitive, {})))
            return (np.expand_dims(image, axis=-1).astype(np.float32),
                    points.astype(np.float32))

        if self.config['on-the-fly']:
            inputs['img'], inputs['kps'] = _gen_shape()

            # Apply preprocessing
            scale = np.array(self.config['preprocessing']['resize']) / inputs['img'].shape[:2]
            inputs['kps'] = inputs['kps'] * scale

            image = torch.from_numpy(inputs['img']).permute((2,0,1)) / 255.
            image = self.preprocessing(image)
            inputs['img'] = np.asarray(image.permute((1,2,0)))* 255.
        else:
            raise NotImplementedError()

        inputs = pipeline.add_dummy_valid_mask(inputs)

        # Apply augmentation
        if self.is_train or self.config['add_augmentation_to_test_set']:
            if self.config['augmentation']['photometric']['enable']:
                image = torch.from_numpy(inputs['img']).permute((2,0,1)) / 255.
                image = self.photometric_aug(image)
                inputs['img'] = np.asarray(image.permute((1,2,0))) * 255.
                inputs = pipeline.photometric_augmentation(inputs, **self.config['augmentation']['photometric'])
                
        if self.config['augmentation']['warp']['enable']:
            inputs = pipeline.warp_augmentation(inputs, **self.config['augmentation']['warp'])

        # Convert the point coordinates to a dense keypoint map
        inputs = pipeline.add_keypoint_map(inputs)

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


if __name__ == '__main__':
    
    dataset = SyntheticShapes(size = 100, is_train=True)
    dataset.__getitem__(0)