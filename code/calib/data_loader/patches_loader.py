import numpy as np
import cv2 as cv
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import sys
import copy

base_path = Path(__file__, '../../../../').resolve()
sys.path.append(str(base_path / 'code/calib'))
DATA_PATH = base_path / 'data'
from misc import pipeline
from misc import tools

class PatchesDataset(Dataset):
    default_config = {
        'dataset': 'hpatches',  # or 'coco'
        'alteration': 'all',  # 'all', 'i' for illumination or 'v' for viewpoint
        'cache_in_memory': False,
        'truncate': None,
        'preprocessing': {
            'resize': [400,400],
            'valid_border_margin':2,
        }
    }

    to_tensor = ToTensor()

    def __init__(self, **config) -> None:
        super().__init__()
        self.config = tools.dict_update(copy.deepcopy(getattr(self, 'default_config', {})), config)
        config = self.config

        dataset_folder = 'COCO/patches' if config['dataset'] == 'coco' else 'HPatches'
        base_path = Path(DATA_PATH, dataset_folder)
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        for path in folder_paths:
            if config['alteration'] == 'i' and path.stem[0] != 'i':
                continue
            if config['alteration'] == 'v' and path.stem[0] != 'v':
                continue
            num_images = 1 if config['dataset'] == 'coco' else 5
            file_ext = '.ppm' if config['dataset'] == 'hpatches' else '.jpg'
            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(path, "1" + file_ext)))
                warped_image_paths.append(str(Path(path, str(i) + file_ext)))
                homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))

        if config['truncate']:
            image_paths = image_paths[:config['truncate']]
            warped_image_paths = warped_image_paths[:config['truncate']]
            homographies = homographies[:config['truncate']]

        self.image_paths = image_paths
        self.warped_image_paths = warped_image_paths
        self.homographies = homographies
    

    def __len__(self):
        return len(self.image_paths)
    
    def _preprocess(self, image):
        image, valid_mask, mask_h = pipeline.mask_and_resize(image, **self.config['preprocessing'])

        return image, valid_mask, mask_h

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        
        """
        def _read_image(path):
            return cv.imread(path, cv.IMREAD_GRAYSCALE)

        img_path = self.image_paths[index]
        warped_image_path = self.warped_image_paths[index]
        homography = self.homographies[index]

        image = _read_image(img_path)
        warped_image = _read_image(warped_image_path)

        inputs = {
            'base':{},
            'warped':{},
        }
        base = inputs['base']
        warped = inputs['warped']
        base['original'] = image
        base['image'],base['mask'],base['h'] = self._preprocess(image)
        warped['original'] = warped_image
        warped['image'],warped['mask'],warped['h'] = self._preprocess(warped_image)
        inputs['homography'] = homography
        inputs['res_homography'] = warped['h'] @ inputs['homography'] @ np.linalg.inv(base['h'])

        inputs['name'] = Path(warped_image_path).parent.name +'_'+ Path(warped_image_path).name.split('.')[0]

        return inputs

    @staticmethod
    def img_to_tensor(img:np.ndarray):
        return PatchesDataset.to_tensor(img.astype(np.uint8))
        