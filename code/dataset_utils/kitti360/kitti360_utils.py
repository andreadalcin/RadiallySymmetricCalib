import os
import numpy as np
import re
import yaml
import sys
import cv2 as cv
from pathlib import Path
from typing import List, Dict, Tuple, Union

dataset_utils = Path(__file__).parent / "../"
sys.path.append(str(dataset_utils))
from imaging import *

IMG_DIR = base_path / "data" / "KITTI-360" / "data_2d_raw"
CALIB_DIR = base_path / "data" / "KITTI-360" / 'calibration'
#MASK_DIR = base_path / "data" / "KITTI-360" / 'masks'
MASK_DIR = base_path / "data" / "KITTI-360" / 'full_masks'

def readYAMLFile(fileName: Path):
    '''make OpenCV YAML file compatible with python'''
    ret = {}
    skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.safe_load(yamlFileOut)
    return ret

def load_descriptor(intrinsic_file:Path) -> cam.ImageDescription:
    intrinsics = readYAMLFile(intrinsic_file)

    des = cam.FisheyeMEI_Description(
        width=intrinsics['image_width'],
        height=intrinsics['image_height'],
        intrinsics=dict(
            mirror_parameters=intrinsics["mirror_parameters"],
            distortion_parameters=intrinsics["distortion_parameters"],
            projection_parameters=intrinsics["projection_parameters"],
        )
    )

    return des

class KittiGenNameParser(GenNameParser):
    """Generated names parser for the Kitti360 dataset.
    Filename example: `00-02_0000000028.png`, composed as `{gen}-{cam}_{idx}.{ext}`"""

    @property
    def unique_name(self) -> str:
        return self.filename.split('-')[-1]

    @property
    def cam_name(self) -> str:
        return self.unique_name.split('_')[0]

    @property
    def generation_num(self) -> str:
        return self.filename.split('-')[0]

    @property
    def idx(self) -> str:
        return self.unique_name.split('_')[-1]
    
    def compose(self, generation_num:int, cam_name:str = None, idx:str = None, unique_name:str=None) -> str:
        if unique_name:
            unique_name = unique_name.split('.')[0]
            cam_name, idx = unique_name.split('_')

        assert cam_name is not None
        assert idx is not None
        self.filename = f"{generation_num:02d}-{cam_name}_{idx}"
        return self.filename