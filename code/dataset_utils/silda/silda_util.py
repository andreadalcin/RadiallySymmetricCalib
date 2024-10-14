from pathlib import Path
import cv2 as cv
import json
import glob
import sys

dataset_utils = Path(__file__).parent / "../"
sys.path.append(str(dataset_utils))
from imaging import *

class SildaGenNameParser(GenNameParser):
    """Generated names parser for the SILDa dataset.
    Filename example: `00-00000_0.png`, composed as `{gen}-{idx}_{cam}}.{ext}`"""

    @property
    def unique_name(self) -> str:
        return self.filename.split('-')[-1]

    @property
    def cam_name(self) -> str:
        return self.unique_name.split('_')[1]

    @property
    def generation_num(self) -> str:
        return self.filename.split('-')[0]

    @property
    def idx(self) -> str:
        return self.unique_name.split('_')[0]

    def compose(self, generation_num:int, cam_name:str = None, idx:str = None, unique_name:str=None) -> str:
        if unique_name:
            unique_name = unique_name.split('.')[0]
            idx, cam_name = unique_name.split('_')

        assert cam_name is not None
        assert idx is not None
        self.filename = f"{generation_num:02d}-{idx}_{cam_name}"
        return self.filename

IMG_DIR = base_path / "data" / "SILDa" / 'mapping' / 'sensors' / "records_data"
MASK_DIR = base_path / "data" / "SILDa" / 'mapping' / 'sensors' / "masks"

def load_descriptor(model = "FOV"):
    if model == "FOV":
        des = cam.FisheyeFOV_Description(
            width = 1024,
            height = 1024,
            is_principle_point_abs=True,
            principle_point = (507.8974358974359,512),
            intrinsics = dict(
                fx = 217.294036,
                fy = 217.214703,
                w = -0.769113,
            )
        )
    elif model == "OPENCV":
        des = cam.FisheyeOPENCV_Description(
            width = 1024,
            height = 1024,
            is_principle_point_abs=True,
            principle_point = (507.8974358974359,512),
            intrinsics = dict(
                fx = 393.299,
                fy = 394.815,
                ks = [-0.223483, 0.117325, -0.0326138, 0.00361082]
            )
        )
    else:
        raise ValueError('Only accepts OPENCV_FISHEYE, or FOV as projection model.')
    return des

def get_camera_masks():
    return {m.split('/')[-1].split('.')[0] :cv.imread(m,cv.IMREAD_GRAYSCALE) for m in glob.glob(str(MASK_DIR) + "/*.*")}

def get_mask(camera_masks, side):
    return camera_masks[side]
