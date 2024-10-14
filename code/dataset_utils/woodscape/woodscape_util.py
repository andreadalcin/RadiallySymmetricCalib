from pathlib import Path
import cv2 as cv
import json
import glob
import sys

dataset_utils = Path(__file__).parent / "../"
sys.path.append(str(dataset_utils))
from imaging import *

class WoodscapeGenNameParser(GenNameParser):
    """Generated names parser for the SUN360 dataset.
    Filename example: `00-00000_FV.png`, composed as `{gen}-{idx}_{cam}}.{ext}`"""

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


IMG_DIR = base_path / "data" / "woodscape" / "rgb_images"
CALIB_DIR = base_path / "data" / "woodscape" / 'calibration'
#MASK_DIR = base_path / "data" / "woodscape" / 'masks'
MASK_DIR = base_path / "data" / "woodscape" / 'full_masks'
MASKS_ASSIGN_PATH = base_path / "data" / "woodscape" / 'mask_assignment.json'
if MASKS_ASSIGN_PATH.exists():
    with open(MASKS_ASSIGN_PATH,"r") as f:
        MASK_ASSIGNMENT = json.load(f)

def get_camera_masks():
    return {'_'.join(m.split('.')[0].split('/')[-2:]):cv.imread(m,cv.IMREAD_GRAYSCALE) for m in glob.glob(str(MASK_DIR) + "/**/*.*")}

def load_descriptor(intrinsic_file:Path):
    """generates a Camera object from a json file"""
    with open(intrinsic_file) as f:
        config = json.load(f)

    intrinsic = config['intrinsic']
    coefficients = [intrinsic['k1'], intrinsic['k2'], intrinsic['k3'], intrinsic['k4']]

    return cam.FisheyePOLY_Description(
        height=intrinsic['height'],
        width=intrinsic['width'],
        principle_point=(intrinsic['cx_offset'], intrinsic['cy_offset']),
        intrinsics=dict(distortion_coeffs=coefficients)
    )

def get_mask(camera_masks,idx_side):
    return camera_masks[MASK_ASSIGNMENT[idx_side]]


def main_gen_mask_assignment():
    # Train
    d = {'FV': [{'idx': '06049', 'mask': 'MASK1'},
        {'idx': '06058', 'mask': 'MASK2'},
        {'idx': '06416', 'mask': 'MASK3'},
        {'idx': '06420', 'mask': 'MASK4'},
        {'idx': '07535', 'mask': 'MASK5'},
        {'idx': '07540', 'mask': 'MASK3'},
        {'idx': '08230', 'mask': 'MASK6'}],
        'RV': [{'idx': '06052', 'mask': 'MASK1'},
        {'idx': '06414', 'mask': 'MASK2'},
        {'idx': '06419', 'mask': 'MASK3'},
        {'idx': '08233', 'mask': 'MASK2'}],
        'MVL': [{'idx': '06050', 'mask': 'MASK1'},
        {'idx': '06916', 'mask': 'MASK2'},
        {'idx': '07379', 'mask': 'MASK2'},
        {'idx': '08231', 'mask': 'MASK2'}],
        'MVR': [{'idx': '06051', 'mask': 'MASK1'},
        {'idx': '06417', 'mask': 'MASK1'},
        {'idx': '06423', 'mask': 'MASK2'},
        {'idx': '06913', 'mask': 'MASK1'},
        {'idx': '07381', 'mask': 'MASK1'},
        {'idx': '08232', 'mask': 'MASK1'}]}
    
    # Test
    d = {'FV': [{'idx': '01555', 'mask': 'MASK1'}],
        'RV': [{'idx': '01493', 'mask': 'MASK1'}],
        'MVL': [{'idx': '00758', 'mask': 'MASK1'}],
        'MVR': [{'idx': '01748', 'mask': 'MASK1'}]}

    res = dict()

    parser = WoodscapeGenNameParser()
    for path in IMG_DIR.iterdir():
        parser.set_filename(path.name)

        for e in d[parser.cam_name]:
            if int(e['idx']) >= int(parser.idx):
                break

        res[parser.unique_name] = f"{parser.cam_name}_{e['mask']}"

    with open(MASKS_ASSIGN_PATH,"w") as f:
        json.dump(res, f)


if __name__ == '__main__':
    main_gen_mask_assignment()