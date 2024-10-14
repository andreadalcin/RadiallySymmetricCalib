import sys
from pathlib import Path

dataset_utils = Path(__file__).parent / "../"
sys.path.append(str(dataset_utils))
from imaging import *

IMG_DIR = base_path / "data" / "SUN360" / "RGB"

class SunGenNameParser(GenNameParser):
    """Generated names parser for the SUN360 dataset.
    Filename example: `00-pano_xyz.png`, composed as `{gen}-{cam}_{idx}.{ext}`"""

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