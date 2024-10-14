from __future__ import annotations
import numpy as np
import sys
import cv2 as cv
from pathlib import Path
from typing import List, Tuple, Union
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R
import os

base_path = Path(__file__).parent / '../..'
base_path = base_path.resolve()
modules = base_path / "modules"
sys.path.append(str(modules))
import projections.cameras as cam
import projections.proj2d as p2d
from projections import mappings

@dataclass
class RotParams():
    extrinsic_rot: Tuple[float, float, float]
    min_intrinsics: dict
    max_intrinsics: dict
    max_rot_delta: list = field(default_factory=lambda: [0,0,0])
    min_rot_delta: list = field(default_factory=lambda:[0,0,0])

@dataclass
class GenParams():
    camera_name:str
    params_by_rot:List[RotParams]

class BigCropException(Exception):
    def __init__(self, crop_size, img_size, message=None):
        self.crop_size = crop_size
        self.img_size = img_size
        self.message = message
        if message is None:
            self.message = f"Image of size {self.img_size} is being cropped to size {self.crop_size}"
        
        super().__init__(self.message)

def rectify_img(img: np.ndarray, des:cam.ImageDescription, afov=100) -> np.ndarray:
    out_des = cam.Perspective_Description(
        width=1000,
        height=1000,
        intrinsics=dict(
            afov = afov,
        )
    )

    img_rect,_ = mappings.map_img(img, [des, out_des] )
    return img_rect

def get_circular_mask(mask:np.ndarray):
    """Refine mask returning the biggest cirular visible area.

    Args:
        mask (np.ndarray): _description_

    Returns:
        _type_: _description_
    """

    size = mask.shape[:2]
    rad = min(size) / 2
    grid = np.mgrid[:size[0],:size[1]].astype(np.float32)
    grid -= np.array([(size[0]-1)/2, (size[1]-1)/2]).reshape(-1,1,1)
    r_2 = grid[0,...] ** 2 + grid[1,...] ** 2

    min_r2 = min(np.min(r_2[~(mask>0)]), rad**2)

    circle_mask = (r_2 <= min_r2) | (r_2 >= (np.sqrt(min_r2)+2)**2)
    full_mask = (r_2 <= min_r2)

    return circle_mask, full_mask

def get_crop_area(mask:np.ndarray):
    mask = mask.astype(bool)
    w = np.argwhere(mask)
    bl = np.min(w, axis=0)
    tr = np.max(w, axis=0)
    return bl, tr

def crop_img_mask(image:np.ndarray, mask:np.ndarray, raise_exception=True, min_crop_size=400):
    bl, tr = get_crop_area(mask)

    crop_img = image[bl[0]:tr[0]+1,bl[1]:tr[1]+1]
    crop_mask = mask[bl[0]:tr[0]+1,bl[1]:tr[1]+1]

    crop_size = (tr-bl)
    img_size = np.array(image.shape[:2])

    if raise_exception and np.any(crop_size < min_crop_size):
        raise BigCropException(crop_size=crop_size, img_size=img_size)

    return crop_img, crop_mask

def project_and_mask(image:np.ndarray, mask:np.ndarray, des_list:List[cam.ImageDescription],\
     raise_exception=True, interpolation:int = cv.INTER_CUBIC, circular_masking:bool=False):
    proj_image,_ = mappings.map_img(img=image, des_list=des_list, interpolation=interpolation)
    proj_mask,_  = mappings.map_img(img=mask, des_list=des_list, interpolation=interpolation)

    circle_mask, full_mask = get_circular_mask(proj_mask)

    proj_mask = circle_mask if circular_masking else full_mask

    fin_image, fin_mask = crop_img_mask(proj_image, proj_mask, raise_exception=raise_exception)
    
    fin_image_masked = cv.bitwise_and(fin_image,fin_image,mask = fin_mask.astype(np.uint8))
    return fin_image_masked

def project_mask(mask:np.ndarray, des_list:List[cam.ImageDescription],\
     interpolation:int = cv.INTER_NEAREST, circular_masking:bool=False):
    proj_mask,_  = mappings.map_img(img=mask, des_list=des_list, interpolation=interpolation)

    circle_mask, full_mask = get_circular_mask(proj_mask)

    proj_mask = circle_mask if circular_masking else full_mask

    bl, tr = get_crop_area(proj_mask)

    crop_mask = proj_mask[bl[0]:tr[0]+1,bl[1]:tr[1]+1]

    crop_size = (tr-bl)

    return crop_mask, crop_size

def project_and_mask_no_crop(image:np.ndarray, mask:np.ndarray, des_list:List[cam.ImageDescription], raise_exception=True):
    proj_image,_ = mappings.map_img(img=image, des_list=des_list)
    proj_mask,_  = mappings.map_img(img=mask, des_list=des_list)

    circle_mask, _ = get_circular_mask(proj_mask)

    fin_image_masked = cv.bitwise_and(proj_image,proj_image,mask = circle_mask.astype(np.uint8))
    return fin_image_masked

### GENERATION

def get_gen_params_from_mask(mask_name:str, params_list: List[GenParams]) -> GenParams:
    for gen_params in params_list:
        if mask_name == gen_params.camera_name:
            return gen_params

    raise ValueError(f"Mask {mask_name} is not valid! ")

def get_random_des(gen_params:GenParams, descriptor_class, height:int = 1000, width:int = 1000 ) -> cam.ImageDescription:

    view:RotParams = np.random.choice(gen_params.params_by_rot)
    extrinsic_rot = view.extrinsic_rot

    intrinsics = {}
    for k in view.max_intrinsics:
        intrinsics[k] = np.random.uniform(
            low=view.min_intrinsics[k],
            high=view.max_intrinsics[k])

    return descriptor_class(
        width=width, 
        height=height,
        intrinsics=intrinsics, 
        extrinsic_rot=extrinsic_rot)

def get_random_intrinsic_param(gen_rot:RotParams, param:str) -> float:

    assert param in gen_rot.min_intrinsics

    return np.random.uniform(
            low=gen_rot.min_intrinsics[param],
            high=gen_rot.max_intrinsics[param])


def get_min_des_by_rot(gen_params:GenParams, descriptor_class, height:int = 1000, width:int = 1000 ) -> List[cam.ImageDescription]:

    descriptors = []

    for view in gen_params.params_by_rot:
        extrinsic_rot = view.extrinsic_rot

        intrinsics = {}
        for k in view.min_intrinsics:
            intrinsics[k] = view.min_intrinsics[k]

        descriptors.append(
            descriptor_class(
                width=width, 
                height=height,
                intrinsics=intrinsics, 
                extrinsic_rot=extrinsic_rot)
        )

    return descriptors

def get_max_des_by_rot(gen_params:GenParams, descriptor_class, height:int = 1000, width:int = 1000 ) -> List[cam.ImageDescription]:

    descriptors = []

    for view in gen_params.params_by_rot:
        extrinsic_rot = view.extrinsic_rot

        intrinsics = {}
        for k in view.max_intrinsics:
            intrinsics[k] = view.max_intrinsics[k]

        descriptors.append(
            descriptor_class(
                width=width, 
                height=height,
                intrinsics=intrinsics, 
                extrinsic_rot=extrinsic_rot)
        )

    return descriptors

class GenNameParser():
    def __init__(self, filename:Union[Path, str] = None) -> None:
        if filename is not None:
            self.set_filename(filename)

    def set_filename(self, filename:Union[Path, str]) -> GenNameParser:
        if isinstance(filename, Path):
            filename = filename.name
        elif isinstance(filename, str):
            filename = filename.split("/")[-1]
        else:
            raise TypeError(f"Not supported: {type(filename)}")

        if "." in filename:
            filename = filename.split('.')[0]
        
        self.filename = filename
        return self

    @property
    def unique_name(self) -> str:
        """Unique identifier of the image it is derived from."""
        return None

    @property
    def generation_num(self) -> str:
        """The number of the image duplicates"""
        return None

    @property
    def cam_name(self) -> str:
        """The name of the camera image duplicates"""
        return None

    @property
    def idx(self) -> str:
        """The orginal identifier of the base image, may not be unique!"""
        return None
    


def _interpolate_rot(r1, r2, alpha):
    _r = R.from_euler("xyz", [r1,r2], degrees = True)

    weights = [1-alpha, alpha ]
    return _r.mean(weights=weights).as_euler("xyz", degrees = True).tolist()

def _add_rots(r1, r2):
    _r1 = R.from_euler("xyz", r1, degrees = True)
    _r2 = R.from_euler("xyz", r2, degrees = True)
    comb_r = _r1 * _r2
    return comb_r.as_euler("xyz", degrees = True).tolist()

def random_rotation(gen_params:GenParams, alpha=None, beta=None):
    if alpha is None:
        alpha = np.random.default_rng().uniform(-1,1)

    r1 = gen_params.params_by_rot[1]
    if alpha < 0:
        alpha *= -1
        # Center to left
        r2 = gen_params.params_by_rot[0]
    else:
        # Center to right
        r2 = gen_params.params_by_rot[2]

    if beta is None:
        d1 = np.random.default_rng().uniform(r1.min_rot_delta,r2.max_rot_delta)
        d2 = np.random.default_rng().uniform(r2.min_rot_delta,r2.max_rot_delta)
    else:
        d1 = np.array(r1.min_rot_delta) * (1-beta) + np.array(r1.max_rot_delta) * beta
        d2 = np.array(r2.min_rot_delta) * (1-beta) + np.array(r2.max_rot_delta) * beta
    
    _r1 = _add_rots(r1.extrinsic_rot, d1)
    _r2 = _add_rots(r2.extrinsic_rot, d2)

    new_rot = _interpolate_rot(_r1, _r2, alpha)
    
    return new_rot
    

def check_distance(va_vecs, new, method="mse"):
    va_vecs = np.asarray(va_vecs)
    res = va_vecs - new
    if method == "mse":
        dist = np.linalg.norm(res, axis=1)
    else:
        raise NotImplementedError()
    return np.min(dist)

def check_afov_distance(va_vecs, new):
    va_vecs = np.asarray(va_vecs)
    res = np.abs(va_vecs[:,-1] - new[-1])
    return np.min(res)

def compute_max_afov(mask:np.ndarray, des:cam.ImageDescription, rot, max_afov = 220):
    big_des = cam.FisheyeDS_Description(
        width=700,
        height=700,
        intrinsics=dict(
        afov = max_afov,
        a = 0.5,
        xi = -0.3,
        ),
        extrinsic_rot=rot,
    )
    proj_mask, cropped_size = project_mask(mask, des_list=[des, big_des])
    big_des.width = cropped_size[0]
    big_des.height = cropped_size[1]

    return big_des.get_afov()