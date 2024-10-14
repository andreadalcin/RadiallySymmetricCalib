""" Adaptation in python3 of the DeepCalib continuous dataset generation
"""
import numpy as np
from pathlib import Path
import sys
import json
import cv2 as cv
import random
from scipy.spatial.transform import Rotation as R

dataset_utils = Path(__file__).parent / "../"
sys.path.append(str(dataset_utils))
import sun360_utils as su
import os
from tqdm.auto import tqdm

random.seed(9001)
np.random.seed(1)

def deg2rad(deg):
    return deg*np.pi/180

def getRotationMat(roll, pitch, yaw):

    rx = np.array([1., 0., 0., 0., np.cos(deg2rad(roll)), -np.sin(deg2rad(roll)), 0., np.sin(deg2rad(roll)), np.cos(deg2rad(roll))]).reshape((3, 3))
    ry = np.array([np.cos(deg2rad(pitch)), 0., np.sin(deg2rad(pitch)), 0., 1., 0., -np.sin(deg2rad(pitch)), 0., np.cos(deg2rad(pitch))]).reshape((3, 3))
    rz = np.array([np.cos(deg2rad(yaw)), -np.sin(deg2rad(yaw)), 0., np.sin(deg2rad(yaw)), np.cos(deg2rad(yaw)), 0., 0., 0., 1.]).reshape((3, 3))

    return np.matmul(rz, np.matmul(ry, rx))

def main(img_dir:Path, calib_dir:Path, num_copies:int=1):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    parser = su.SunGenNameParser()

    H=400
    W=400

    max_iters = 100

    for img_path in tqdm(su.IMG_DIR.iterdir(),total= len(os.listdir(su.IMG_DIR))):
            
        img = cv.imread(str(img_path))
        des = su.cam.Equirectangular_Description(
            width=img.shape[1],
            height=img.shape[0]
        )

        i = 0 
        for g in tqdm(range(num_copies), leave=False):
            while i<max_iters:
                try:
                    f = random.randint(50,500) / 299 * H # rescale the focal to the new resolution
                    xi = random.uniform(0,1.2)

                    # 3. Rotation of the sphere
                    Rot = []
                    Rot.append(((np.random.default_rng().random() - 0.5) * 2) * 10) # roll
                    Rot.append(((np.random.default_rng().random() - 0.5) * 2) * 15) # pitch
                    Rot.append(((np.random.default_rng().random() - 0.5) * 2) * 180) # yaw

                    # extrinsic_rot = [roll, pitch, yaw]
                    gen_des = su.cam.FisheyeUCM_Description(
                        width=W,
                        height=H,
                        intrinsics=dict(
                            gamma=f,
                            xi=xi,
                        ),
                        extrinsic_rot=Rot,
                    )

                    max_afov = gen_des.get_va_vector()[-1]
                    if max_afov < np.radians(40): # Half-afov
                        raise su.BigCropException(message="Small fov", img_size=0, crop_size=0)
                    
                    if max_afov > np.radians(100): # Half-afov
                        raise su.BigCropException(message="Big fov", img_size=0, crop_size=0)

                    delta = (max_afov - np.radians(40)) / np.radians(30) # After 140 degrees the distortion can be maximum
                    if np.abs(gen_des.a_) > delta:
                        raise su.BigCropException(message="Small fov", img_size=0, crop_size=0)

                    fin_img = su.mappings.map_img(img, des_list=[des,gen_des])[0]

                    filename = parser.compose(g, unique_name=img_path.name)
                    path_img = img_dir / f"{filename}.png"
                    path_calib = calib_dir / f"{filename}.json"

                    cv.imwrite(str(path_img), fin_img)

                    with open(path_calib,'w') as f:
                        json.dump(gen_des.to_dict(),f,indent=3)
                    
                    break
                except su.BigCropException:
                    i += 1
                

if __name__ == "__main__":
    num_copies = 20
    data_dir = su.base_path / 'data' / 'final_sun360'
    img_dir   = data_dir / 'rgb_images'
    calib_dir = data_dir / 'calibration'
    main(
        calib_dir=calib_dir,
        img_dir=img_dir,
        num_copies=num_copies,
    )