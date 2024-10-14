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

MAX_AFOV = 190
MIN_AFOV = 80

def main(img_dir:Path, calib_dir:Path, num_copies:int=1):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    parser = su.SunGenNameParser()

    height=400
    width=400

    max_iters = 1000
    va_vec_window = 20
    va_th = np.radians(30)
    gen_va_vecs = []

    for img_path in tqdm(su.IMG_DIR.iterdir(),total= len(os.listdir(su.IMG_DIR))):
            
        img = cv.imread(str(img_path))
        des = su.cam.Equirectangular_Description(
            width=img.shape[1],
            height=img.shape[0]
        )


        g = 0
        while g < num_copies:
            # 3. Rotation of the sphere
            rot = []
            rot.append(((np.random.default_rng().random() - 0.5) * 2) * 10) # roll
            rot.append(((np.random.default_rng().random() - 0.5) * 2) * 15) # pitch
            rot.append(((np.random.default_rng().random() - 0.5) * 2) * 180) # yaw
            
            afov = np.random.default_rng().uniform(MIN_AFOV, MAX_AFOV)
          
            max_delta = min((afov - 80) / 80, 1) # range that increases linearly from 0 - 1 ranging from 80 to 160 degrees
            min_delta = max((afov - 120) / 80, 0) # range that increases linearly from 0 - 1 ranging from 120 to 200 degrees
            
            va_delta = (afov - MIN_AFOV) / (MAX_AFOV - MIN_AFOV) # range that increases linearly from 0 - 1 

            i = 0
            while i<max_iters:
                try:
                    a = np.random.default_rng().uniform(0,1)
                    xi = np.random.default_rng().uniform(-1,1)

                    alpha = np.random.default_rng().uniform()

                    # Avoid big distortions on small afovs
                    xi = (1 if xi >= 0 else -1) * (np.abs(xi) * (max_delta - alpha * min_delta) + alpha * min_delta)
                    a = a * (max_delta - (1- alpha)* min_delta) + (1- alpha)* min_delta
                    # if np.abs(xi) > max_delta/2 or a > max_delta: continue

                    # Avoid pinchushion distortion
                    #if xi< 0 and a < 0.4: continue

                    gen_des = su.cam.FisheyeDS_Description(
                        width=width,
                        height=height,
                        intrinsics=dict(afov=afov, xi=xi, a=a),
                        extrinsic_rot=rot,
                    )

                    va_vec = gen_des.get_va_vector()

                    persp_des = gen_des.copy()
                    persp_des.a_ = 0
                    persp_des.xi_ = 0
                    persp_va_vec = persp_des.get_va_vector()
                    if np.alltrue(va_vec <= persp_va_vec):
                        # print(f"Pinchushion! {gen_des}")
                        continue

                    if not np.alltrue(va_vec >= persp_va_vec) and gen_des.f < 35:
                        # Strange type of distortion, remove focals < 35
                        continue

                    if len(gen_va_vecs) > 0:
                        dist = su.check_distance(gen_va_vecs, va_vec)
                        if dist < va_th * va_delta:
                            raise su.BigCropException(message="Similar to existing vectors", img_size=0, crop_size=0)
                        
                    est_afov = np.degrees(va_vec[-1]*2)
                    if not np.allclose(afov,est_afov):
                        print(f'{afov}, , {est_afov}')
                        print(xi, a, gen_des)
                        raise su.BigCropException(message="Bad va_vec generation", img_size=0, crop_size=0)
                    gen_va_vecs.append(va_vec)

                    if len(gen_va_vecs) >= va_vec_window:
                        gen_va_vecs = []

                    fin_img = su.mappings.map_img(img, [des, gen_des])[0]

                    filename = parser.compose(g, unique_name=img_path.name)

                    path_img = img_dir / f"{filename}.png"
                    path_calib = calib_dir / f"{filename}.json"

                    cv.imwrite(str(path_img), fin_img)

                    with open(path_calib,'w') as f:
                        json.dump(gen_des.to_dict(),f,indent=3)
                    break
                except su.BigCropException:
                    i += 1
                except su.cam.CameraException:
                    continue

            g +=1
            if i == max_iters:
                print("Failed", len(gen_va_vecs), afov)
                for va_vec in gen_va_vecs:
                    print(np.degrees(va_vec[-1])*2)
                

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