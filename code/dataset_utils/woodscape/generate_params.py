import json
from dataclasses import dataclass, field
from typing import Tuple, List
import woodscape_util as wu
from tqdm.auto import tqdm
import os
import numpy as np
from pathlib import Path
import cv2 as cv
from scipy.spatial.transform import Rotation as R

np.random.seed(333)

_min_a = 0
_min_xi = -1
_max_a = 1
_max_xi = 1

_base_min =dict(
                afov=80,
                a=_min_a,
                xi=_min_xi)
_base_max =  dict(
                afov=190,
                a=_max_a,
                xi= _max_xi)

GEN_PARAMS_DS = [
    ## FV  ##
    wu.GenParams(
        camera_name="FV_MASK1",
        params_by_rot=[
            wu.RotParams(
                extrinsic_rot=[0,0,28],
                max_rot_delta = [0,5,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max),
            wu.RotParams(
                extrinsic_rot=[0,0,0],
                max_rot_delta = [0,30,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max),
            wu.RotParams(
                extrinsic_rot=[0,0,324],
                max_rot_delta = [0,5,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max)]
    ),
    ## RV  ##
    wu.GenParams(
        camera_name="RV_MASK1",
        params_by_rot=[
            wu.RotParams(
                extrinsic_rot=[0,0,25],
                max_rot_delta = [0,5,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max),
            wu.RotParams(
                extrinsic_rot=[0,0,0],
                max_rot_delta = [0,20,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max),
            wu.RotParams(
                extrinsic_rot=[0,0,329],
                max_rot_delta = [0,5,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max)]
    ),
    ]

def main(img_dir:Path, calib_dir:Path, num_copies:int=1, max_imgs=2000):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    allowed_masks = ['FV_MASK1', 'RV_MASK1']

    height=400
    width=400

    max_iters = 1000
    va_vec_window = 20
    va_th = np.radians(30)
    todo_afovs = []
    gen_va_vecs = []

    parser = wu.WoodscapeGenNameParser()
    camera_masks = wu.get_camera_masks()

    def is_accepted(x):
        mask_name = wu.MASK_ASSIGNMENT[x.split('.')[0]]
        for am in allowed_masks:
            if am == mask_name:
                return True
        return False

    imgs_list = list(filter(is_accepted, os.listdir(wu.IMG_DIR)))
    if max_imgs is not None:
        imgs_list = np.random.default_rng().choice(imgs_list, max_imgs, replace=False)

    for img_name in tqdm(imgs_list):
        img_path = wu.IMG_DIR / img_name

        curr_name = img_name.split('.')[0]
        
        mask = wu.get_mask(camera_masks, curr_name)
        intrinsic_file = wu.CALIB_DIR / f'{curr_name}.json'
        des = wu.load_descriptor(intrinsic_file)
        gen_params = wu.get_gen_params_from_mask(params_list=GEN_PARAMS_DS, mask_name=wu.MASK_ASSIGNMENT[curr_name])
        rot = wu.random_rotation(gen_params)

        gen_intr = gen_params.params_by_rot[0]
        max_afov = wu.compute_max_afov(mask, des, rot, max_afov=gen_intr.max_intrinsics['afov'])
        img = cv.imread(str(img_path))


        stored_afovs_idx = 0

        g = 0
        while g < num_copies:

            afov = None

            if g < num_copies / 2:
                while stored_afovs_idx < len(todo_afovs) and todo_afovs[stored_afovs_idx] > max_afov:
                    stored_afovs_idx += 1
            
                if stored_afovs_idx < len(todo_afovs):
                    afov = todo_afovs[stored_afovs_idx]
                    del todo_afovs[stored_afovs_idx]
                    
            if afov is None:
                afov = wu.get_random_intrinsic_param(gen_intr, "afov")

            if afov > max_afov:
                todo_afovs.append(afov)
                todo_afovs = sorted(todo_afovs, reverse=True)
                stored_afovs_idx += 1
                continue

            max_delta = min((afov - 80) / 80, 1) # range that increases linearly from 0 - 1 ranging from 80 to 160 degrees
            min_delta = max((afov - 120) / 80, 0) # range that increases linearly from 0 - 1 ranging from 120 to 200 degrees
            
            va_delta = (afov - gen_intr.min_intrinsics['afov']) / (gen_intr.max_intrinsics['afov'] - gen_intr.min_intrinsics['afov']) # range that increases linearly from 0 - 1 

            i = 0
            while i<max_iters:
                try:
                    a = wu.get_random_intrinsic_param(gen_intr, "a")
                    xi = wu.get_random_intrinsic_param(gen_intr, "xi")

                    alpha = np.random.default_rng().uniform()

                    # Avoid big distortions on small afovs
                    xi = (1 if xi >= 0 else -1) * (np.abs(xi) * (max_delta - alpha * min_delta) + alpha * min_delta)
                    a = a * (max_delta - (1- alpha)* min_delta) + (1- alpha)* min_delta
                    # if np.abs(xi) > max_delta/2 or a > max_delta: continue

                    # Avoid pinchushion distortion
                    #if xi< 0 and a < 0.4: continue

                    gen_des = wu.cam.FisheyeDS_Description(
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
                        dist = wu.check_distance(gen_va_vecs, va_vec)
                        if dist < va_th * va_delta:
                            raise wu.BigCropException(message="Similar to existing vectors", img_size=0, crop_size=0)
                        
                    est_afov = np.degrees(va_vec[-1]*2)
                    if not np.allclose(afov,est_afov):
                        print(f'{afov}, , {est_afov}')
                        print(xi, a, gen_des)
                        raise wu.BigCropException(message="Bad va_vec generation", img_size=0, crop_size=0)
                    gen_va_vecs.append(va_vec)

                    if len(gen_va_vecs) >= va_vec_window:
                        gen_va_vecs = []

                    fin_img = wu.mappings.map_img(img, [des, gen_des])[0]

                    filename = parser.compose(g, unique_name=curr_name)

                    path_img = img_dir / f"{filename}.png"
                    path_calib = calib_dir / f"{filename}.json"

                    cv.imwrite(str(path_img), fin_img)

                    with open(path_calib,'w') as f:
                        json.dump(gen_des.to_dict(),f,indent=3)
                    break
                except wu.BigCropException:
                    i += 1
                except wu.cam.CameraException:
                    continue

            g +=1
            if i == max_iters:
                print("Failed", len(gen_va_vecs), afov, max_afov)
                for va_vec in gen_va_vecs:
                    print(np.degrees(va_vec[-1])*2)
    

if __name__ == "__main__":
    num_copies = 5
    data_dir = wu.base_path / 'data' / 'WS-T1'
    img_dir   = data_dir / 'rgb_images'
    calib_dir = data_dir / 'calibration'
    main(
        calib_dir=calib_dir,
        img_dir=img_dir,
        num_copies=num_copies,
        max_imgs=1000,
    )


# _base_min =dict(
#                 f=150,
#                 a=_min_a,
#                 xi=_min_xi)
# _base_max =  dict(
#                 f=1600,
#                 a=_max_a,
#                 xi= _max_xi)

# GEN_PARAMS_DS = [
#     ## FV  ##
#     wu.GenParams(
#         camera_name="FV_MASK1",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,21,28],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max),
#             wu.RotParams(
#                 extrinsic_rot=[0,17,0],
#                 max_rot_delta = [0,20,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max),
#             wu.RotParams(
#                 extrinsic_rot=[0,23,324],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max)]
#     ),
#     wu.GenParams(
#         camera_name="FV_MASK2",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,9,44],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=dict(
#                     f=260,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1200,
#                     a=_max_a,
#                     xi=_max_xi)),
#             wu.RotParams(
#                 extrinsic_rot=[0,14,0],
#                 max_rot_delta = [0,20,0],
#                 min_intrinsics=dict(
#                     f=150,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1000,
#                     a=_max_a,
#                     xi=_max_xi)),
#             wu.RotParams(
#                 extrinsic_rot=[0,4,326],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=dict(
#                     f=230,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1150,
#                     a=_max_a,
#                     xi=_max_xi)),]
#     ),
#     wu.GenParams(
#         camera_name="FV_MASK3",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,15,4],
#                 max_rot_delta = [0,20,40],
#                 min_rot_delta = [0,0,-40],
#                 min_intrinsics=dict(
#                     f=100,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=770,
#                     a=_max_a,
#                     xi=_max_xi))]
#     ),
#     wu.GenParams(
#         camera_name="FV_MASK4",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,11,0],
#                 min_intrinsics=dict(
#                     f=150,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=750,
#                     a=_max_a,
#                     xi=_max_xi))]
#     ),
#     wu.GenParams(
#         camera_name="FV_MASK5",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,8,25],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=dict(
#                     f=130,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=940,
#                     a=_max_a,
#                     xi=_max_xi)),
#             wu.RotParams(
#                 extrinsic_rot=[0,13,0],
#                 max_rot_delta = [0,20,0],
#                 min_intrinsics=dict(
#                     f=130,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1050,
#                     a=_max_a,
#                     xi=_max_xi)),
#             wu.RotParams(
#                 extrinsic_rot=[0,8,333],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=dict(
#                     f=200,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1050,
#                     a=_max_a,
#                     xi=_max_xi)),]
#     ),
#     wu.GenParams(
#         camera_name="FV_MASK6",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,19,34],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=dict(
#                     f=190,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1040,
#                     a=_max_a,
#                     xi=_max_xi)),
#             wu.RotParams(
#                 extrinsic_rot=[0,22,0],
#                 max_rot_delta = [0,20,0],
#                 min_intrinsics=dict(
#                     f=130,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1120,
#                     a=_max_a,
#                     xi=_max_xi)),
#             wu.RotParams(
#                 extrinsic_rot=[0,19,338],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=dict(
#                     f=220,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1120,
#                     a=_max_a,
#                     xi=_max_xi)),]
#     ),

#     ## RV  ##

#     wu.GenParams(
#         camera_name="RV_MASK1",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,17,25],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max),
#             wu.RotParams(
#                 extrinsic_rot=[0,26,0],
#                 max_rot_delta = [0,10,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max),
#             wu.RotParams(
#                 extrinsic_rot=[0,16,329],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max)]
#     ),
#     wu.GenParams(
#         camera_name="RV_MASK2",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,17,0],
#                 max_rot_delta = [0,20,0],
#                 min_intrinsics=dict(
#                     f=240,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1250,
#                     a=_max_a,
#                     xi=_max_xi)),
#             wu.RotParams(
#                 extrinsic_rot=[0,24,43],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=dict(
#                     f=425,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1400,
#                     a=_max_a,
#                     xi=_max_xi)),
#             wu.RotParams(
#                 extrinsic_rot=[0,24,322],
#                 max_rot_delta = [0,5,0],
#                 min_intrinsics=dict(
#                     f=340,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1400,
#                     a=_max_a,
#                     xi=_max_xi)),]
#     ),
#     wu.GenParams(
#         camera_name="RV_MASK3",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,12,0],
#                 min_intrinsics=dict(
#                     f=100,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=750,
#                     a=_max_a,
#                     xi=_max_xi))]
#     ),

#     ## MVL ##

#     wu.GenParams(
#         camera_name="MVL_MASK1",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,42,22],
#                 min_intrinsics=dict(
#                     f=500,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=900,
#                     a=_max_a,
#                     xi=_max_xi)),
#             wu.RotParams(
#                 extrinsic_rot=[0,25,317],
#                 min_intrinsics=dict(
#                     f=300,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=800,
#                     a=_max_a,
#                     xi=_max_xi))]
#     ),
#     wu.GenParams(
#         camera_name="MVL_MASK2",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,45,22],
#                 min_intrinsics=dict(
#                     f=340,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1400,
#                     a=_max_a,
#                     xi=_max_xi)),
#             wu.RotParams(
#                 extrinsic_rot=[0,24,315],
#                 min_intrinsics=dict(
#                     f=150,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=950,
#                     a=_max_a,
#                     xi=_max_xi))]
#     ),

#     ## MVR ##

#     wu.GenParams(
#         camera_name="MVR_MASK1",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,24,43],
#                 min_intrinsics=dict(
#                     f=300,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=800,
#                     a=_max_a,
#                     xi=_max_xi)),
#             wu.RotParams(
#                 extrinsic_rot=[0,45,334],
#                 min_intrinsics=dict(
#                     f=580,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=1150,
#                     a=_max_a,
#                     xi=_max_xi))]
#     ),
#     wu.GenParams(
#         camera_name="MVR_MASK2",
#         params_by_rot=[
#             wu.RotParams(
#                 extrinsic_rot=[0,12,0],
#                 min_intrinsics=dict(
#                     f=100,
#                     a=_min_a,
#                     xi=_min_xi),
#                 max_intrinsics=dict(
#                     f=750,
#                     a=_max_a,
#                     xi=_max_xi))]
#     ),
# ]