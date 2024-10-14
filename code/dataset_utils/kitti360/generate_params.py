import numpy as np
from pathlib import Path
import sys
import json
import cv2 as cv

dataset_utils = Path(__file__).parent / "../"
sys.path.append(str(dataset_utils))
import kitti360_utils as ku
import os
from tqdm.auto import tqdm

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
                afov=220,
                a=_max_a,
                xi= _max_xi)

# GEN_PARAMS_DS = [
#     ku.GenParams(
#         camera_name="image_02",
#         params_by_rot=[
#             ku.RotParams(
#                 extrinsic_rot=[0,21,28],
#                 max_rot_delta = [0,5,0],
#                 min_rot_delta = [0,-5,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max),
#             ku.RotParams(
#                 extrinsic_rot=[0,17,0],
#                 max_rot_delta = [0,20,0],
#                 min_rot_delta = [0,-15,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max),
#             ku.RotParams(
#                 extrinsic_rot=[0,23,324],
#                 max_rot_delta = [0,5,0],
#                 min_rot_delta = [0,-5,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max)]
#     ),
#     ku.GenParams(
#         camera_name="image_03",
#         params_by_rot=[
#             ku.RotParams(
#                 extrinsic_rot=[0,21,28],
#                 max_rot_delta = [0,5,0],
#                 min_rot_delta = [0,-5,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max),
#             ku.RotParams(
#                 extrinsic_rot=[0,17,0],
#                 max_rot_delta = [0,20,0],
#                 min_rot_delta = [0,-15,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max),
#             ku.RotParams(
#                 extrinsic_rot=[0,23,324],
#                 max_rot_delta = [0,5,0],
#                 min_rot_delta = [0,-5,0],
#                 min_intrinsics=_base_min,
#                 max_intrinsics=_base_max)]
#     ),
    
# ]

GEN_PARAMS_DS = [
    ku.GenParams(
        camera_name="image_02",
        params_by_rot=[
            ku.RotParams(
                extrinsic_rot=[0,0,28],
                max_rot_delta = [0,5,0],
                min_rot_delta = [0,0,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max),
            ku.RotParams(
                extrinsic_rot=[0,0,0],
                max_rot_delta = [0,20,0],
                min_rot_delta = [0,0,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max),
            ku.RotParams(
                extrinsic_rot=[0,0,324],
                max_rot_delta = [0,5,0],
                min_rot_delta = [0,0,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max)]
    ),
    ku.GenParams(
        camera_name="image_03",
        params_by_rot=[
            ku.RotParams(
                extrinsic_rot=[0,0,28],
                max_rot_delta = [0,5,0],
                min_rot_delta = [0,0,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max),
            ku.RotParams(
                extrinsic_rot=[0,0,0],
                max_rot_delta = [0,20,0],
                min_rot_delta = [0,0,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max),
            ku.RotParams(
                extrinsic_rot=[0,0,324],
                max_rot_delta = [0,5,0],
                min_rot_delta = [0,0,0],
                min_intrinsics=_base_min,
                max_intrinsics=_base_max)]
    ),
    
]


def main(img_dir:Path, calib_dir:Path,  num_copies:int=1, max_imgs=None):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    height=400
    width=400

    parser = ku.KittiGenNameParser()

    max_iters = 1000
    va_vec_window = 20
    va_th = np.radians(100)
    todo_afovs = []
    gen_va_vecs = []

    for sequence in ku.IMG_DIR.iterdir():

        num_cameras = len(os.listdir(sequence))
        for camera in sequence.iterdir():
            camera_idx = camera.name.split("_")[-1]
            camera_data = camera / "data_rgb"
            mask_name = camera.name
            mask_file = ku.MASK_DIR / f'{camera.name}.png'
            mask = cv.imread(str(mask_file),cv.IMREAD_GRAYSCALE)
            intrinsic_file = ku.CALIB_DIR / f'{camera.name}.yaml'
            des = ku.load_descriptor(intrinsic_file)
            gen_params = ku.get_gen_params_from_mask(params_list=GEN_PARAMS_DS, mask_name=mask_name)

            img_names = os.listdir(camera_data)
            print(len(img_names))

            # filter image duplicates
            def is_accepted(x):
                img_idx = x.split('.')[0]
                if int(img_idx) < 100 or int(img_idx) > 13580:
                    return False
                return True

            img_names = list(filter(is_accepted, img_names))

            if max_imgs is not None:
                img_names = np.random.default_rng().choice(img_names, max_imgs//num_cameras, replace=False)

            for img_name in tqdm(img_names):
                img_path = camera_data / '0000000560.png'
                img = cv.imread(str(img_path))
                
                rot = ku.random_rotation(gen_params, alpha=0, beta=0.5)

                gen_intr = gen_params.params_by_rot[0]
                max_afov = ku.compute_max_afov(mask, des, rot, max_afov=gen_intr.max_intrinsics['afov'])

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
                        afov = ku.get_random_intrinsic_param(gen_intr, "afov")

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
                            a = ku.get_random_intrinsic_param(gen_intr, "a")
                            xi = ku.get_random_intrinsic_param(gen_intr, "xi")

                            alpha = np.random.default_rng().uniform()

                            # Avoid big distortions on small afovs
                            xi = (1 if xi >= 0 else -1) * (np.abs(xi) * (max_delta - alpha * min_delta) + alpha * min_delta)
                            a = a * (max_delta - (1- alpha)* min_delta) + (1- alpha)* min_delta
                            # if np.abs(xi) > max_delta/2 or a > max_delta: continue

                            # Avoid pinchushion distortion
                            #if xi< 0 and a < 0.4: continue

                            gen_des = ku.cam.FisheyeDS_Description(
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
                                dist = ku.check_distance(gen_va_vecs, va_vec)
                                if dist < va_th * va_delta:
                                    raise ku.BigCropException(message="Similar to existing vectors", img_size=0, crop_size=0)
                                
                            est_afov = np.degrees(va_vec[-1]*2)
                            if not np.allclose(afov,est_afov):
                                print(f'{afov}, , {est_afov}')
                                print(xi, a, gen_des)
                                raise ku.BigCropException(message="Bad va_vec generation", img_size=0, crop_size=0)
                            gen_va_vecs.append(va_vec)

                            if len(gen_va_vecs) >= va_vec_window:
                                gen_va_vecs = []
                                
                            print(afov)

                            fin_img = ku.mappings.map_img(img, [des, gen_des])[0]

                            filename = parser.compose(g, cam_name=camera_idx, idx=img_name.split('.')[0])

                            path_img = img_dir / f"{filename}.png"
                            path_calib = calib_dir / f"{filename}.json"

                            cv.imwrite(str(path_img), fin_img)

                            with open(path_calib,'w') as f:
                                json.dump(gen_des.to_dict(),f,indent=3)
                            break
                        except ku.BigCropException:
                            i += 1
                        except ku.cam.CameraException:
                            continue

                    g +=1
                    if i == max_iters:
                        print("Failed", len(gen_va_vecs), afov, max_afov)
                        for va_vec in gen_va_vecs:
                            print(np.degrees(va_vec[-1])*2)

if __name__ == "__main__":
    num_copies = 20
    data_dir = ku.base_path / 'data' / 'random'
    img_dir   = data_dir / 'rgb_images'
    calib_dir = data_dir / 'calibration'
    max_imgs = 1
    main(
        calib_dir=calib_dir,
        img_dir=img_dir,
        num_copies=num_copies,
        max_imgs= max_imgs,
    )