import cv2 as cv
import numpy as np
import os
import sys
from pathlib import Path
import json
from projections.cameras import *
from projections.proj2d import *
from tqdm.auto import tqdm
import glob
from woodscape_util import *

sys.path.append("modules")

def nothing(x):
    pass

def show_image_fish(img, mask, base_des):
    def_f = 333
    def_a_ = 0
    def_xi_ = -0.5
    def_x = 0
    def_y = 0
    def_z = 0
    fish_des = FisheyeDS_Description(
        height=1000,
        width=1000,
        intrinsics=dict(
            f=def_f,
            a=def_a_,
            xi=def_xi_),
        extrinsic_rot=(def_x,def_y,def_z),
        compute_va_map=True
    )
    
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, height= fish_des.height, width= fish_des.width)

    # cv.namedWindow(WINDOW_NAME_VA, cv.WINDOW_NORMAL)
    # cv.resizeWindow(WINDOW_NAME_VA, height= fish_des.height, width= fish_des.width)

    cv.createTrackbar("f", WINDOW_NAME, def_f , 1400, nothing)
    cv.createTrackbar("a_ * 100", WINDOW_NAME, int(def_a_ * 100), 100, nothing)
    cv.createTrackbar("xi_ * 100 +50 [-0.5:0.5]", WINDOW_NAME, int(def_xi_ * 100 + 50), 100, nothing)
    cv.createTrackbar("x_angle", WINDOW_NAME, def_x, 360, nothing)
    cv.createTrackbar("y_angle", WINDOW_NAME, def_y, 360, nothing)
    cv.createTrackbar("z_angle", WINDOW_NAME, def_z, 360, nothing)

    while True:
        if True:
            
            fish_des.f = cv.getTrackbarPos("f", WINDOW_NAME)
            fish_des.a_ = cv.getTrackbarPos("a_ * 100", WINDOW_NAME) /100
            fish_des.xi_ = (cv.getTrackbarPos("xi_ * 100 +50 [-0.5:0.5]", WINDOW_NAME) - 50)/100
            x_angle = cv.getTrackbarPos("x_angle", WINDOW_NAME)
            y_angle = cv.getTrackbarPos("y_angle", WINDOW_NAME)
            z_angle = cv.getTrackbarPos("z_angle", WINDOW_NAME)
            fish_des.extrinsic_rot = (x_angle, y_angle, z_angle)

            frame = project_and_mask_no_crop(img, mask, [base_des, fish_des], raise_exception=False)

            va_map = (fish_des.get_last_va_map() / np.pi * 180).astype(np.uint8)
            

            cv.imshow(WINDOW_NAME, frame)
            cv.imshow(WINDOW_NAME_VA, cv.cvtColor(va_map,cv.COLOR_GRAY2BGR))
            if cv.waitKey(1) & 0xFF == 27:
                cv.destroyAllWindows()
                break

def main_gen_mask_assignment():
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

    res = dict()

    for path in IMG_PATH.iterdir():
        idx_side, idx, side = parse_img_path(path.name)

        for e in d[side]:
            if int(e['idx']) >= idx:
                break

        res[idx_side] = f"{side}_{e['mask']}"

    with open(MASKS_ASSIGN_PATH,"w") as f:
        json.dump(res, f)


# generate
def main():
    assert IMG_PATH.exists()
    assert CALIB_PATH.exists()

    mask_done = []

    for path in IMG_PATH.iterdir():
        print(path)
        idx_side, idx, side = parse_img_path(path.name)

        cam_mask = MASK_ASSIGNMENT[idx_side]
        if cam_mask in mask_done:
            continue

        mask_done.append(cam_mask)
        mask = CAMERA_MASKS[cam_mask]
        print(cam_mask)

        
        fisheye_des = read_description_from_json(CALIB_PATH / f"{idx_side}.json")
        
        # load example image and re-project it to a central cylindrical projection
        fisheye_image = cv.imread(str(path))

        
        #fisheye_image_masked = cv.bitwise_and(fisheye_image,fisheye_image,mask = mask.astype(np.uint8))
        show_image_fish(fisheye_image, mask, fisheye_des)


if __name__ == "__main__":
    main()