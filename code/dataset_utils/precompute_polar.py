import os
import json
import imaging as im
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import cv2 as cv

def gen_polar_input(data_dir:Path, polar_width:int = 800):
    img_dir = data_dir / 'rgb_images'
    polar_img_dir = data_dir/ 'rgb_polar'

    os.makedirs(polar_img_dir)

    for img_path in tqdm(img_dir.iterdir(), total=len(os.listdir(img_dir))):
        img = cv.imread(str(img_path))

        c2p = im.mappings.Cart2Polar()
        c2p.set_sizes_from_src_w(img.shape[:2], d_width=polar_width)
        polar = im.mappings.map_img_2d(
            img=img,
            des=c2p,
        )

        polar_path = polar_img_dir / img_path.name
        cv.imwrite(str(polar_path), polar)


if __name__ == "__main__" :
    
    data_dir = im.base_path / 'data' / 'gen_woodscape_u_resized'

    gen_polar_input(
        data_dir = data_dir,
        polar_width = 800,
    )

