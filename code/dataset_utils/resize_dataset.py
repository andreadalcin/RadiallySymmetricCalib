import os
import json
import imaging as im
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import cv2 as cv

def gen_resized_dataset(img_dir:Path, calib_dir:Path, resized_data_dir:Path, width:int=1000, height:int=1000):
    resized_img_dir = resized_data_dir / "rgb_images"
    resized_calib_dir = resized_data_dir / "calibration"
    resized_va_vec_dir = resized_data_dir / "va_vec"

    os.makedirs(resized_img_dir)
    os.makedirs(resized_calib_dir)
    os.makedirs(resized_va_vec_dir)

    for img_path in tqdm(img_dir.iterdir(), total=len(os.listdir(img_dir))):
        filename = img_path.name.split('.')[0]
        img = cv.imread(str(img_path))

        calib_path = calib_dir / f'{filename}.json'
        with open(calib_path) as f:
            config = json.load(f)
        calib = im.cam.ImageDescription.from_des_dict(config)

        # TODO robust resize
        scale = float(width) / img.shape[1]
        resized_img = cv.resize(img, dsize=(width, height))
        resized_calib = calib
        resized_calib.width = width
        resized_calib.height = height
        resized_calib.f *= scale

        resized_va_vec = resized_calib.get_va_vector()

        cv.imwrite(str(resized_img_dir/img_path.name),resized_img)
        np.save(resized_va_vec_dir / f'{filename}.npy', resized_va_vec)
        with open(resized_calib_dir / calib_path.name,'w') as f:
            json.dump(resized_calib.to_dict(),f,indent=3)

if __name__ == "__main__" :
    data_dir = im.base_path / 'data' / 'gen_woodscape_fr'
    img_dir = data_dir / 'rgb_images'
    calib_dir = data_dir / 'calibration'
    
    resized_data_dir = im.base_path / 'data' / 'gen_woodscape_fr_resized'

    gen_resized_dataset(
        img_dir=img_dir,
        calib_dir=calib_dir,
        resized_data_dir=resized_data_dir,
        width=1000,
        height=1000,
    )
