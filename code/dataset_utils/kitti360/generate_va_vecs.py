import os
import json
import kitti360_utils as ku
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

OUT_DIR = Path("data/gen_kitti360")
OUT_IMG_DIR   = OUT_DIR / 'rgb_images'
OUT_CALIB_DIR = OUT_DIR / 'calibration' 
OUT_VA_DIR = OUT_DIR / "va_vec"

def gen_va_vectors():
    os.makedirs(OUT_VA_DIR, exist_ok=True)
    for calib in tqdm(OUT_CALIB_DIR.iterdir(), total=len(os.listdir(OUT_CALIB_DIR))):
        with open(calib) as f:
            config = json.load(f)
        filename = calib.name.split(".")[0]
        des = ku.cam.ImageDescription.from_des_dict(config)
        va_vec = des.get_va_vector()
        
        np.save(OUT_VA_DIR / f'{filename}.npy', va_vec)

def show_va_vectors():
    calib = OUT_CALIB_DIR / "06-04656_MVL.json"
    
    with open(calib) as f:
        config = json.load(f)

    des = ku.cam.ImageDescription.from_des_dict(config)
    va_vec = des.get_va_vector()

    print(va_vec[0])
    print(np.sum(np.isnan(va_vec)))

    print(np.sum(np.isnan(np.load(OUT_VA_DIR / "06-04656_MVL.npy"))))

def main():
    gen_va_vectors()

if __name__ == "__main__" :
    main()