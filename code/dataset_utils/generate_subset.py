import os
import shutil
from tqdm.auto import tqdm
from pathlib import Path

OUT_DIR = Path('data/gen_kitti360_beta')
OUT_IMG_DIR   = OUT_DIR / 'rgb_images'
OUT_CALIB_DIR = OUT_DIR / 'calibration'
OUT_VA_DIR = OUT_DIR / 'va_vec'

SRC_DIR = Path('data/gen_kitti360')
SRC_IMG_DIR   = SRC_DIR / 'rgb_images'
SRC_CALIB_DIR = SRC_DIR / 'calibration'
SRC_VA_DIR = SRC_DIR / 'va_vec'

SRC_SUBSET_PATH = SRC_DIR / 'subset_beta.txt'

def main():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_CALIB_DIR, exist_ok=True)
    os.makedirs(OUT_VA_DIR, exist_ok=True)

    with open(SRC_SUBSET_PATH) as file:
        img_names = [line.rstrip('\n') for line in file]

    for img_name in tqdm(img_names):
        src_img_path = SRC_IMG_DIR / img_name
        src_calib_path = SRC_CALIB_DIR / f"{img_name.split('.')[0]}.json"
        src_va_path = SRC_VA_DIR / f"{img_name.split('.')[0]}.npy"

        shutil.copy(src_img_path, OUT_IMG_DIR)
        shutil.copy(src_calib_path, OUT_CALIB_DIR)
        shutil.copy(src_va_path, OUT_VA_DIR)

if __name__ == "__main__":
    main()