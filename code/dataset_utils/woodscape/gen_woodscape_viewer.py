import cv2 as cv
import sys
from pathlib import Path
import json
from projections.cameras import *
from projections.mappings import map_img
from matplotlib import pyplot as plt

sys.path.append("modules")

def get_description(data_folder_calib, seq_idx_side:str) -> ImageDescription:
    file = f"{seq_idx_side}.json"
    with open(data_folder_calib / file , "r") as f:
        des_dict = json.load(f)

    des = ImageDescription.from_des_dict(des_dict=des_dict)
    return des

# generate
def main():
    GEN_WOODSCAPE_PATH = Path("data/gen_woodscape")
    IMG_PATH = GEN_WOODSCAPE_PATH / "rgb_images"
    CALIB_PATH = GEN_WOODSCAPE_PATH / "calibration"

    assert IMG_PATH.exists()
    assert CALIB_PATH.exists()

    for path in IMG_PATH.iterdir():
        print(path)        
        fisheye_des = get_description(CALIB_PATH , path.name.split('.')[0])
        fisheye_des.extrinsic_rot = [0,0,0]
        
        # load example image and re-project it to a central cylindrical projection
        fisheye_image = cv.imread(str(path))

        out_des = Perspective_Description(
            height=fisheye_des.height,
            width=fisheye_des.width,
            intrinsics= dict(f=fisheye_des.f)
        )

        out_image,_ = map_img(fisheye_image, [fisheye_des, out_des])

        f, axarr = plt.subplots(2)
        axarr[0].imshow(cv.cvtColor(fisheye_image, cv.COLOR_BGR2RGB))
        axarr[1].imshow(cv.cvtColor(out_image, cv.COLOR_BGR2RGB))
        plt.show()

        exit()

if __name__ == "__main__":
    main()