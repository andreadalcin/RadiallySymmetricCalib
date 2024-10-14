from __future__ import annotations
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from typing import List, Tuple, Union, List

from projections.cameras import *
from projections.proj2d import *
from datatypes.basetypes import BaseImage

def plot(img):
    plt.figure(figsize=(18, 6), dpi=80)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

def intersect_masks(*args):
    assert len(args) > 1

    base = args[0]

    for mask in args[1:]:
        if base is None:
            base = mask
        elif mask is not None:
            base &= mask

    return base

##############################################################
#####                                                   ######
#####                  POINTS MAPPING                   ######
#####                                                   ######
##############################################################

def _map_points(points: List[np.ndarray], src_des:ImageDescription, dst_des:ImageDescription, force_va_computation:Optional[bool]=None):
    w_x,w_y,w_z,w_mask = src_des.image2world(coordinates=points, force_va_computation=force_va_computation)
    
    s_x,s_y,s_mask = dst_des.world2image(coordinates=[w_x,w_y,w_z])

    return s_x, s_y, intersect_masks(w_mask, s_mask)

def map_points(points: List[np.ndarray], des_list:List[ImageDescription] ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    des_list = list(filter(lambda x: x is not None, des_list))
    assert len(des_list) > 1

    base_points = points
    base_des = des_list[0]
    base_mask = None
    force_va_map = None

    for end_des in des_list[1:]:
        u, v, mask = _map_points(base_points, src_des= base_des, dst_des = end_des, force_va_computation = force_va_map)
        force_va_map = False
        base_points = [u, v]
        base_des = end_des
        base_mask = intersect_masks(base_mask, mask)

    return base_points[0], base_points[1], base_mask

##############################################################
#####                                                   ######
#####                  IMAGE MAPPING                    ######
#####                                                   ######
##############################################################

def _map_img(img: np.ndarray, des_list:List[ImageDescription], interpolation:int = cv.INTER_CUBIC) -> Tuple[np.ndarray, List[ImageDescription]]:
    des_list = list(filter(lambda x: x is not None, des_list))
    assert len(des_list) > 1

    last_des = des_list[-1]

    H = last_des.height
    W = last_des.width
    u,v = np.meshgrid(np.arange(0,int(W)),np.arange(0,int(H)),indexing="xy")

    map_x, map_y, mask = map_points([u,v], des_list=des_list[::-1])

    dest_img:np.ndarray = cv.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), interpolation, borderMode=cv.BORDER_WRAP)
    
    if mask is not None:
        dest_img = cv.bitwise_and(dest_img,dest_img,mask = mask.astype(np.uint8))

    if len(img.shape) > len(dest_img.shape):
        dest_img = dest_img[...,None]
    return dest_img, des_list

def map_img(img: Union[BaseImage, np.ndarray], des_list:List[ImageDescription], interpolation:int = cv.INTER_CUBIC)-> Tuple[Union[BaseImage, np.ndarray], List[ImageDescription]]:
    if isinstance(img, BaseImage):
        dest_img, des_list = _map_img(img=img.img, des_list=[img.img_des] + des_list, interpolation=interpolation)
        return BaseImage(path=img.path, db_index=img.db_index, img_des=des_list[-1], img=dest_img), des_list    

    return _map_img(img=img, des_list=des_list)


def circular_mask(img: np.ndarray) -> np.ndarray:
    size = img.shape[:2]
    max_r2 = (min(size)/2) ** 2
    i = np.indices(size)
    r2 = (i[0,...]-size[0]/2)**2 + (i[1,...]-size[1]/2)**2
    img[r2 > max_r2, :] =0
    return img

def map_img_2d(img: np.ndarray, des:Proj2D, inverse=False) -> np.ndarray:

    if inverse:
        H, W = des.s_height, des.s_width
    else:
        H, W = des.d_height, des.d_width
    u,v = np.meshgrid(np.arange(0,int(W)),np.arange(0,int(H)),indexing="xy")

    map_x, map_y, mask = des.project([u,v], inverse= not inverse)
    dest_img = cv.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv.INTER_CUBIC, borderMode=cv.BORDER_WRAP)
    
    if mask is not None:
        dest_img = cv.bitwise_and(dest_img,dest_img,mask = mask.astype(np.uint8))
    return dest_img

def reproject_kps(kps, des_list:List[ImageDescription]) -> Tuple[List[cv.KeyPoint], List[int]]:
        """ Given a set of keypoints and corresponding descriptors taken from an image of type src_des,
         this method converts them in the dest_des type. 
        """
        
        kp_x = np.array([k.pt[0] for k in kps])
        kp_y = np.array([k.pt[1] for k in kps])
        n_kp_x, n_kp_y, mask = map_points([kp_y, kp_x], des_list = des_list)
        
        bad_kps = [] if mask is None else np.where(~mask)[0].tolist()

        n_kps = []
        for i,kp in enumerate(kps):
            if i not in bad_kps:
                n_kps.append(cv.KeyPoint(x=n_kp_x[i], y= n_kp_y[i], size=kp.size, angle=kp.angle, response=kp.response))

        return n_kps, bad_kps

##############################################################
#####                                                   ######
#####                   TEST MAPPING                    ######
#####                                                   ######
##############################################################

def __main_fishEUCM():
    img = cv.cvtColor(cv.imread("./data/matching/equirectangular/white_chapel_morning/Image0013.png"),cv.COLOR_BGR2RGB)
    img_des = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    out_des = FisheyeEUCM_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=dict(a=0.571, b=1.18, afov = 186), 
        extrinsic_rot=[0,0,0],
        compute_va_map=True)
    out_des2 = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    
    out_img,_ = map_img(img, des_list=[img_des,out_des,out_des2])
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(img)
    axarr[1].imshow(out_img)
    if out_des.get_last_va_map() is not None:
        axarr[2].imshow(out_des.get_last_va_map())
    print(out_des.get_va_vector())
    plt.show()

def __main_fishEUCM_focal():
    img = cv.cvtColor(cv.imread("./data/matching/equirectangular/white_chapel_morning/Image0013.png"),cv.COLOR_BGR2RGB)
    img_des = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    out_des = FisheyeEUCM_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=dict(a=0.571, b=2.18, f = 800), 
        extrinsic_rot=[0,0,0],
        compute_va_map=True)
    out_des2 = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    
    out_img1,_ = map_img(img, des_list=[img_des,out_des])
    out_img2,_ = map_img(img, des_list=[img_des,out_des,out_des2])
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(img)
    axarr[1].imshow(out_img1)
    axarr[2].imshow(out_img2)
    plt.show()

def __main_fishDS_to_persp():
    img = cv.cvtColor(cv.imread("/mnt/d/Documenti/GitHub/thesis/data/gen_woodscape/rgb_images/00-00000_FV.png"),cv.COLOR_BGR2RGB)
    img_des = ImageDescription.from_des_dict({
        "type": "FisheyeDS",
        "width": 790,
        "height": 790,
        "intrinsics": {
            "f": 304.4297689797567,
            "a": 0.5225509182876455,
            "xi": -0.1671849837128908
        }
        })

    afov = np.degrees(img_des.get_va_vector()[-1]) * 2
    print(afov)
    out_des = Perspective_Description(
        width=300,#img.shape[1],
        height=300,#img.shape[0],
        intrinsics=dict(afov=afov),
        
        )
    
    out_img1,_ = map_img(img, des_list=[img_des,out_des])
    print(out_img1.shape)
    
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img)
    axarr[1].imshow(out_img1)
    plt.show()

def __main_fishUCM():
    img = cv.cvtColor(cv.imread("./data/matching/equirectangular/white_chapel_morning/Image0013.png"),cv.COLOR_BGR2RGB)
    img_des = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    out_des = FisheyeUCM_Description(
        width=img.shape[0], 
        height=img.shape[0],
        intrinsics=dict(a=0.571, f = 800), 
        extrinsic_rot=[0,0,0],
        compute_va_map=True)
    out_des2 = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    
    out_img1,_ = map_img(img, des_list=[img_des,out_des])
    out_img2,_ = map_img(img, des_list=[img_des,out_des,out_des2])
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(img)
    axarr[1].imshow(out_img1)
    axarr[2].imshow(out_img2)
    plt.show()

def __main_fishUCM_to_persp():

    img = cv.cvtColor(cv.imread("/mnt/d/Documenti/GitHub/thesis/data/gen_woodscape/rgb_images/03-01654_FV.png"),cv.COLOR_BGR2RGB)
    img_des = FisheyeUCM_Description(
        width=img.shape[1], 
        height=img.shape[0],
        intrinsics=dict(xi=1.2, gamma = 100 *  img.shape[0] / 299), 
        extrinsic_rot=[0,0,0],)

    print(np.degrees(img_des.get_va_vector()[-1]))
    out_des = Perspective_Description(
        width=img.shape[1],
        height=img.shape[0],
        intrinsics=dict(f=100 *  img.shape[0] / 299)
        
        )
    
    out_img1,_ = map_img(img, des_list=[img_des,out_des])
    print(out_img1.shape)
    
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img)
    axarr[1].imshow(out_img1)
    plt.show()

def __main_fishPOLY():
    img = cv.cvtColor(cv.imread("./data/matching/equirectangular/white_chapel_morning/Image0013.png"),cv.COLOR_BGR2RGB)
    img_des = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    out_des = FisheyePOLY_Description(
        width=200, 
        height=200,
        intrinsics=dict(distortion_coeffs=[409.749,-31.988,48.275,-7.201]), 
        extrinsic_rot=[0,0,0])
    out_des2 = Equirectangular_Description(width=600, height=300)
    
    out_img1,_ = map_img(img, des_list=[img_des,out_des])
    out_img2,_ = map_img(img, des_list=[img_des,out_des,out_des2])
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(img)
    axarr[1].imshow(out_img1)
    axarr[2].imshow(out_img2)
    plt.show()

def __main_fishDS():
    img = cv.cvtColor(cv.imread("./data/matching/equirectangular/white_chapel_afternoon/Image0013.png"),cv.COLOR_BGR2RGB)
    img_des = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    out_des = FisheyeDS_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=dict(a=0.57, xi=-0.27, f = 364), 
        extrinsic_rot=[0,0,0],
        compute_va_map=True)
    out_des2 = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    
    
    out_img1,_ = map_img(img, des_list=[img_des,out_des])
    out_img2,_ = map_img(img, des_list=[img_des,out_des,out_des2])
    f, axarr = plt.subplots(1,4)
    axarr[0].imshow(img)
    axarr[1].imshow(out_img1)
    axarr[2].imshow(out_img2)
    axarr[3].imshow(out_des.get_last_va_map())

    
    print(out_des.get_last_va_map()[:out_des.height//2+1, out_des.width//2 ])
    print(out_des.get_va_vector())

    plt.show()

def __main_fishDS_resize():
    img = cv.cvtColor(cv.imread("./data/matching/equirectangular/white_chapel_afternoon/Image0013.png"),cv.COLOR_BGR2RGB)
    img_des = Equirectangular_Description(width=img.shape[1], height=img.shape[0])

    resize_dim = 700
    scale = resize_dim/img.shape[1]
    out_des = FisheyeDS_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=dict(a=0.57, xi=-0.27, f = 364), 
        extrinsic_rot=[0,0,0],
        compute_va_map=True)
    out_des2 = FisheyeDS_Description(
        width=resize_dim, 
        height=resize_dim,
        intrinsics=dict(a=0.57, xi=-0.27, f = out_des.f * scale ), 
        extrinsic_rot=[0,0,0],
        compute_va_map=True)
    
    
    out_img1,_ = map_img(img, des_list=[img_des,out_des])
    out_img2,_ = map_img(img, des_list=[img_des,out_des2])
    f, axarr = plt.subplots(1,5)
    axarr[0].imshow(img)
    axarr[1].imshow(out_img1)
    axarr[2].imshow(out_img2)
    axarr[3].imshow(out_des.get_last_va_map())
    axarr[4].imshow(out_des2.get_last_va_map())

    
    print(out_des.get_last_va_map()[:out_des.height//2+1, out_des.width//2 ])
    print(out_des.get_va_vector())

    plt.show()

def __main_fishDS_to_polar():
    img = cv.cvtColor(cv.imread("./data/matching/equirectangular/white_chapel_afternoon/Image0013.png"),cv.COLOR_BGR2RGB)
    img_des = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    out_des = FisheyeDS_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=dict(a=0.57, xi=-0.27, f = 364), 
        extrinsic_rot=[0,0,0])    
    
    out_img,_ = map_img(img, des_list=[img_des, out_des])
    f, axarr = plt.subplots(1,4)
    axarr[0].imshow(img)
    axarr[1].imshow(out_img)

    c2p = Cart2Polar(src_size=out_img.shape[:2], dst_size=img.shape[:2])
    img_pol = map_img_2d(img=out_img, des=c2p, inverse=False)
    axarr[2].imshow(img_pol)
    axarr[3].imshow(map_img_2d(img=img_pol, des=c2p, inverse=True))
    plt.show()

def __main_Persp():
    img = cv.cvtColor(cv.imread("./data/distortion/equirectangular_angles.png"),cv.COLOR_BGR2RGB)
    img_des = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    out_des = Perspective_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=dict(afov = 150), 
        extrinsic_rot=[0,0,0],
        compute_va_map=True)
    out_des2 = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    
    out_img1,_ = map_img(img, des_list=[img_des,out_des])
    out_img2,_ = map_img(img, des_list=[img_des,out_des,out_des2])
    f, axarr = plt.subplots(1,4)
    axarr[0].imshow(img)
    axarr[1].imshow(out_img1)
    axarr[2].imshow(out_img2)
    axarr[3].imshow(out_des.get_last_va_map())
    print(out_des.get_va_vector())
    plt.show()

if __name__ == "__main__":
    __main_fishDS_resize()
