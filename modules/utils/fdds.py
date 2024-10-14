from typing import List, Tuple, Optional
import cv2 as cv
import numpy as np
from d2net.d2net_front import D2FrontEnd
from datatypes.basetypes import BaseImage, RichInfo
from projections.cameras import ImageDescription, Equirectangular_Description
from sphorb import compute_sphorb as cs

from abc import ABC, abstractmethod

from superpoint.superpoint import MYJET, WEIGHTS_DIR, SuperPointFrontend

class FDD(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def detect_and_describe(self, img: np.ndarray) -> Tuple[List[cv.KeyPoint], np.ndarray]:
        """Performs keypoints deteciton and description using this specific algorithm on the input image.

        Args:
            img (np.ndarray): input image.

        Returns:
            Tuple[List[cv.KeyPoint], np.ndarray]: List of keypoinys and array of descriptors.
        """

    def rich_detect_and_describe(self, img: BaseImage) -> RichInfo:
        kps, des = self.detect_and_describe(img=img.img)
        return RichInfo(kps=kps, kps_des=des, width=img.img_des.width)

    def input_des(self) -> Optional[ImageDescription]:
        return None

    @abstractmethod
    def matcher_norm(self) -> int:
        """Return the norm type most suitable for the given type of descriptors. Usually Hamming for int, L2 for floats."""

class ORB(FDD):

    def __init__(self, name="ORB"):
        super().__init__(name)
        self.orb = cv.ORB_create()

    def detect_and_describe(self, img: np.ndarray) -> Tuple[List[cv.KeyPoint], np.ndarray]:
        # find the keypoints with ORB
        kp = self.orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = self.orb.compute(img, kp)
        return list(kp),des

    def matcher_norm(self) -> int:
        return cv.NORM_HAMMING

class SPHORB(FDD):

    def __init__(self, name="SPHORB"):
        super().__init__(name)
        self._input_des = Equirectangular_Description(
            width=cs.SPHORB_INPUT_SHAPE[1], 
            height=cs.SPHORB_INPUT_SHAPE[0])
    
    def detect_and_describe(self, img: np.ndarray) -> Tuple[List[cv.KeyPoint], np.ndarray]:
        assert img.shape[:2] == self.input_size()
        kp, des = cs.compute_sphorb(img)
        return kp, des

    def input_des(self) -> Optional[ImageDescription]:
        return self._input_des
            
    def matcher_norm(self) -> int:
        return cv.NORM_HAMMING

class SuperPoint(FDD):

    def __init__(self, name="SuperPoint", nms_dist = 4, conf_thresh=0.015, cuda=True):
        super().__init__(name)
        self.net = SuperPointFrontend(weights_path=WEIGHTS_DIR,
                          nms_dist=nms_dist,
                          conf_thresh=conf_thresh,
                          cuda=cuda)
    
    def detect_and_describe(self, img: np.ndarray) -> Tuple[List[cv.KeyPoint], np.ndarray]:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255.
        pts, des, heatmap = self.net.run(img.astype(np.float32))
        kp = pts_to_cv_kps(pts)
        return kp, des.T

    def matcher_norm(self) -> int:
        return cv.NORM_L2

class D2Net(FDD):

    def __init__(self, name="D2Net", multiscale:bool=False, preprocessing:str='caffe'):
        super().__init__(name)
        self.net = D2FrontEnd(multiscale=multiscale, preprocessing=preprocessing)
    
    def detect_and_describe(self, img: np.ndarray) -> Tuple[List[cv.KeyPoint], np.ndarray]:
        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255.
        pts, scores, des = self.net.run(img)


        kp = pts_to_cv_kps(pts.T)
        return kp, des

    def matcher_norm(self) -> int:
        return cv.NORM_L2

def pts_to_cv_kps(pts:np.ndarray) -> List[cv.KeyPoint]:
    """Converts a numpy array of keypoints to one of opencv keypoints.

    Args:
        pts (np.ndarray): array of shape (3, m) where m is the number of keypoints.

    Returns:
        List[cv.KeyPoint]: List of len m of keypoints.
    """
    return [cv.KeyPoint(x=pts[0,i], y=pts[1,i], size=pts[2,i]) for i in range(pts.shape[1])]

def __main():
    sup = SuperPoint()
    img  = cv.imread("data/perspective/Como.jpg")
    kp, des =  sup.detect_and_describe(img)
    print(len(kp))
    print(des.shape)

if __name__ == "__main__":
    __main()
