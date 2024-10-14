from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import List
from dataclasses import InitVar, dataclass, field
from projections.cameras import ImageDescription

@dataclass
class BaseImage():
    """Wrapper for an image adding information regarding its location, name, description. 
    """
    path : str
    db_index : int
    img_des : ImageDescription
    img : np.ndarray = None

    # Needed for a better visualization
    img_rot90 : np.ndarray = field(init=False)
    
    def __post_init__(self) -> None:
        self.img_rot90 = cv.rotate(self.img, cv.ROTATE_90_COUNTERCLOCKWISE)

    def load(self):
        self.img = cv.imread(self.path)
        self.img_rot90 = cv.rotate(self.img, cv.ROTATE_90_COUNTERCLOCKWISE)
        self.img_des.width = self.img.shape[1]
        self.img_des.height = self.img.shape[0]

@dataclass
class RichInfo():
    """Additional information regarding a Base image retrieved after running a feature detection and description algorithm on it.
    """
    kps : List[cv.KeyPoint]
    kps_des : np.ndarray
    width : InitVar[int]

    # Needed for a better visualization
    kps_rot90 : List[cv.KeyPoint] = field(init=False)

    def __post_init__(self, width) -> None:
        self.__rotate_kps_ccw90(width)

    def __rotate_kps_ccw90(self, width) -> None:
        self.kps_rot90 = [ cv.KeyPoint(x=kp.pt[1], y=width-kp.pt[0], size=kp.size, angle=kp.angle, response=kp.response) for kp in self.kps ]
    

def __main():
    pass


if __name__ == "__main__":
    __main()