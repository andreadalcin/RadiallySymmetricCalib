from __future__ import annotations
from typing import List, Optional, Tuple, ClassVar, Union
from dataclasses import dataclass, field, fields
import numpy as np
from abc import ABC, abstractmethod

#######################################
######        Abs classes       #######
#######################################

@dataclass
class Proj2D(ABC):
    s_width: int = field(init=False)
    s_height: int = field(init=False)

    d_width: int = field(init=False)
    d_height: int = field(init=False)

    def __init__(self, src_size=None, dst_size=None) -> None:
        if src_size is None or dst_size is None:
            return
        self.s_height, self.s_width = src_size
        self.d_height, self.d_width = dst_size
        self.__post_init__()

    def __post_init__(self):
        pass

    def project(self, coordinates: Tuple[np.ndarray, np.ndarray], inverse:bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if inverse:
            return self._inverse(coordinates=coordinates)
        return self._direct(coordinates=coordinates)

    @abstractmethod
    def _direct(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray,np.ndarray, np.ndarray]:
        """_summary_

        Args:
            coordinates (Tuple[np.ndarray, np.ndarray]): Tuple of arrays [X,Y], image coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of arrays [X,Y,mask], according to the following convension. 
            The frame origin is the top-left corner of the image, the X axis points to the right and the Y axis points downwards.
            The mask is used to define which points are visible in the image.
        """
        raise NotImplementedError( "Method not implemented" )

    @abstractmethod
    def _inverse(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """_summary_

        Args:
            coordinates (Tuple[np.ndarray, np.ndarray]): Tuple of arrays [X,Y], image coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of arrays [X,Y,mask], according to the following convension. 
            The frame origin is the top-left corner of the image, the X axis points to the right and the Y axis points downwards.
            The mask is used to define which points are visible in the image.
        """
        raise NotImplementedError( "Method not implemented" )

#######################################
######       Impl classes       #######
#######################################

class Cart2Polar(Proj2D):

    s_radius: int = field(init=False)

    def __post_init__(self):
        self.s_radius = min(self.s_height, self.s_width)//2

    def set_sizes_from_src_w(self, src_size, d_width):
        self.s_height, self.s_width = src_size
        self.d_height = self.s_height//2 + 1
        self.d_width  = d_width
        self.s_radius = min(self.s_height, self.s_width)//2
        
    def _direct(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        i_x, i_y = coordinates
        
        # Principal point
        c_x = self.s_width / 2
        c_y = self.s_height / 2

        # Remove pc
        s_x, s_y = i_x - c_x, -(i_y - c_y)

        # projection
        rho = np.sqrt(s_x**2 + s_y**2)
        theta = np.pi - np.arctan2(s_x, s_y) 

        rho_p = rho/ self.s_radius * self.d_height
        theta_p = theta / (2*np.pi) * self.d_width

        mask = rho <= self.s_radius

        return theta_p, rho_p, mask

    def _inverse(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # theta is not represented as angles but as pixels
        i_theta_p, i_rho_p = coordinates

        i_rho = i_rho_p / self.d_height * self.s_radius
        # angular resolution
        i_theta = i_theta_p / self.d_width * 2 * np.pi
        i_theta -= np.pi/2 # Move zero degrees in the standard spot

        # Projection
        s_x = np.cos(i_theta) * i_rho
        s_y = np.sin(i_theta) * i_rho
        
        # Principal point
        c_x = self.s_width / 2
        c_y = self.s_height / 2

        # Add pc
        i_x, i_y = s_x + c_x, -s_y + c_y

        mask = None

        return i_x, i_y, mask

class Cart2Polar_v2(Proj2D):

    s_radius: int = field(init=False)

    def __post_init__(self):
        self.s_radius = min(self.s_height, self.s_width) // 2 - 1 
        """ ex: H=W=1000 : radius = 499, offset = 499.5 - 499 = 0.5 """
        """ ex: H=W=1001 : radius = 499, offset = 500 - 499 = 1. """
        self.s_offset = (min(self.s_height, self.s_width) - 1) / 2 - self.s_radius 

    def set_sizes_from_src_w(self, src_size, d_width):
        self.s_height, self.s_width = src_size
        self.d_height = self.s_height//2
        self.d_width  = d_width
        self.s_radius = min(self.s_height, self.s_width) // 2 - 1 
        """ ex: H=W=1000 : radius = 499, offset = 499.5 - 499 = 0.5 """
        """ ex: H=W=1001 : radius = 499, offset = 500 - 499 = 1. """
        self.s_offset = (min(self.s_height, self.s_width) - 1) / 2 - self.s_radius 
        
    def _direct(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        i_x, i_y = coordinates
        
        # Principal point
        c_x = (self.s_width - 1) / 2
        c_y = (self.s_height - 1) / 2

        # Remove pc
        s_x, s_y = i_x - c_x, -(i_y - c_y)

        # projection
        rho = np.sqrt(s_x**2 + s_y**2)
        theta = np.pi - np.arctan2(s_x, s_y) 
        theta[theta == 2*np.pi] = 0


        rho_p = (rho ) / (self.s_radius + self.s_offset) * (self.d_height - .5) # max vaue of prho mapped to the last pixel:
            # when rho == max (top pixels), rho_p = self.d_height - 1
            # when rho == min (center pixels), rho_p = 0
        theta_p = theta / (2*np.pi) * self.d_width # 2pi should not be included since 0 is already in 

        mask = rho <= self.s_radius

        return theta_p, rho_p, mask

    def _inverse(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # theta is not represented as angles but as pixels
        i_theta_p, i_rho_p = coordinates

        i_rho = i_rho_p / (self.d_height - .5) * (self.s_radius + self.s_offset)
        # angular resolution
        i_theta = i_theta_p / self.d_width * 2 * np.pi
        i_theta -= np.pi/2 # Move zero degrees in the standard spot

        # Projection
        s_x = np.cos(i_theta) * i_rho
        s_y = np.sin(i_theta) * i_rho
        
        # Principal point
        c_x = (self.s_width - 1) / 2
        c_y = (self.s_height - 1) / 2

        # Add pc
        i_x, i_y = s_x + c_x, -s_y + c_y

        mask = None

        return i_x, i_y, mask

#######################################
######            Test          #######
#######################################

def __main():
    pass


if __name__=="__main__":
    __main()