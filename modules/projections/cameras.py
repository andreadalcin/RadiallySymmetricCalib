from __future__ import annotations
from typing import List, Optional, Tuple, ClassVar, Union, Dict
from dataclasses import dataclass, field, fields
import numpy as np
from abc import ABC, abstractmethod
import importlib, inspect
from scipy.spatial.transform import Rotation as Rot
import warnings

class CameraException(ValueError):
    pass

def rmat(rx,ry,rz) -> np.ndarray:
	return Rot.from_euler(seq="xyz", angles=[rx,ry,rz], degrees = True).as_matrix()

def rmat_to_euler(rmat: Union[np.ndarray, Rot]) -> Tuple[float]:
    if isinstance(rmat, np.ndarray):
        rmat = Rot.from_matrix(rmat)
    return list(rmat.as_euler(seq="xyz", degrees = True))

def apply_rotation(coordinates:Tuple[np.ndarray, np.ndarray, np.ndarray], angles=[0,0,0], inverse = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Rotate along x, y, z axes respectively.
    """
    e_x, e_y, e_z = angles

    R = rmat(e_x,e_y,e_z)
    if inverse:
        R = R.T

    Ps = np.stack(coordinates,-1)
    Ps = np.matmul(Ps,R)

    Ps_x,Ps_y,Ps_z = np.split(Ps,3,axis=-1)
    Ps_x = Ps_x.squeeze(-1)
    Ps_y = Ps_y.squeeze(-1)
    Ps_z = Ps_z.squeeze(-1)

    return Ps_x, Ps_y, Ps_z

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector, axis=-1)[...,None]

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#######################################
######      Image classes       #######
#######################################

@dataclass
class ImageDescription(ABC):
    width: int
    height: int

    # Instrinsic parameters of the model
    intrinsics: Optional[dict] = field(default_factory=dict)

    # List of possible parameters configurations. Each configuartion is a list of strings.
    intrinsic_names : ClassVar[List] = [[]]
    # Index of the intrinsic parameter configuartion given as input.
    intrinsic_configuartion_idx: int = field(default=None, init=False, repr=False) 

    # Principle point offset w.r.t. image center (d_u, d_v)
    principle_point: Optional[Tuple[float,float]] = field(default_factory=lambda :(0,0))
    is_principle_point_abs:bool = False

    # Absolute principal point (p_u, p_v)
    principle_point_abs: np.ndarray = field(init=False)

    # Rotation of the sphere
    extrinsic_rot:Optional[Tuple[float,float,float]] = (0,0,0)

    # Viewing angle map
    va_map_cache:np.ndarray = field(init=False, repr=False, default=None)

    # Compute viewing angle map during image_to_world
    compute_va_map:Optional[bool] = field(default=False, repr=False) 

    def __post_init__(self) -> None:
        if self.width is None or self.height is None:
            raise ValueError("Width or Heigth cannot be None!")
        
        if self.is_principle_point_abs:
            self.principle_point_abs = np.array(self.principle_point)
        else:
            # Pu, Pv
            self.principle_point_abs = np.array(self.principle_point) + np.array([
                (self.width - 1) / 2, (self.height - 1) / 2
            ])

        for idx, intrinsic_configuration in enumerate(self.intrinsic_names):
            if set(intrinsic_configuration).issubset(set(self.intrinsics.keys())):
                self.intrinsic_configuartion_idx = idx
                return

        raise ValueError(f"The possible intrinsics configurations are: {self.intrinsic_names}! \
            \nThis are given: {list(self.intrinsics.keys())}")
    

    def copy(self) -> ImageDescription:
        return type(self)(
            width = self.width,
            height = self.height,
            intrinsics = self.get_intrinsics(),
            principle_point = self.principle_point,
            extrinsic_rot = self.extrinsic_rot)

    def image2world(self, coordinates: Tuple[np.ndarray, np.ndarray], force_va_computation:Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Converts image coordinates to worlds coordinates according to this specific model.

        Args:
            coordinates (Tuple[np.ndarray, np.ndarray]): Tuple of arrays [u,v], according to the following convension. 
            The frame origin is the top-left corner of the image, the u=X axis points to the right and the v=Y axis points downwards.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray: Tuple of arrays [X,Y,Z,mask], according to the following convension. 
            The frame origin is the center of the camera, the Z axis points upwards, the Y axis points to the right and the X axis points towards the viewer.
            The mask is used to define which points are visible in the image.
        """  
        w_x,w_y,w_z,mask = self._image2world_impl(coordinates=coordinates)
        
        if (force_va_computation is None and self.compute_va_map) or (force_va_computation is True):
            self.compute_image_va_map( w_x,w_y,w_z, mask )

        coordinates = apply_rotation([w_x,w_y,w_z], angles=self.extrinsic_rot, inverse = False)
        return *coordinates, mask


    def compute_image_va_map(self, xs:np.ndarray, ys:np.ndarray, zs:np.ndarray, mask:np.ndarray):
        """Should be called only in the generated images is of this description type. For instance, in case of
        a mapping that involes a chain of descriptors, this method can be used only on the last one. The reason
        is that the shape of (xs, ys, zs) is the shape of the output descriptor, and this method requires that 
        the shape is the same as the descriptor. This is unlikely to ahppen for the intermediate descriptors.

        Args:
            xs (np.ndarray): array of shape (H,W) or (H/2 + 1) containing world X coordinates for each image coordinate cell.
            H and W are the image (thus the descriptor) height and width respectively.
            ys (np.ndarray): array of shape (H,W) or (H/2 + 1) containing world Y coordinates for each image coordinate cell.
            H and W are the image (thus the descriptor) height and width respectively.
            zs (np.ndarray): array of shape (H,W) or (H/2 + 1) containing world Z coordinates for each image coordinate cell.
            H and W are the image (thus the descriptor) height and width respectively.
            mask (np.ndarray): boolean array of shape (H,W) or (H/2 + 1), used to identify whether an image point is valid or not.
            H and W are the image (thus the descriptor) height and width respectively.
        """
        
        if len(xs.shape) == 2:
            assert xs.shape == (self.height, self.width) and \
                ys.shape    == (self.height, self.width) and  \
                zs.shape    == (self.height, self.width) and \
                mask.shape  == (self.height, self.width), \
                f" The 2D parameters' shape must be (H,W)! {xs.shape} is given."

        elif len(xs.shape) == 1:
            # assert xs.shape == (self.height//2,) and \
            #     ys.shape    == (self.height//2,) and  \
            #     zs.shape    == (self.height//2,) and \
            #     mask.shape  == (self.height//2,), \
            #     f" The 1D parameters' shape must be (H/2 + 1)! {xs.shape} is given."
            pass

        else:
            raise ValueError(f"The parameters shape must be (H,W) or (H)! Shape of dim {len(xs.shape)} is given.")

        ref_v = [
                1,
                0,
                0]
        
        self.va_map_cache = angle_between(np.stack([xs,ys,zs], axis=-1),ref_v)
        self.va_map_cache_masked = self.va_map_cache.copy()
        self.va_map_cache_masked[~mask] = np.NaN

    def get_va_vector(self):
        if self.height % 2 != 0:
            warnings.warn("There may be a slight inconsistency in the values if the height is not even!")

        r = (min(self.height,self.width) - 1) /2
        i_y = np.arange(0,self.height//2) + r
        i_x = np.repeat(r,self.height//2)

        # Enable the va map computation and perform the coordinate projection
        self.image2world(coordinates=[i_x, i_y], force_va_computation=True)

        return self.get_last_va_map()
    
    def get_afov(self):
        r = (min(self.height,self.width) - 1) /2
        i_y = np.array([self.height//2 -1 + r])
        i_x = np.array([r])

        mask = False
        # Enable the va map computation and perform the coordinate projection
        while not mask:
            mask = self.image2world(coordinates=[i_x, i_y], force_va_computation=True)[-1]
            i_y -= 1
            

        return np.degrees(self.get_last_va_map() * 2)[0]
        

    def world2image(self, coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            coordinates (Tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple of arrays [X,Y,Z], according to the following convension. 
            The frame origin is the center of the camera, the Z axis points upwards, the Y axis points to the right and the X axis points towards the viewer.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of arrays [u,v,mask], according to the following convension. 
            The frame origin is the top-left corner of the image, the u=X axis points to the right and the v=Y axis points downwards.
            The mask is used to define which points are visible in the image.
        """
        coordinates = apply_rotation(coordinates=coordinates, angles=self.extrinsic_rot, inverse = True)        
        return self._world2image_impl(coordinates=coordinates)

    @staticmethod
    def get_instrinsics_bounds() -> List[Tuple]:
        """Use one specific intrinsic configuration among the available ones, the configuration
        should be the first in the intrinsic_names list. 

        Returns
        -------
        List[Tuple]
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError( "Camera model not implemented" )

    @staticmethod
    def radial_proj_func(xyz: np.ndarray, intrinsics:dict) -> np.ndarray:
        """Implements the projection function for a specific dconfiguration of intrinsic parameters. 
        The configuration used should be the default one, thus the first in the intrinsic_names list.

        Parameters
        ----------
        xyz : np.ndarray
            _description_
        intrinsics : dict
            _description_

        Returns
        -------
        np.ndarray
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError( "Camera model not implemented" )
    
    @staticmethod
    def jacobian(xyz: np.ndarray, *intrinsics:dict) -> np.ndarray:
        raise NotImplementedError( "Camera model not implemented" )

    @abstractmethod
    def _image2world_impl(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Converts image coordinates to worlds coordinates according to this specific model.

        Args:
            coordinates (Tuple[np.ndarray, np.ndarray]): Tuple of arrays [u,v], according to the following convension. 
            The frame origin is the top-left corner of the image, the u=X axis points to the right and the v=Y axis points downwards.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray: Tuple of arrays [X,Y,Z,mask], according to the following convension. 
            The frame origin is the center of the camera, the Z axis points upwards, the Y axis points to the right and the X axis points towards the viewer.
            The mask is used to define which points are visible in the image.
        """        
        raise NotImplementedError( "Camera model not implemented" )
        
    @abstractmethod
    def _world2image_impl(self, coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            coordinates (Tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple of arrays [X,Y,Z], according to the following convension. 
            The frame origin is the center of the camera, the Z axis points upwards, the Y axis points to the right and the X axis points towards the viewer.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of arrays [X,Y,mask], according to the following convension. 
            The frame origin is the top-left corner of the image, the X axis points to the right and the Y axis points downwards.
            The mask is used to define which points are visible in the image.
        """        
        raise NotImplementedError( "Camera model not implemented" )

    def is_compatible(self, des: ImageDescription) -> bool:
        return type(self) == type(des) and self.intrinsics == des.intrinsics

    def combine(self, des: ImageDescription) :
        assert self.is_compatible(des)
        self.height = des.height
        self.width = des.width
        self.extrinsic_rot = rmat_to_euler(rmat(self.extrinsic_rot) @ rmat(des.extrinsic_rot))

    def get_last_va_map(self) -> np.ndarray:
        return self.va_map_cache
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.height == other.height and \
                self.width == other.width and \
                self.get_intrinsics() == other.get_intrinsics()
                #self.extrinsic_rot == other.extrinsic_rot 
        return False

    @staticmethod
    def from_des_dict(des_dict: dict) -> ImageDescription:
        names = inspect.getmembers(importlib.import_module("projections.cameras"),inspect.isclass)
        elem = list(filter(lambda x:des_dict['type'].lower() in x[0].lower(),names))[0]

        fs = [ f.name for f in fields(ImageDescription)]

        # remove wrong parameters
        to_del = [ key for key in des_dict.keys() if key not in fs ]
        for k in to_del:
            des_dict.pop(k, None)

        #build the object
        return elem[1](**des_dict)

    def to_dict(self) -> dict:
        return dict(
            type=self.__class__.__name__.split("_")[0],
            width = self.width,
            height = self.height,
            intrinsics = self.get_intrinsics(),
            extrinsic_rot = self.extrinsic_rot
            )

    @abstractmethod
    def get_intrinsics(self) -> Dict[str, float]:
        raise NotImplementedError()

#######################################
######      Image classes       #######
#######################################

@dataclass
class FisheyeEUCM_Description(ImageDescription):
    name: str = 'EUCM'
    
    # focal length
    f: float = field(init=False)

    # Radial distortion parameters
    a_: float = field(init=False)
    b_: float = field(init=False)

    intrinsic_names : ClassVar[List] = [
        ["f","a","b"],       # configuration 0
        ["afov","a","b"],   # configuration 1
        ]

    def __post_init__(self) -> None:
        super().__post_init__()
        intrinsic_configuration = self.intrinsic_names[self.intrinsic_configuartion_idx]
            
        self.a_   = self.intrinsics[intrinsic_configuration[1]]
        self.b_   = self.intrinsics[intrinsic_configuration[2]]

        if self.intrinsic_configuartion_idx == 0:
            self.f = self.intrinsics[intrinsic_configuration[0]]

        elif self.intrinsic_configuartion_idx == 1:
            afov = self.intrinsics[intrinsic_configuration[0]]
            self.f = self._compute_focal_from_afov(afov = afov)

    def get_intrinsics(self):
        return dict(
            f=self.f,
            a=self.a_,
            b=self.b_,
        )
    
    @staticmethod
    def get_instrinsics_bounds() -> List[Tuple]:
        parameterBounds = []
        parameterBounds.append([0., 500.]) # search bounds for f
        parameterBounds.append([0., 1.]) # search bounds for a_
        parameterBounds.append([0., 2.]) # search bounds for b_
        return parameterBounds

    def _compute_focal_from_afov(self, afov: float):
        x_ref = [1,0,0]
        coordinates = apply_rotation(coordinates=x_ref, angles=[0,0,afov/2], inverse = True)

        max_r = min(self.width, self.height)//2 -1

        x,y,z = coordinates
        # Change frame from the world to the camera one.
        x,y,z = y,z,x   

        rho = np.sqrt(self.b_*(x**2 + y**2) + z**2)
        norm = self.a_*rho + (1-self.a_)*z
        
        focal = max_r /( x / norm)

        w = self.a_/(1 - self.a_) if self.a_ <= 0.5 else (1-self.a_)/self.a_

        if z < -w*rho:
            raise CameraException("This combination of parameters is not valid!")

        return focal

    def _image2world_impl(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # u: cols, dim 1, width, x
        # v: rows, dim 0, height, y
        u,v = coordinates

        Cu = (self.width - 1) / 2
        Cv = (self.height - 1) / 2 # image y

        C = min(Cv, Cu)

        # To add the black circle every time
        r_2_max = C**2 / self.f**2

        # invert the camera calibration projection, divide by K
        my = -(v - Cv)/self.f # The minus sign is due to the change in the Y axis direction from image to camera plane.
        mx = (u - Cu)/self.f

        r_2 = my**2 + mx**2

        mask = None
        if self.a_ > 0.5:
            # maximum value for which r is defined (according to the paper)
            r_2_max = min(r_2_max, np.abs(1/(self.b_*(2*self.a_-1))))
        
        mask = r_2 <= r_2_max

        # Inverse projection according to the EUCM, gamma = (1 - a)
        mz = np.real((1-self.b_*self.a_*self.a_*r_2)/(self.a_*np.lib.scimath.sqrt(1-(2*self.a_-1)*self.b_*r_2) + (1-self.a_)))

        # Normalization: projection on the sphere surface
        coef = 1/np.sqrt(my**2 + mx**2 + mz**2)

        out_x = mx*coef
        out_y = my*coef
        out_z = mz*coef


        # DEFAULT CAMERA FRAME: if v'm the object, Z points to me, Y points upwards, Z points right.
        # Change frame from camera to world
        out_x,out_y,out_z = out_z,out_x,out_y

        # lats = np.arctan2(out_z,np.sqrt(out_x**2+out_y**2))
        # longs = np.arctan2(out_y,out_x)

        return out_x, out_y, out_z, mask

    def _world2image_impl(self,coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        
        x,y,z = coordinates
        # Change frame from the world to the camera one.
        x,y,z = y,z,x

        Cu = (self.width - 1) / 2
        Cv = (self.height - 1) / 2 # image y

        C = min(Cv, Cu)
        
        rho = np.sqrt(self.b_*(x**2 + y**2) + z**2)
        norm = self.a_*rho + (1-self.a_)*z

        m_x = x / norm 
        # The minus sign is due to the change in the Y axis direction from image to camera plane.
        m_y = - y / norm

        u = m_x * self.f + Cu
        v = m_y * self.f + Cv
        
        r_2_max = C**2 / self.f**2
        r_2 = m_y**2 + m_x**2

        w = self.a_/(1 - self.a_) if self.a_ <= 0.5 else (1-self.a_)/self.a_

        in_plane_mask = (r_2 <= r_2_max) & (z > -w*rho)

        return u, v, in_plane_mask
    
    def __eq__(self, other):
        return ImageDescription.__eq__(self,other)
    
    @staticmethod
    def radial_proj_func(xyz: np.ndarray, f, a_, b_) -> np.ndarray:

        rho = np.sqrt(b_*(xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1]) + xyz[:,2]*xyz[:,2])
        norm = 1/(a_*rho + (1-a_)*xyz[:,2])
        
        i_xy = xyz[:,:2] * norm[:,None] * f
        i_rho = np.linalg.norm(i_xy, axis=1)
        return i_rho
    
    @staticmethod
    def jacobian(xyz: np.ndarray, f, a_, b_) -> np.ndarray:
        # Projection
        
        rho = np.sqrt(b_*(xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1]) + xyz[:,2]*xyz[:,2])
        norm = 1/(a_*rho + (1-a_)*xyz[:,2])
        
        w_rho = np.linalg.norm(xyz[:,:2], axis=1) 
        norm2 = norm*norm
        
        return np.asarray([w_rho*norm, w_rho*f* norm2*(xyz[:,2] -rho), -w_rho*f*norm2*(a_*(xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1])/(2*rho))]).T

@dataclass
class FisheyeUCM_Description(FisheyeEUCM_Description):
    name: str = 'UCM'

    intrinsic_names : ClassVar[List] = [
        ["f","a"],       # configuration 0
        ["gamma","xi"],     # configuration 1
        ["afov","a"],     # configuration 2
    ]

    def get_intrinsics(self):
        return dict(
            f=self.f,
            a=self.a_,
        )
        
    def __eq__(self, other):
        return ImageDescription.__eq__(self,other)

    def __post_init__(self) -> None:
        ImageDescription.__post_init__(self=self)
        intrinsic_configuration = self.intrinsic_names[self.intrinsic_configuartion_idx]
            
        self.b_ = 1

        # [f, a]
        if self.intrinsic_configuartion_idx == 0:
            self.f  = self.intrinsics[intrinsic_configuration[0]]
            self.a_ = self.intrinsics[intrinsic_configuration[1]]

        # [gamma, xi] converted following: https://arxiv.org/pdf/1807.08957.pdf
        elif self.intrinsic_configuartion_idx == 1:
            gamma  = self.intrinsics[intrinsic_configuration[0]]
            xi     = self.intrinsics[intrinsic_configuration[1]]
            self.a_ = xi / (1 + xi)
            self.f  = gamma * ( 1 - self.a_)

        elif self.intrinsic_configuartion_idx == 2:
            self.a_ = self.intrinsics[intrinsic_configuration[1]]
            self.f = self._compute_focal_from_afov(afov = self.intrinsics[intrinsic_configuration[0]])

    @staticmethod
    def get_instrinsics_bounds() -> List[Tuple]:
        parameterBounds = []
        parameterBounds.append([0., 1000.]) # search bounds for f
        parameterBounds.append([0., 1.]) # search bounds for a_
        return parameterBounds

    @staticmethod
    def radial_proj_func(xyz: np.ndarray, f, a_) -> np.ndarray:
        return FisheyeEUCM_Description.radial_proj_func(xyz, f, a_, 1)
    
    @staticmethod
    def jacobian(xyz: np.ndarray, f, a_) -> np.ndarray:
        return FisheyeEUCM_Description.jacobian(xyz, f, a_, 1)[:,:2]

@dataclass
class FisheyePOLY_Description(ImageDescription):
   
    distortion_coeffs: np.ndarray = field(init=False)
    distortion_power: np.ndarray = field(init=False)

    intrinsic_names : ClassVar[List] = [["distortion_coeffs"]]

    def get_intrinsics(self):
        return dict(
            intrinsics = self.intrinsics
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        intrinsic_configuration = self.intrinsic_names[self.intrinsic_configuartion_idx]
        self.distortion_coeffs = np.asarray( self.intrinsics[intrinsic_configuration[0]])
        self.distortion_power = np.array([np.arange(start=1, stop=self.distortion_coeffs.size + 1)])[None,:]

    def _theta_to_rho(self, theta):
        return np.dot(np.power(np.array(theta)[:,:,None], self.distortion_power), self.distortion_coeffs)

    def _rho_to_theta(self, rho):
        coeff = list(self.distortion_coeffs[::-1])

        def f(x):
            theta = np.roots([*coeff, -x])
            theta = np.real(theta[theta.imag == 0])
            theta = theta[np.where(np.abs(theta) < np.pi)]
            theta = np.min(theta) if theta.size > 0 else 0
            return theta

        return np.vectorize(f)(rho)

    def _image2world_impl(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # u: cols, dim 1, width, x
        # v: rows, dim 0, height, y
        
        principle_point = 0.5 * np.array([self.width,  self.height]) + np.array(self.principle_point, dtype=float) - 0.5

        lens_points = (np.array(coordinates).transpose((1,2,0)) - principle_point)

        rhos = np.linalg.norm(lens_points, axis=-1)
        thetas = self._rho_to_theta(rhos)
        chis = np.sin(thetas)
        zs = np.cos(thetas)
        xy = np.divide(chis, rhos, where=(rhos != 0))[:,:, np.newaxis] * lens_points
       
        mx = xy[:,:,0]
        my = -xy[:,:,1]
        mz = zs
        # Normalization: projection on the sphere surface
        coef = 1/np.sqrt(mx**2 + my**2 + mz**2)

        out_x = mx*coef
        out_y = my*coef
        out_z = mz*coef

        # DEFAULT CAMERA FRAME: if v'm the object, Z points to me, Y points upwards, Z points right.
        # Change frame from camera to world
        out_x,out_y,out_z = out_z,out_x,out_y

        mask = np.full_like(out_x, True, dtype=bool)
        return out_x, out_y, out_z, mask

    def _world2image_impl(self,coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # Change frame from the world to the camera one.
        # x,y,z = y,z,x
        coordinates  = [coordinates[1], coordinates[2], coordinates[0]]
        
        principle_point = 0.5 * np.array([ self.width,  self.height]) + np.array(self.principle_point, dtype=float) - 0.5

        chi = np.sqrt(coordinates[0] * coordinates[0] + coordinates[1] * coordinates[1])
        theta = np.pi / 2.0 - np.arctan2(coordinates[2], chi)
        rho = self._theta_to_rho(theta)
        lens_points = np.divide(rho, chi, where=(chi != 0))[:,:, np.newaxis] * np.array(coordinates)[0:2].transpose(1,2,0)
        
        # y = -y
        lens_points[:,:,1] = -lens_points[:,:,1]
        screen_points = lens_points + principle_point

        u = screen_points[:,:,0]
        v = screen_points[:,:,1]

        in_plane_mask = ~((chi == 0) & (coordinates[2] == 0)) & (u>=0) & (u <  self.width) & (v>=0) & (v< self.height)
        return u, v, in_plane_mask

@dataclass
class FisheyeMEI_Description(ImageDescription):
    # https://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf
    
    xi: Dict[str, float] = field(init=False)
    k1: Dict[str, float] = field(init=False)
    k2: Dict[str, float] = field(init=False)
    gamma1: Dict[str, float] = field(init=False)
    gamma2: Dict[str, float] = field(init=False)
    u0: Dict[str, float] = field(init=False)
    v0: Dict[str, float] = field(init=False)

    intrinsic_names : ClassVar[List] = [["mirror_parameters", "distortion_parameters", "projection_parameters"]]

    def get_intrinsics(self):
        return dict(
            intrinsics = self.intrinsics
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        intrinsic_configuration = self.intrinsic_names[self.intrinsic_configuartion_idx]
        mirror_parameters = self.intrinsics[intrinsic_configuration[0]]
        distortion_parameters = self.intrinsics[intrinsic_configuration[1]]
        projection_parameters = self.intrinsics[intrinsic_configuration[2]]

        self.xi = mirror_parameters['xi']
        self.k1 = distortion_parameters['k1']
        self.k2 = distortion_parameters['k2']
        
        self.gamma1 = projection_parameters['gamma1']
        self.gamma2 = projection_parameters['gamma2']
        self.u0 = projection_parameters['u0']
        self.v0 = projection_parameters['v0']
        
    def _world2image_impl(self,coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x,y,z = coordinates
        # Change frame from the world to the camera one.
        x,y,z = y,z,x

        norm = np.sqrt(x**2 + y**2 + z**2)
        
        x = x / norm
        y = y / norm
        z = z / norm

        x /= z+self.xi
        y /= z+self.xi

        ro2 = x*x + y*y
        x *= 1 + self.k1*ro2 + self.k2*ro2*ro2
        y *= 1 + self.k1*ro2 + self.k2*ro2*ro2

        u = self.gamma1*x + self.u0
        v = -self.gamma2*y + self.v0

        in_plane_mask = (u>=0) & (u < self.width) & (v>=0) & (v< self.height)

        return u, v, in_plane_mask

    def _image2world_impl(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Inverse projection is not defined!")

@dataclass
class FisheyeDS_Description(ImageDescription):
    name: str = 'DS'
    
    # focal length
    f: float = field(init=False)

    # Radial distortion parameters
    a_: float = field(init=False)
    xi_: float = field(init=False)

    intrinsic_names : ClassVar[List] = [["f","a","xi"],
                                        ["afov","a","xi"]]

    def get_intrinsics(self):
        return dict(
            f=self.f,
            a=self.a_,
            xi=self.xi_,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        intrinsic_configuration = self.intrinsic_names[self.intrinsic_configuartion_idx]


        # ["f","a","xi"]
        if self.intrinsic_configuartion_idx == 0:
            self.f = self.intrinsics[intrinsic_configuration[0]]    
            self.a_ = self.intrinsics[intrinsic_configuration[1]]  
            self.xi_ = self.intrinsics[intrinsic_configuration[2]]  

        # ["afov","a","xi"]
        elif self.intrinsic_configuartion_idx == 1:
            afov = self.intrinsics[intrinsic_configuration[0]]    
            self.a_ = self.intrinsics[intrinsic_configuration[1]]  
            self.xi_ = self.intrinsics[intrinsic_configuration[2]]  
            self.f = self._focal_from_afov(afov)
    

    def _focal_from_afov(self, afov) -> float:
        x_ref = [1,0,0]
        coordinates = apply_rotation(coordinates=x_ref, angles=[0,0,afov/2], inverse = True)

        max_r = min(self.width, self.height)//2 -1

        x,y,z = coordinates
        # Change frame from the world to the camera one.
        x,y,z = y,z,x   

        # Projection
        d1 = np.sqrt(x**2 + y**2 + z**2)
        d2 = np.sqrt(x**2 + y**2 + (self.xi_*d1 + z)**2)

        norm = 1 / (self.a_ * d2 + (1- self.a_)*(self.xi_*d1+z))
        
        focal = max_r /( x * norm)

        w1 = self.a_ / (1-self.a_) if self.a_ <= 0.5 else (1-self.a_)/self.a_

        arg = 2*w1*self.xi_ + self.xi_**2 + 1
        
        w2 = (w1 + self.xi_) / np.sqrt(arg) 
        z_min = -w2*d1

        if z < z_min:
            raise CameraException("This combination of parameters is not valid!")

        return focal

    def _image2world_impl(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # u: cols, dim 1, width, x
        # v: rows, dim 0, height, y

        u,v = coordinates

        Cu = (self.width - 1) / 2     # even img size: center pixel does not exists, odd: center pixel is the middle one
        Cv = (self.height - 1) / 2    
        C = min(self.width, self.height) / 2

        # invert the camera calibration projection, divide by K
        my = -(v - Cv)/self.f
        mx = (u - Cu)/self.f

        r_2 = my**2 + mx**2

        # max radius for an inscribed circle
        r_2_max = C**2 / self.f**2
        if self.a_ > 0.5:
            # maximum value for which r is defined (according to the paper)
            r_2_max = min(r_2_max, np.abs(1/(2*self.a_-1)))
        
        mask = r_2 <= r_2_max

        # Inverse projection according to the DS
        mz = np.real(
            (1 - self.a_ * self.a_ * r_2)
            / (self.a_ * np.lib.scimath.sqrt(1 - (2 * self.a_ - 1) * r_2) +
               1 - self.a_)
        )

        omega = np.real(
            (mz * self.xi_ + np.lib.scimath.sqrt(mz ** 2 +
                                                 (1 - self.xi_ ** 2) * r_2))
            / (mz ** 2 + r_2)
        )
        out_x = mx*omega
        out_y = my*omega
        out_z = mz*omega - self.xi_

        # Normalization: projection on the sphere surface
        coef = 1/np.sqrt(out_x**2 + out_y**2 + out_z**2)
        out_x = out_x*coef
        out_y = out_y*coef
        out_z = out_z*coef

        # DEFAULT CAMERA FRAME: if v'm the object, Z points to me, Y points upwards, Z points right.
        # Change frame from camera to world
        out_x,out_y,out_z = out_z,out_x,out_y

        return out_x, out_y, out_z, mask

    def _world2image_impl(self,coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
       
        x,y,z = coordinates
        # Change frame from the world to the camera one.
        x,y,z = y,z,x

        Cu = (self.width - 1) / 2     # even img size: center pixel does not exists, odd: center pixel is the middle one
        Cv = (self.height - 1) / 2       

        # Projection
        d1 = np.sqrt(x**2 + y**2 + z**2)
        d2 = np.sqrt(x**2 + y**2 + (self.xi_*d1 + z)**2)

        norm = 1 / (self.a_ * d2 + (1- self.a_)*(self.xi_*d1+z))
        
        m_y = -y * norm
        m_x = x * norm

        u = m_x * self.f + Cu
        v = m_y * self.f + Cv

        # Valid projection check
        w1 = self.a_ / (1-self.a_) if self.a_ <= 0.5 else (1-self.a_)/self.a_

        w2 = (w1 + self.xi_) / np.sqrt(2*w1*self.xi_ + self.xi_**2 + 1) 
        z_min = -w2*d1

        in_plane_mask = (z>=z_min) & (u>=0) & (u <  self.width) & (v>=0) & (v< self.height)

        return u, v, in_plane_mask

    @staticmethod
    def get_instrinsics_bounds() -> List[Tuple]:
        parameterBounds = []
        parameterBounds.append([0., 500.]) # search bounds for f
        parameterBounds.append([0., 1.]) # search bounds for a_
        parameterBounds.append([-1., 1.]) # search bounds for xi_
        return parameterBounds
    
    def __eq__(self, other):
        return ImageDescription.__eq__(self,other)

    @staticmethod
    def radial_proj_func(xyz: np.ndarray, f, a_, xi_) -> np.ndarray:

        # Projection
        d1 = np.linalg.norm(xyz, axis=1)
        d2 = np.sqrt(xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1] + np.power(xi_*d1 + xyz[:,2], 2))

        norm = 1 / (a_ * d2 + (1- a_)*(xi_*d1+xyz[:,2]))
        
        i_xy = xyz[:,:2] * norm[:,None] * f
        i_rho = np.linalg.norm(i_xy, axis=1)
        return i_rho
    
    @staticmethod
    def jacobian(xyz: np.ndarray, f, a_, xi_) -> np.ndarray:
        # Projection
        d1 = np.linalg.norm(xyz, axis=1)
        d2 = np.sqrt(xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1] + np.power(xi_*d1 + xyz[:,2], 2))

        coeff = xi_*d1+xyz[:,2]
        norm = 1 / (a_ * d2 + (1- a_)*(coeff))
        
        w_rho = np.linalg.norm(xyz[:,:2], axis=1) 
        norm2 = norm*norm
        
        return np.asarray([w_rho*norm, w_rho*f* norm2*(coeff -d2), -w_rho*f*norm2*d1*(1-a_+a_*coeff/d2)]).T

@dataclass
class FisheyeFOV_Description(ImageDescription):
    # focal length
    fx: float = field(init=False)
    fy: float = field(init=False)

    # Radial distortion parameters
    w_: float = field(init=False)

    intrinsic_names : ClassVar[List] = [["fx","fy","w"]]

    def get_intrinsics(self):
        return dict(
            fx=self.fx,
            fy=self.fy,
            w=self.w_,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        intrinsic_configuration = self.intrinsic_names[self.intrinsic_configuartion_idx]
        self.fx = self.intrinsics[intrinsic_configuration[0]]    
        self.fy = self.intrinsics[intrinsic_configuration[1]]    
        self.w_ = self.intrinsics[intrinsic_configuration[2]]  

    def _image2world_impl(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Inverse projection is not defined!")

    def _world2image_impl(self,coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
       
        x,y,z = coordinates
        # Change frame from the world to the camera one.
        x,y,z = y,z,x

        Cu = self.principle_point_abs[0]
        Cv = self.principle_point_abs[1]

        ru = np.sqrt(x**2 + y**2)
        rd = np.arctan2(2*ru*np.tan(self.w_/2), z) / self.w_
        

        m_y = np.divide(-y * rd, ru, out=np.zeros_like(ru), where=ru!=0)
        m_x = np.divide(x * rd, ru, out=np.zeros_like(ru), where=ru!=0)

        u = m_x * self.fx + Cu
        v = m_y * self.fy + Cv

        in_plane_mask = (u>=0) & (u <  self.width) & (v>=0) & (v< self.height)

        return u, v, in_plane_mask


# Libraries imported for fast mathematical computations.
import math
import numpy as np


# Main Function takes in the coefficient of the Cubic Polynomial
# as parameters and it returns the roots in form of numpy array.
# Polynomial Structure -> ax^3 + bx^2 + cx + d = 0

def solve(a, b, c, d):
    assert b==0

    if (a == 0 and b == 0):                     # Case for handling Liner Equation
        return (-d * 1.0) / c                 # Returning linear root as numpy array.

    f = findF(a, b, c)                          # Helper Temporary Variable
    g = findG(a, b, c, d)                       # Helper Temporary Variable
    h = findH(g, f)                             # Helper Temporary Variable
    
    outs = np.zeros_like(d)
    
    case_1 = (f == 0) & (g == 0) & (h == 0)
    case_2 = (~case_1) & (h <= 0)
    case_3 = (~case_1) & (h>0)
    
    d_1 = d[case_1]
    
    if d_1.shape[0] > 0:
        outs[case_1] = np.where((d_1 / a) >= 0, (d_1 / (1.0 * a)) ** (1 / 3.0) * -1, (-d_1 / (1.0 * a)) ** (1 / 3.0))
    
    d_2 = d[case_2]
    if d_2.shape[0] > 0:

        i = np.sqrt(((g[case_2] ** 2.0) / 4.0) - h[case_2])   # Helper Temporary Variable
        j = i ** (1 / 3.0)                      # Helper Temporary Variable
        k = np.arccos(-(g[case_2] / (2 * i)))           # Helper Temporary Variable
        L = j * -1                              # Helper Temporary Variable
        M = np.cos(k / 3.0)                   # Helper Temporary Variable
        N = np.sqrt(3) * np.sin(k / 3.0)    # Helper Temporary Variable
        P = (b / (3.0 * a)) * -1                # Helper Temporary Variable

        x1 = 2 * j * np.cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P
        
        
        sols = np.stack([x1, x2, x3],-1)
        sols.sort(axis=-1)

        outs[case_2] =  sols[...,1]          # Returning Real Roots as numpy array.
        
    d_3 = d[case_3]
    if d_3.shape[0] > 0:
                                                  # One Real Root and two Complex Roots
        R = -(g[case_3] / 2.0) + np.sqrt(h[case_3])           # Helper Temporary Variable
        S =  np.cbrt(R) 
        T = -(g[case_3] / 2.0) - np.sqrt(h[case_3])
        
        U =  np.cbrt(T) 

        x1 = (S + U) - (b / (3.0 * a))
        # x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * math.sqrt(3) * 0.5j
        # x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * math.sqrt(3) * 0.5j

        outs[case_3] =  x1
        
    return outs


# Helper function to return float value of f.
def findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


# Helper function to return float value of g.
def findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a **2.0)) + (27.0 * d / a)) /27.0


# Helper function to return float value of h.
def findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)

@dataclass
class FisheyeWakai_Description(ImageDescription):
    """https://arxiv.org/pdf/2111.12927.pdf"""
    name:str = 'KB3'
    
    # focal length
    f: float = field(init=False)

    # Radial distortion parameters
    k: float = field(init=False)

    intrinsic_names : ClassVar[List] = [["f","k"]]

    def get_intrinsics(self):
        return dict(
            f=self.f,
            k=self.k,
        )
        
    @staticmethod
    def get_instrinsics_bounds() -> List[Tuple]:
        parameterBounds = []
        parameterBounds.append([0., 400.]) # search bounds for f
        parameterBounds.append([-1/6, 1./3]) # search bounds for k
        return parameterBounds

    def __post_init__(self) -> None:
        super().__post_init__()
        intrinsic_configuration = self.intrinsic_names[self.intrinsic_configuartion_idx]
        self.f = self.intrinsics[intrinsic_configuration[0]]    
        self.k = self.intrinsics[intrinsic_configuration[1]]

    def _image2world_impl(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # u: cols, dim 1, width, x
        # v: rows, dim 0, height, y

        u,v = coordinates

        Cu = (self.width - 1) / 2     # even img size: center pixel does not exists, odd: center pixel is the middle one
        Cv = (self.height - 1) / 2    

        # invert the camera calibration projection, divide by K
        my = -(v - Cv)
        mx = (u - Cu)
        
        rd = np.sqrt(mx**2 + my**2)
        _w = rd /self.f
        
        theta = solve(self.k,0,1,-_w)
         
        # rd = f ( theta + k * theta^3)
        # w = theta + k * theta^3
        # theta^3 + theta 1/k - w/k = 0
        # p = 1/self.k
        # q = - _w/self.k
        # delta = np.sqrt(q**2 / 4 + p **3 / 27)
        # theta = np.cbrt(-q/2 + delta) + np.cbrt(-q/2 - delta)
        
        # print(theta, )
        
        phi = np.arctan2(my, mx)


        out_x = np.cos(phi) * np.sin(theta)
        out_y = np.sin(phi) * np.sin(theta)
        out_z = np.cos(theta)

        mask = np.full_like(out_x, True, dtype=bool)
        # DEFAULT CAMERA FRAME: if v'm the object, Z points to me, Y points upwards, Z points right.
        # Change frame from camera to world
        out_x,out_y,out_z = out_z,out_x,out_y

        return out_x, out_y, out_z, mask
    
    @staticmethod
    def radial_proj_func(xyz: np.ndarray, f, k) -> np.ndarray:

        ru = np.sqrt(xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1]+ xyz[:,2]*xyz[:,2])
        theta = np.arccos(xyz[:,2]/ru)
        rd = f*(theta + k* np.power(theta,3))
        return rd
    
    @staticmethod
    def jacobian(xyz: np.ndarray, f, k) -> np.ndarray:
        ru = np.sqrt(xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1]+ xyz[:,2]*xyz[:,2])
        theta = np.arccos(xyz[:,2]/ru)
        
        theta_3 = np.power(theta,3)        
        return np.asarray([theta + k* theta_3, theta_3*f ]).T

    def _world2image_impl(self,coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
       
        x,y,z = coordinates
        # Change frame from the world to the camera one.
        x,y,z = y,z,x

        Cu = (self.width - 1) / 2     # even img size: center pixel does not exists, odd: center pixel is the middle one
        Cv = (self.height - 1) / 2    
        
        rr = np.sqrt(x**2 + y**2)

        ru = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/ru)
        rd = self.f * (theta + self.k* np.power(theta,3))

        m_y = np.divide(-y * rd, rr, out=np.zeros_like(rr), where=rr!=0)
        m_x = np.divide(x * rd, rr, out=np.zeros_like(rr), where=rr!=0)
        
        assert np.allclose(np.sqrt(m_y**2 + m_x**2), rd)

        u = m_x + Cu
        v = m_y + Cv

        in_plane_mask = (u>=0) & (u <  self.width) & (v>=0) & (v< self.height)

        return u, v, in_plane_mask

@dataclass
class FisheyeOPENCV_Description(ImageDescription):
    # focal length
    fx: float = field(init=False)
    fy: float = field(init=False)

    # Radial distortion parameters
    ks: List[float] = field(init=False)

    intrinsic_names : ClassVar[List] = [["fx","fy","ks"]]

    def get_intrinsics(self):
        return dict(
            fx=self.fx,
            fy=self.fy,
            ks=self.ks,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        intrinsic_configuration = self.intrinsic_names[self.intrinsic_configuartion_idx]
        self.fx = self.intrinsics[intrinsic_configuration[0]]
        self.fy = self.intrinsics[intrinsic_configuration[1]]
        self.ks = self.intrinsics[intrinsic_configuration[2]]  

    def _image2world_impl(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Inverse projection is not defined!")

    def _world2image_impl(self,coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
       
        x,y,z = coordinates
        # Change frame from the world to the camera one.
        x,y,z = y,z,x

        r2 = x**2 + y**2
        r = np.sqrt(r2)
        theta = np.arctan2(r, z)

        theta2 = theta*theta
        theta4 = theta2*theta2
        theta6 = theta2*theta4
        theta8 = theta4*theta4
        theta_d = theta * (1 + self.ks[0]* theta2 +  self.ks[1] * theta4 + self.ks[2] * theta6 + self.ks[3] * theta8)

        m_x = np.divide(theta_d, r, out=np.zeros_like(r), where=r!=0) * x
        m_y = -np.divide(theta_d, r, out=np.zeros_like(r), where=r!=0) * y

        Cu = self.principle_point_abs[0]
        Cv = self.principle_point_abs[1]
        u = m_x * self.fx + Cu
        v = m_y * self.fy + Cv

        in_plane_mask = (u>=0) & (u <  self.width) & (v>=0) & (v< self.height)
        in_plane_mask = (u>=0) & (u <  self.width) & (v>=0) & (v< self.height)

        return u, v, in_plane_mask

@dataclass
class Perspective_Description(ImageDescription):
    # angular fov, used instead of focal length
    f: float = field(init=False)

    intrinsic_names : ClassVar[List]= [
        ["f"],
        ["afov"], # in degrees
        ]

    def get_intrinsics(self):
        return dict(
            f=self.f,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        intrinsic_configuration = self.intrinsic_names[self.intrinsic_configuartion_idx]

        # [f]
        if self.intrinsic_configuartion_idx == 0:
            self.f = self.intrinsics[intrinsic_configuration[0]]

        # [afov]
        elif self.intrinsic_configuartion_idx == 1:
            afov = self.intrinsics[intrinsic_configuration[0]]
            self.f = self._compute_focal_from_afov(afov = afov)

    def _compute_focal_from_afov(self, afov: float):
        # The angolar field of view will be seen along the larger dimension (width or heigth)
        f = max(self.height, self.width) / (2 * np.tan(np.radians(afov / 2)))
        return f 

    def _image2world_impl(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # u: cols, dim 1, width, x
        # v: rows, dim 0, height, y
        u,v = coordinates

        Cu = (self.width - 1) / 2     # even img size: center pixel does not exists, odd: center pixel is the middle one
        Cv = (self.height - 1) / 2

        # image plane is placed in the center of the eqr image, so the plaine is x=1.
        x = np.ones(v.shape, np.float32)
        y = (u - Cu) / self.f
        z = - (v - Cv) / self.f
        D = np.sqrt(x**2 + y**2 + z**2)

        x = x/D
        y = y/D
        z = z/D

        mask = np.full_like(x, True, dtype=bool)
        return x, y, z, mask

    def _world2image_impl(self, coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x,y,z = coordinates
        # Change frame from the world to the camera one.
        x,y,z = y,z,x

        c_x = (self.width - 1) / 2     # even img size: center pixel does not exists, odd: center pixel is the middle one
        c_y = (self.height - 1) / 2 

        # project every point on the planes z=|1|
        x /= abs(z)
        y /= abs(z)
        z /= abs(z)

        u = x * self.f + c_x
        v = -y * self.f + c_y

        in_plane_mask = (z>0) & (u>=0) & (u <  self.width) & (v>=0) & (v< self.height)

        return u, v, in_plane_mask

@dataclass
class Equirectangular_Description(ImageDescription):

    def get_intrinsics(self):
        return dict()

    def _image2world_impl(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert  self.width ==  self.height*2
        # u: cols, dim 1, width, x
        # v: rows, dim 0, height, y
        
        u,v = coordinates

        # latitudes and longitudes on sphere
        lats = (v/( self.height-1) - 0.5) * -np.pi # sign is negative in order to make the upper region positive
        longs = (u/( self.width-1) - 0.5) * 2*np.pi

        # euclidean coordinates on the unit sphere
        x = np.cos(lats) * np.cos(longs)
        y = np.cos(lats) * np.sin(longs)
        z = np.sin(lats)

        mask = np.full_like(x, True, dtype=bool)

        return x, y, z, mask

    def _world2image_impl(self, coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        #  self.heighteight and width of the equirectangular image
        assert  self.width ==  self.height*2

        x, y, z = coordinates
        lats = np.arctan2(z,np.sqrt(x**2+y**2)) # better the arcsin in case of radius different from 1.
        longs = np.arctan2(y,x)

        u = (longs/(2*np.pi) + 0.5) * ( self.width-1)
        v = (-lats/(np.pi) + 0.5) * ( self.height-1)
        mask = np.full_like(u, True, dtype=bool)

        return u, v, mask

def __main():
    print(type(Equirectangular_Description(width=0,height=0)).__name__)
    print(Equirectangular_Description.__name__)
    d = {'type': 'fisheyeEUCM', 'width': 13, 'height': 12, 'intrinsics':{'afov': 111, 'a': 32132, 'b': 1231233}, 'extrinsic_rot': (0, 10, 0)}
        
    print(ImageDescription.from_des_dict(d))

    print(Equirectangular_Description(width=0,height=0).to_dict())

def __angle_main():
    v1 = np.array((1,0,0))
    v2 = np.array([[(0,1,0),(0,0,1)],[(0,5,5),(5,5,0)]])
    print(angle_between(v2,v1))

    for v22 in v2.reshape((-1,3)):
        print(angle_between(v22,v1))

def __va_main():
    d = FisheyeDS_Description(
        width=1000,
        height=1000,
        intrinsics=dict(afov = 200, a = 0.5, xi = -0.3),
        extrinsic_rot=[0,0,0]
    )

    print(d.get_afov())

    # d = Perspective_Description(
    #     width=1000,
    #     height=1000,
    #     intrinsics=dict(afov=90),
    #     extrinsic_rot=[0,0,0]
    # )

    va_vec = d.get_va_vector()
    print(np.degrees(va_vec[-1])*2)


#    print(d.get_va_vector().shape)


def __proj_main():

    xyz = np.random.default_rng().random((10,3))
    xyz = np.array([[0,0,1]])
    print(xyz)
    print(FisheyeDS_Description.radial_proj_func(xyz, f = 100, a_ = 0.5, xi_ = -0.3))

    xyz = np.random.default_rng().random((10,3))
    print(xyz)
    print(FisheyeUCM_Description.radial_proj_func(xyz, f = 100, a_ = 0.5))

if __name__=="__main__":
     # __va_main()
    ds = FisheyeWakai_Description(width=100, height=100, intrinsics=dict(f=10, k=-.1 ))
    print(ds.get_afov())
    print(ds.get_va_vector())
    
