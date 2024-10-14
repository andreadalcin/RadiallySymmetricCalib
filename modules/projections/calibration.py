from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.optimize import differential_evolution
from dataclasses import dataclass, field
from typing import Tuple, List
import cv2 as cv
from projections.cameras import *
import warnings


@dataclass
class Calibration():

    i_rho:np.ndarray 
    """ Contains the distance between image points and image center.
        The image center will be (0,0).
        Shape: `(n, )`, where `n` is the number of points.
    """

    w_xyz:np.ndarray
    """ Contains the respective values of X,Y,Z of the world points.

        Shape: `(n, 3)`, where `n` is the number of points.
        `w_xyz[:,0] -> x coordinates`
        `w_xyz[:,1] -> y coordinates`
        `w_xyz[:,2] -> z coordinates`
    """

    camera:ImageDescription

    sigma: Optional[np.ndarray] = None
    """ Contains the stddev associated with each point correspodence.
        Shape: `(n, )`, where `n` is the number of points.
    """

    # @staticmethod
    # def from_va_map(va_map: np.ndarray, camera:ImageDescription) -> Calibration:
    #     ref_p = np.split(np.array([[0,0,1]]), 3, axis=-1)
        
    #     # get central column values of the va_map
    #     c_x = va_map.shape[1]//2
    #     c_y = va_map.shape[0]//2

    #     vals = np.array([va_map[y,c_x] for y in range(c_y+1)])
    #     mask = ~np.isnan(vals)
    #     vals = vals[mask]
        
    #    # take only the i_y since the i_x will be zero.
    #     i_xy = (- np.arange(0,c_y+1) + c_y)[mask]

    #     w_vectors = np.array([apply_rotation(ref_p, angles=[np.rad2deg(rad),0,0]) for rad in vals])
    #     w_xy = w_vectors[:,1] # take only the w_Y since the w_x will be zero.
    #     w_z = w_vectors[:,2]

    #     return Calibration(i_xy=i_xy, w_xy=w_xy, w_z = w_z, camera=camera)

    @staticmethod
    def from_va_vec(va_vec: np.ndarray, camera:ImageDescription, sigma:Optional[np.ndarray] = None) -> Calibration:
        """_summary_

        Args:
            va_vec (np.ndarray): A 1D-array containing angles in radiants for each of the image pixels starting from the center.
                For instance, va_vec[0] should containg the viewing angle at the center location of the image. 
            camera (ImageDescription): _description_

        Returns:
            Calibration: _description_
        """
        n = va_vec.shape[0]

        mask = ~np.isnan(va_vec)
        va_vec = va_vec[mask]

        # i_rho will be zero.
        i_rho = np.arange(0,n)[mask]

        return Calibration.from_rho_angles(i_rho=i_rho, w_angles=va_vec, camera=camera, sigma=sigma)
    

    @staticmethod
    def from_rho_angles(i_rho: np.ndarray, w_angles:np.ndarray, camera:ImageDescription, sigma:Optional[np.ndarray] = None) -> Calibration:
       
        assert i_rho.shape == w_angles.shape
        assert len(i_rho.shape) == 1

        w_xyz = np.zeros((w_angles.shape[0],3))

        # Apply viweing angles to center rays
        w_xyz[:,1] = np.sin(w_angles)
        w_xyz[:,2] = np.cos(w_angles)

        return Calibration(i_rho=i_rho, w_xyz=w_xyz, camera=camera, sigma=sigma)


    # function for genetic algorithm to minimize (sum of squared error)
    def sumOfSquaredError(self, parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = self.camera.radial_proj_func(self.w_xyz, *parameterTuple)
        if self.sigma is not None:
            return np.sum(((self.i_rho - val)/self.sigma) ** 2.0)
        return np.sum((self.i_rho - val) ** 2.0)

    def run(self, method = "ls", loss='cauchy') -> dict:
        # generate initial parameter values
        parameterBounds = self.camera.get_instrinsics_bounds()
        geneticParameters = differential_evolution(
            self.sumOfSquaredError, 
            parameterBounds, 
            popsize=5,
            maxiter=100,
            seed=3).x
       
        #curve fit the test data
        #geneticParameters = [200,.5,0]

        tuple_bounds = ([pb[0] for pb in parameterBounds], [pb[1] for pb in parameterBounds] )

        try:
            if method == 'curve':
                fittedParameters, pcov = curve_fit(
                    self.camera.radial_proj_func, 
                    self.w_xyz, 
                    self.i_rho, 
                    geneticParameters, 
                    bounds= tuple_bounds,
                    sigma=self.sigma)
            elif method == 'ls':
                def func(params, w_xyz, i_rho, sigma):
                    if sigma is None:
                        return self.camera.radial_proj_func(w_xyz, *params) - i_rho
                    return (self.camera.radial_proj_func(w_xyz, *params) - i_rho)/sigma
                
                try:
                    self.camera.jacobian(0)
                except NotImplementedError:
                    warnings.warn("Jacobian not implemented!")
                    jac = '2-point'
                except TypeError:
                    def jac(params, w_xyz, i_rho, sigma):
                        return self.camera.jacobian(w_xyz, *params)

                fittedParameters = least_squares(
                    func, 
                    x0=geneticParameters,
                    bounds= tuple_bounds,
                    loss=loss,
                    jac=jac,
                    kwargs=dict(
                        w_xyz = self.w_xyz,
                        i_rho =self.i_rho,
                        sigma = self.sigma,
                    )
                      ).x
            else:
                raise NotImplementedError(f"Method {method} is not implemented!")
        except RuntimeError:
            fittedParameters = geneticParameters
            warnings.warn('Least square fitting failed!')
        intrinsics = dict(zip(self.camera.intrinsic_names[0], fittedParameters))
        return intrinsics
        

def __main_fishDS():
    from matplotlib import pyplot as plt
    from projections.mappings import map_img

    img = cv.cvtColor(cv.imread("./data/matching/equirectangular/white_chapel_afternoon/Image0013.png"),cv.COLOR_BGR2RGB)
    img_des = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    out_des = FisheyeDS_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=dict(a=0.55, xi=-0.23, f = 565), 
        extrinsic_rot=[0,0,0])
    out_des2 = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    
    
    out_img,_ = map_img(img, des_list=[img_des, out_des])
    f, axarr = plt.subplots(2,3)
    axarr[0,0].imshow(img)
    axarr[1,0].imshow(out_img)
    axarr[0,1].imshow(out_des.get_last_va_map())
    

    cal = Calibration.from_va_map(out_des.get_last_va_map(), camera=FisheyeDS_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=dict(a=0, xi=0, f = 1000), 
        extrinsic_rot=[0,0,0]))

    estimated = cal.run()
    print(estimated)
    est_des = FisheyeDS_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=estimated, 
        extrinsic_rot=[0,0,0])
    est_img,_ = map_img(img, des_list=[img_des, est_des])
    axarr[1,1].imshow(est_img)
    axarr[1,2].imshow(cv.bitwise_xor(out_img, est_img))
    ax= f.add_subplot(2,3,3,projection='3d')
    ax.plot(cal.w_xy, cal.w_z, zs=cal.i_xy, zdir='z', label='curve')

    idxs = np.linspace(start=0,stop=cal.w_xy.shape[0]-1,num=min(cal.w_xy.shape[0]-1, 50), dtype=int)
    ax.scatter(cal.w_xy[idxs], cal.w_z[idxs], zs=cal.i_xy[idxs], zdir='z', c='r', label='points', marker='2')
    axarr[0,2].set_axis_off()
    # Make legend, set axes limits and labels
    ax.legend()
    plt.show()
    

def __main_vec_fishDS():
    from matplotlib import pyplot as plt
    from projections.mappings import map_img

    img = cv.cvtColor(cv.imread("./data/matching/equirectangular/white_chapel_afternoon/Image0013.png"),cv.COLOR_BGR2RGB)
    img_des = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    out_des = FisheyeDS_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=dict(a=0.55, xi=-0.23, f = 565), 
        extrinsic_rot=[0,0,0])
    out_des2 = Equirectangular_Description(width=img.shape[1], height=img.shape[0])
    
    out_img,_ = map_img(img, des_list=[img_des, out_des])
    f, axarr = plt.subplots(2,3)
    axarr[0,0].imshow(img)
    axarr[1,0].imshow(out_img)
    axarr[0,1].imshow(out_des.get_last_va_map())

    cal = Calibration.from_va_vec(out_des.get_va_vector(), camera=FisheyeDS_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=dict(a=0, xi=0, f = 1000), 
        extrinsic_rot=[0,0,0]))
        
    estimated = cal.run()
    print(estimated)
    est_des = FisheyeDS_Description(
        width=img.shape[1], 
        height=img.shape[1],
        intrinsics=estimated, 
        extrinsic_rot=[0,0,0])
    est_img,_ = map_img(img, des_list=[img_des, est_des])
    axarr[1,1].imshow(est_img)
    axarr[1,2].imshow(cv.bitwise_xor(out_img, est_img))
    ax= f.add_subplot(2,3,3,projection='3d')
    ax.plot(cal.w_xy, cal.w_z, zs=cal.i_xy, zdir='z', label='curve')

    idxs = np.linspace(start=0,stop=cal.w_xy.shape[0]-1,num=min(cal.w_xy.shape[0]-1, 50), dtype=int)
    ax.scatter(cal.w_xy[idxs], cal.w_z[idxs], zs=cal.i_xy[idxs], zdir='z', c='r', label='points', marker='2')
    axarr[0,2].set_axis_off()
    # Make legend, set axes limits and labels
    ax.legend()
    plt.show()


def _test_cal_fishDS():
    from matplotlib import pyplot as plt
    from projections.mappings import map_img

    out_des = FisheyeDS_Description(
        width=1001, 
        height=1001,
        intrinsics=dict(a=0.55, xi=-0.3, f = 565), 
        extrinsic_rot=[0,0,0])
    
    va_vec = out_des.get_va_vector()
   # print(va_vec)

    cal = Calibration.from_va_vec(va_vec, camera=FisheyeDS_Description)
        
    estimated = cal.run()
    print(estimated)
    est_des = FisheyeDS_Description(
        width=1000, 
        height=1000,
        intrinsics=estimated, 
        extrinsic_rot=[0,0,0])
   # print(est_des.get_va_vector())



def _test_cal_fishUCM():
    from matplotlib import pyplot as plt
    from projections.mappings import map_img

    out_des = FisheyeUCM_Description(
        width=1000, 
        height=1000,
        intrinsics=dict(a=0.5, f = 565), 
        extrinsic_rot=[0,0,0])
    
    va_vec = out_des.get_va_vector()
    #print(va_vec)

    cal = Calibration.from_va_vec(va_vec, camera=FisheyeUCM_Description)
        
    estimated = cal.run()
    print(estimated)
    est_des = FisheyeUCM_Description(
        width=1000, 
        height=1000,
        intrinsics=estimated, 
        extrinsic_rot=[0,0,0])
    #print(est_des.get_va_vector())

if __name__ == "__main__":
    _test_cal_fishDS()

    
