import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from metrics.metric_utils import RunningMean, Mean
from typing import List, Union
from projections.cameras import ImageDescription

@dataclass
class CalibrationMetric(ABC):
    name:str
    lower_is_better:bool = True
    des_cache: ImageDescription = field(init=False,default=None)

    def evaluate(self, a:Union[np.ndarray,ImageDescription], b:Union[np.ndarray,ImageDescription]) -> float:
        if isinstance(a,ImageDescription):
            return self.eval_des(a, b)
        
        else:
            return self.eval_static(a, b)
    
    @classmethod
    @abstractmethod
    def eval_static(self, pts_a:np.ndarray, pts_b:np.ndarray) -> float:
        pass
        
    @classmethod
    @abstractmethod
    def eval_des(self, des_a:ImageDescription, des_b:ImageDescription) -> float:
        pass


@dataclass
class MRE_L1(CalibrationMetric):
    """Mean L1 reprojection error"""
    name: str = "MRE_L1"
    lower_is_better:bool = True
    ref_points:np.ndarray = field(init=False)
    ref_pts_y:np.ndarray = field(init=False)
  
    def eval_static(self, pts_a:np.ndarray, pts_b:np.ndarray) -> float:
        return np.mean(np.abs(pts_a-pts_b))
    
    
    def rescale_intrinsics(self, ref_des:ImageDescription, des:ImageDescription):
        fact = ref_des.height / des.height
        if fact != 1:
            des.f *= fact
            des.height = ref_des.height
            des.width = ref_des.width
        
    def eval_des(self, des_a:ImageDescription, des_b:ImageDescription) -> float:
        if self.des_cache is None or self.des_cache != des_a:
            self.des_cache = des_a.copy()
            self.des_cache.extrinsic_rot = [0,0,0]
            self.ref_pts_y = np.arange(0,self.des_cache.height//2) + (self.des_cache.width-1)/2
            pts_x = np.repeat((self.des_cache.width-1)/2, self.ref_pts_y.shape[0])
            self.ref_points = self.des_cache.image2world([pts_x, self.ref_pts_y])[:3]     
        else:
            print('mai')   
        
        des_b = des_b.copy()
        des_b.extrinsic_rot = [0,0,0]

        self.rescale_intrinsics(self.des_cache, des_b)

        _, proj_y, mask = des_b.world2image(self.ref_points)
        return self.eval_static(self.ref_pts_y[mask], proj_y[mask])
    
@dataclass
class MRE_L2(MRE_L1):
    """Mean L1 reprojection error"""
    name: str = "MRE_L2"
    lower_is_better:bool = True

    def eval_static(self, pts_a:np.ndarray, pts_b:np.ndarray) -> float:
        return np.mean((pts_a-pts_b)**2)    

@dataclass
class VA_L1(CalibrationMetric):
    """Mean reprojection error"""
    name: str = "VA_L1"
    lower_is_better: bool = True
    ref_va: np.ndarray = field(init=False, default=None)
    
    def eval_des(self, des_a:ImageDescription, des_b:ImageDescription) -> float:
        if self.des_cache is None or self.des_cache != des_a:
            self.des_cache = des_a
            self.ref_va = des_a.get_va_vector()
        
        return self.eval_static(self.ref_va, des_b.get_va_vector())
        
    def eval_static(self, pts_a:np.ndarray, pts_b:np.ndarray) -> float:
        return np.mean(np.abs((pts_a-pts_b)))
    
@dataclass
class VA_L2(VA_L1):
    """Mean reprojection error"""
    name: str = "VA_L2"

    def eval_static(self, pts_a:np.ndarray, pts_b:np.ndarray) -> float:
        return np.mean((pts_a-pts_b)**2)

@dataclass
class MeanCalibration(Mean):

    calibration_metric : CalibrationMetric

    def initialize(self, a:Union[np.ndarray,ImageDescription], b:Union[np.ndarray,ImageDescription]):
        val = self.calibration_metric.evaluate(a, b)
        super().initialize(val)

    def update(self, a:Union[np.ndarray,ImageDescription], b:Union[np.ndarray,ImageDescription]):
        if not self.initialized:
            self.initialize(a, b)
        else:
            self.add(a, b)

    def add(self, a:Union[np.ndarray,ImageDescription], b:Union[np.ndarray,ImageDescription]):
        val = self.calibration_metric.evaluate(a, b)
        super().add(val)

    def var(self):
        return np.mean(np.asarray(self.vals)**2) - self.average()**2

    def stdev(self):
        return np.sqrt(self.var()) 

    def __repr__(self) -> str:
        return f"Mean {self.calibration_metric.name} : {self.average():.3f}"