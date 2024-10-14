from __future__ import annotations
import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Type
from datatypes.basetypes import BaseImage, RichInfo
from dataclasses import dataclass, field
from datatypes.consts import COLOR_MATCH_FAR, COLOR_MATCH_IN, COLOR_MATCH_OUT

from projections.mappings import reproject_kps
from projections.cameras import ImageDescription, Equirectangular_Description, Perspective_Description
from utils.util import remove_indxs_from_list

@dataclass
class RichImage():
    """Wrapper for a BaseImage adding informaiton retrieved after running a feature detection and description algorithm on it.
    """
    img_base : BaseImage
    img_transf : BaseImage

    # Descriptors of intermediate transformations to move form base image to transf image
    inter_dess: List[ImageDescription] = field(default_factory=lambda: [])

    rich_transf : RichInfo = field(init=False)
    rich_base : RichInfo = field(init=False)

    def set_rich_trasf(self, rich_transf: RichInfo) -> None:
        """Function to set the rich information fo the transformed image after reprojecting and filtering the keypoint in the base image.
        This function also sets the rich information of the base image.

        Args:
            rich_transf (RichInfo): Rich information of the transfomred image.
        """
        self.rich_transf = rich_transf
        base_kps, bad_idxs = reproject_kps(rich_transf.kps, des_list=[self.img_transf.img_des, *self.inter_dess, self.img_base.img_des])

        # filtering 
        remove_indxs_from_list(self.rich_transf.kps, indxs=bad_idxs)
        remove_indxs_from_list(self.rich_transf.kps_rot90, indxs=bad_idxs)
        self.rich_transf.kps_des = np.delete(self.rich_transf.kps_des,bad_idxs,axis=0)

        # build base rich info
        self.rich_base = RichInfo(kps=base_kps, kps_des=self.rich_transf.kps_des, width=self.img_base.img_des.width)


#######################################
######      Match classes       #######
#######################################

@dataclass
class MatchGtMetrics():
    error_th : float
    mask_inlier : np.ndarray
    mask_far : np.ndarray = None
    mask_outlier : np.ndarray = field(init=False, default = None)
    
    inlier_ratio : float = field(init=False, default = 0.0)
    inlier_count : int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.mask_inlier is None:
            return

        if self.mask_far is None:
            self.mask_far = self.mask_inlier & False

        self.mask_inlier = self.mask_inlier & ~self.mask_far
        self.mask_outlier = ~self.mask_inlier & ~self.mask_far

        self.inlier_count = np.count_nonzero(self.mask_inlier)
        self.inlier_ratio = self.inlier_count / np.count_nonzero(~self.mask_far)

        self.mask_far = self.mask_far.astype(int)
        self.mask_inlier = self.mask_inlier.astype(int)
        self.mask_outlier = self.mask_outlier.astype(int)

class MatchGtMetricsNoMatches(MatchGtMetrics):
    def __init__(self, error_th:float):
        super().__init__(error_th=error_th, mask_inlier=None, mask_far=None)

class MatchGtMetricsAllFar(MatchGtMetrics):
    def __init__(self, error_th:float, mask_far:np.ndarray):
        super().__init__(error_th=error_th, mask_inlier=None, mask_far=mask_far.astype(int))
        
@dataclass
class Match():
    img_a: RichImage
    img_b: RichImage
    matches : List[List[cv.DMatch]]
    gt_metrics: List[MatchGtMetrics] = field(init=False, default_factory=lambda: [])

    def __str__(self) -> str:
        return f"({self.img_a.img_base.db_index}-{self.img_b.img_base.db_index})"

    def get_match_imgs(self):
        """ Return two matches images, the first displaying the matches bewteen the images in the original configuration,
         and the second siplaying the matches in the transformed configuration.
        """
        img_match_o = cv.drawMatchesKnn(
            img1=self.img_a.img_base.img_rot90,
            img2=self.img_b.img_base.img_rot90,
            keypoints1= self.img_a.rich_base.kps_rot90,
            keypoints2= self.img_b.rich_base.kps_rot90,
            matches1to2=self.matches,
            outImg=None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

        img_match_t = cv.drawMatchesKnn(
            img1=self.img_a.img_transf.img_rot90,
            img2=self.img_b.img_transf.img_rot90,
            keypoints1=self.img_a.rich_transf.kps_rot90,
            keypoints2=self.img_b.rich_transf.kps_rot90,
            matches1to2=self.matches,
            outImg=None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

        img_match_o = cv.rotate(img_match_o, cv.ROTATE_90_CLOCKWISE)
        img_match_t = cv.rotate(img_match_t, cv.ROTATE_90_CLOCKWISE)

        return img_match_o, img_match_t

    @staticmethod
    def draw_gt_matches_rot( img1, kp1, img2, kp2, matches, mask_far, mask_inlier, mask_outlier):
        img = cv.drawMatches(img1, kp1, img2, kp2, matches,None,matchesMask=mask_far,   matchColor=COLOR_MATCH_FAR, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.drawMatches(img1, kp1, img2, kp2,matches,outImg=img,matchesMask=mask_inlier, matchColor=COLOR_MATCH_IN,  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS + cv.DrawMatchesFlags_DRAW_OVER_OUTIMG )
        cv.drawMatches(img1, kp1, img2, kp2,matches,outImg=img,matchesMask=mask_outlier,matchColor=COLOR_MATCH_OUT, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS + cv.DrawMatchesFlags_DRAW_OVER_OUTIMG )
        
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        return img

    def get_gt_match_imgs(self, th_index = 0):
        """ Return two matches images, the first displaying the matches bewteen the images in the original configuration,
         and the second siplaying the matches in the transformed configuration. The color of the matches indicates the inliers and the outliers.
        """
        
        matches = [m[0] for m in self.matches] # (to use DrawMatches and not DrawMAtchesKnn)

        gt_metric = self.gt_metrics[th_index]

        img_match_o = self.draw_gt_matches_rot(
            img1=self.img_a.img_base.img_rot90,
            img2=self.img_b.img_base.img_rot90,
            kp1=self.img_a.rich_base.kps_rot90,
            kp2=self.img_b.rich_base.kps_rot90,
            matches=matches,
            mask_far=gt_metric.mask_far, 
            mask_inlier=gt_metric.mask_inlier, 
            mask_outlier=gt_metric.mask_outlier)
        img_match_t = self.draw_gt_matches_rot(
            img1=self.img_a.img_transf.img_rot90,
            img2=self.img_b.img_transf.img_rot90,
            kp1=self.img_a.rich_transf.kps_rot90,
            kp2=self.img_b.rich_transf.kps_rot90,
            matches=matches,
            mask_far=gt_metric.mask_far, 
            mask_inlier=gt_metric.mask_inlier, 
            mask_outlier=gt_metric.mask_outlier)

        return img_match_o, img_match_t

    def get_src_pts(self) -> np.ndarray:
        return np.array([ np.array(self.img_a.rich_base.kps[m[0].queryIdx].pt) for m in self.matches])

    def get_dst_pts(self) -> np.ndarray:
        return np.array([ np.array(self.img_b.rich_base.kps[m[0].trainIdx].pt) for m in self.matches])



#######################################
###### Ground Truth Evaluation  #######
#######################################

@dataclass
class GtEvaluator(ABC):

    gt_paths: List[str]

    @staticmethod
    def from_img_type(img_type: ImageDescription) -> Type[GtEvaluator]:
        return GT_EVALUTAOR_FACTORY[type(img_type).__name__]

    @abstractmethod
    def load(self) -> None:
        """Loads the ground trouths from file into memory
        """

    @abstractmethod
    def evaluate_matches(self, matches:List[Match], inlier_ths:List[float]) -> None:
        """ Evaluates the matches against a ground truth, producing metrics for each threshold level.
            Metrics are stored in the match object.
        """

@dataclass
class PerspectiveGtEvaluator(GtEvaluator):

    hms: List[np.ndarray] = field(init=False, default=None)

    def load(self):
        self.hms = [np.genfromtxt(h_name) for h_name in self.gt_paths]

    def evaluate_matches(self, matches:List[Match], inlier_ths:List[float]) -> None:
        for gt_h, match in zip(self.hms, matches):
            self.__evaluate_match(match, gt_h, inlier_ths) 

    def __evaluate_match(self, match:Match, gt_h: np.ndarray, inlier_ths:List[float]) -> None:
        pt1s = match.get_src_pts()
        pt2s = match.get_dst_pts()

        if len(pt1s.shape) != 2:
            match.gt_metrics = [MatchGtMetricsNoMatches(error_th=th) for th in inlier_ths]
            return

        pt1s = np.concatenate((pt1s, np.ones((pt1s.shape[0],1))) ,axis=-1 )
        pt_gt = (gt_h @ pt1s.T).T
        #normalize points
        pt_gt = (pt_gt / pt_gt[:,2:])[:,:2]
        errors = np.sqrt(np.sum(np.power(pt_gt-pt2s,2),axis=1))

        match.gt_metrics = []
        for inlier_th in inlier_ths:
            # normalization of the threshold based on hte resoluztion of the image TODO is it ok?
            th = inlier_th * match.img_a.img_base.img_des.width
            mask_inlier = errors < th

            match.gt_metrics.append(MatchGtMetrics(inlier_th, mask_inlier=mask_inlier))

@dataclass
class EquirectangularGtEvaluator(GtEvaluator):

    gts: List[np.ndarray] = field(init=False, default=None)

    def load(self):
        self.gts = [np.load(gt_name) for gt_name in self.gt_paths]

    def evaluate_matches(self, matches:List[Match], inlier_ths:List[float]) -> None:
        for match in matches:
            self.__evaluate_match(match, self.gts[match.img_a.img_base.db_index], self.gts[match.img_b.img_base.db_index], inlier_ths)

    def __evaluate_match(self, match:Match, gt1: np.ndarray, gt2: np.ndarray, inlier_ths:List[float]) -> None:
        pt1s = match.get_src_pts().astype(int)
        pt2s = match.get_dst_pts().astype(int)

        if len(pt1s.shape) != 2:
            match.gt_metrics = [MatchGtMetricsNoMatches(error_th=th) for th in inlier_ths]
            return

        pt1_gt = gt1[pt1s[:,1],pt1s[:,0],:]
        pt2_gt = gt2[pt2s[:,1],pt2s[:,0],:]

        mask_far = (pt1_gt[:,3] == 0) + (pt2_gt[:,3] == 0)
        mask_far_count = np.sum(mask_far)
        # All the points are non matchable
        if mask_far_count == len(match.matches):
            match.gt_metrics = [MatchGtMetricsAllFar(error_th=th, mask_far=mask_far) for th in inlier_ths]
            return
        
        errors = np.sqrt(np.sum(np.power(pt1_gt-pt2_gt,2),axis=-1))
        for inlier_th in inlier_ths:
            mask_inlier = errors < inlier_th
            match.gt_metrics.append(MatchGtMetrics(inlier_th, mask_inlier=mask_inlier, mask_far=mask_far))

GT_EVALUTAOR_FACTORY = {
    Perspective_Description.__name__: PerspectiveGtEvaluator,
    Equirectangular_Description.__name__: EquirectangularGtEvaluator,
}

#######################################
######           Main           #######
#######################################

def __main():
    img_type = Equirectangular_Description(width=0,height=0)

    print(GtEvaluator.from_img_type(img_type=img_type))


if __name__ == "__main__":
    __main()