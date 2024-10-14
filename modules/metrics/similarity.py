import numpy as np
from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim, \
    mean_squared_error as mse, peak_signal_noise_ratio as psnr
from metrics.metric_utils import RunningMean, Mean
from typing import List

@dataclass
class SimilarityMetric(ABC):
    name:str
    lower_is_similar:bool = True

    @abstractclassmethod
    def evaluate(self, img_a:np.ndarray, img_b:np.ndarray) -> float:
        pass

@dataclass
class MSE(SimilarityMetric):
    name: str = "MSE"
    lower_is_similar:bool = True

    def evaluate(self, img_a:np.ndarray, img_b:np.ndarray) -> float:
        return mse(img_a, img_b)

@dataclass
class SSIM(SimilarityMetric):
    name:str = "SSIM"
    lower_is_similar:bool = False

    def evaluate(self, img_a:np.ndarray, img_b:np.ndarray) -> float:

        channel_axis = None

        if len(img_a.shape) == 3:
            channel_axis = -1

        mssim = ssim(img_a, img_b, channel_axis=channel_axis)

        return mssim

@dataclass
class PSNR(SimilarityMetric):
    name:str = "PSNR"
    lower_is_similar:bool = False

    def evaluate(self, img_a:np.ndarray, img_b:np.ndarray) -> float:
        return psnr(img_a, img_b)

@dataclass
class RunningMeanSimilarity(RunningMean):

    similarity_metric : SimilarityMetric

    def initialize(self, img_a:np.ndarray, img_b:np.ndarray):
        val = self.similarity_metric.evaluate(img_a, img_b)
        super().initialize(val)

    def update(self, img_a:np.ndarray, img_b:np.ndarray):
        if not self.initialized:
            self.initialize(img_a, img_b)
        else:
            self.add(img_a, img_b)

    def add(self, img_a:np.ndarray, img_b:np.ndarray):
        val = self.similarity_metric.evaluate(img_a, img_b)
        super().add(val)

    def __repr__(self) -> str:
        return f"Mean {self.similarity_metric.name} : {self.average():.3f}"

@dataclass
class MeanSimilarity(Mean):

    similarity_metric : SimilarityMetric

    def initialize(self, img_a:np.ndarray, img_b:np.ndarray):
        val = self.similarity_metric.evaluate(img_a, img_b)
        super().initialize(val)

    def update(self, img_a:np.ndarray, img_b:np.ndarray):
        if not self.initialized:
            self.initialize(img_a, img_b)
        else:
            self.add(img_a, img_b)

    def add(self, img_a:np.ndarray, img_b:np.ndarray):
        val = self.similarity_metric.evaluate(img_a, img_b)
        super().add(val)

    def get_rankings_average(self, top_k:List[float], names=List[str]):
        self.vals.sort(reverse = not self.similarity_metric.lower_is_similar)
        sorted_val = list(self.vals)
        tot_elems = len(sorted_val)

        # Remove last percentile if they already sum up to 1, since the last value is precisely evaluated
        # to avoid leaving out elemnts. Otherwise the sum of the number of samples is not ensure to be 
        # equal to the total.
        if sum(top_k) == 1:
            top_k.pop()

        rankings = {}
        for i, top_k_th in enumerate(top_k):
            rankings[names[i]] = np.mean(sorted_val[:int(tot_elems*top_k_th)])
            sorted_val = sorted_val[int(tot_elems*top_k_th):]

        # Evaluate ranking for the last bucket, containing the remaining items.
        if sum(top_k) < 1:
            rankings[names[-1]] = np.mean(sorted_val)

        return rankings

    def __repr__(self) -> str:
        return f"Mean {self.similarity_metric.name} : {self.average():.3f}"
