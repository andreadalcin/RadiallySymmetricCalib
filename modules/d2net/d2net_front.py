from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import torch
import cv2 as cv
import os
from d2net.model_test import D2Net
from d2net.d2_utils import preprocess_image
from d2net.pyramid import process_multiscale

@dataclass
class D2FrontEnd():
    preprocessing:str = 'caffe'

    model_file:str = 'd2_tf.pth'

    max_edge:int = 1600
    max_sum_edges:int = 2800
    multiscale:bool = True
    use_relu:bool = True

    # CUDA
    use_cuda:bool = field(init=False,default=torch.cuda.is_available())
    device:torch.device = field(init=False, default=torch.device("cuda:0" if use_cuda else "cpu"))

    model:torch.nn.Module = field(init=False, default=None)

    def __post_init__(self):
        # Creating CNN model
        model_file = os.path.join(os.path.dirname(__file__),self.model_file)
        self.model = D2Net(
            model_file=model_file,
            use_relu=self.use_relu,
            use_cuda=self.use_cuda
        )

    # Process the file
    def run(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        resized_image = image
        if max(resized_image.shape) > self.max_edge:
            scale =self.max_edge / max(resized_image.shape)
            resized_image = cv.resize(resized_image, dsize=None, fx=scale, fy = scale).astype('float')
        if sum(resized_image.shape[: 2]) > self.max_sum_edges:
            scale = self.max_sum_edges / sum(resized_image.shape[: 2])
            resized_image = cv.resize(resized_image, dsize=None, fx=scale, fy = scale).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=self.preprocessing
        )
        with torch.no_grad():
            if self.multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=self.device
                    ),
                    self.model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=self.device
                    ),
                    self.model,
                    scales=[1]
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]
        return keypoints, scores, descriptors
