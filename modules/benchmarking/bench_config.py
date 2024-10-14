
from dataclasses import dataclass, field
from typing import Optional
from datatypes.basetypes import ImageDescription
from utils.fdds import FDD

@dataclass
class TestConfig():

    name : str
    detector_src: FDD 
    img_des_src : Optional[ImageDescription] = None
    detector_dst : Optional[FDD] = None
    img_des_dst : Optional[ImageDescription] = None
    ratio_test_th : Optional[float] = 0.8
    descriptor_des_src : ImageDescription = field(init=False, default=None)
    descriptor_des_dst : ImageDescription = field(init=False, default=None)
    

    def __post_init__(self):
        if self.detector_dst == None:
            self.detector_dst = self.detector_src

   
        """Running configuration for a test. This is used to specify how the source and destination images should be modified in order to be matched. 
        It is also mandatory to specify the descriptors taht should be used for both the src and destination images. If the descriptors are not compatible an exception will be thrown.

        Args:
            name (str): name of the test configuration.
            detector_src (fdds.FDD): FDD to be used on the src image.
            detector_dst (fdds.FDD, optional): FDD to be used on the dst image. If none the src descriptor will be used.
            test_des_src (map.ImageDescription, optional): ImageDescriptor object that specifies how the src image should be transformed before appliying the FDD. If None the iamge will be left unchanged.
            test_des_dst (map.ImageDescription, optional): ImageDescriptor object that specifies how the dst image should be transformed before appliying the FDD. If None the iamge will be left unchanged.
            ratio_test_th (float, optional): threshold for the Lowe ratio test to be used in the matching phase. Defaults to 0.8.
        """

    def __detector_req(self, img_des:ImageDescription, detector: FDD):
        assert img_des is not None

        descriptor_des = detector.input_des()
        if descriptor_des is None:
            return None

        if descriptor_des.is_compatible(img_des):
            img_des.combine(descriptor_des)

        return descriptor_des

    def __update(self, img_des:ImageDescription, detector: FDD, original_des:ImageDescription):
        if img_des is None:
            img_des = original_des.copy()
        else:
            img_des = img_des.copy()

        descriptor_des = self.__detector_req(img_des, detector)

        return img_des, descriptor_des

    def update(self, original_des:ImageDescription):
        assert self.detector_dst.matcher_norm() == self.detector_src.matcher_norm()

        self.img_des_src, self.descriptor_des_src = self.__update(self.img_des_src, self.detector_src, original_des=original_des)
        self.img_des_dst, self.descriptor_des_dst = self.__update(self.img_des_dst, self.detector_dst, original_des=original_des)

