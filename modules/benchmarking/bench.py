from dataclasses import InitVar, dataclass, field
import cv2 as cv
import numpy as np
import pandas as pd
from datatypes.basetypes import BaseImage
from benchmarking.bench_config import TestConfig
from datatypes.consts import DATA_DESCRIPTION_FILE
from datatypes.richtypes import GtEvaluator, Match, RichImage
import projections.mappings as map
from projections.cameras import ImageDescription
import os
import json
from typing import List, Tuple
from benchmarking.saver import SAVER
from tqdm.auto import tqdm
from benchmarking import LOGGER

def build_tests(data_folder, data_conf_file:str):
    # file containing the image description
    img_description_file = os.path.join(data_folder, DATA_DESCRIPTION_FILE)
    # file containing the run test configuration
    run_test_config = os.path.join(data_folder, data_conf_file)
    conf_file = data_conf_file.split(".")[0]

    tests = []

    with open(img_description_file) as img_des_js:
        img_des_dict = json.load(img_des_js)
        imgs_des = ImageDescription.from_des_dict(img_des_dict)

    with open(run_test_config) as run_test_jf:
        conf = json.load(run_test_jf)
        for test in conf['tests']:
            tests.append(MatchingBenchmark(**test, test_name=conf['name'], conf_file=conf_file, imgs_des=imgs_des))
    
    return tests


@dataclass
class MatchingBenchmark():
    test_name: str
    name: str
    pairs: List[Tuple[int,int]]
    imgs_des : ImageDescription
    
    gt_paths: InitVar[List[str]]
    img_paths: InitVar[List[str]]

    conf_file: str
    
    imgs        : List[BaseImage] = field(init=False, default=None)
    gt_evaluator: GtEvaluator     = field(init=False, default=None)
    test_config : TestConfig      = field(init=False, default=None)

    rich_imgs_src: List[RichImage] = field(init=False, default=None)
    rich_imgs_dst: List[RichImage] = field(init=False, default=None)

    matches: List[Match] = field(init=False, default=None)

    matcher: cv.DescriptorMatcher = field(init=False, default=None)

    def __post_init__(self, gt_paths, img_paths):
        self.imgs = [ BaseImage( path=path, db_index=i, img_des=self.imgs_des) for i,path in enumerate(img_paths) ]
        self.gt_evaluator = GtEvaluator.from_img_type(self.imgs_des)(gt_paths)

    def __init_test(self, test_config: TestConfig):
        for img in self.imgs:
            img.load()
        self.gt_evaluator.load()
        self.rich_imgs_src = []
        self.rich_imgs_dst = []
        self.matches = []

        self.test_config = test_config
        self.test_config.update(original_des = self.imgs_des)

        self.matcher = cv.BFMatcher(normType=test_config.detector_src.matcher_norm())

        SAVER.run_data_conf = self.conf_file
        SAVER.run_test_name = self.test_name
        SAVER.run_data_name = self.name
        SAVER.run_method_name = self.test_config.name

    def start(self, test_config : TestConfig):
        LOGGER.debug(f"Starting initialization...")
        self.__init_test(test_config)
        LOGGER.debug(f"Initalization completed")

        LOGGER.debug(f"Starting mapping...")
        # Modify the input image as required by the test configurations
        for img in self.imgs:
            LOGGER.debug(f" - Mapping image {img.path}")
            img_trasf_src, des_list = map.map_img(img=img, des_list = [self.test_config.img_des_src, self.test_config.descriptor_des_src])
            self.rich_imgs_src.append(RichImage(img_base=img, img_transf=img_trasf_src, inter_dess=des_list[1:-1]))

            img_trasf_dst, des_list = map.map_img(img=img, des_list = [self.test_config.img_des_dst, self.test_config.descriptor_des_dst])
            self.rich_imgs_dst.append(RichImage(img_base=img, img_transf=img_trasf_dst, inter_dess=des_list[1:-1]))
        LOGGER.debug(f"Mapping completed")

        LOGGER.debug(f"Starting keypoint estraction...")
        for img_src, img_dst in zip(self.rich_imgs_src, self.rich_imgs_dst):
            # Run the detectors on the transformed images
            LOGGER.debug(f" - Extracting image {img_src.img_base.path}")
            img_src.set_rich_trasf(rich_transf=test_config.detector_src.rich_detect_and_describe(img_src.img_transf))
            #SAVER.show_img_keypoints(img_src)
            img_dst.set_rich_trasf(rich_transf=test_config.detector_dst.rich_detect_and_describe(img_dst.img_transf))
            #SAVER.show_img_keypoints(img_dst)
        LOGGER.debug(f"Keypoint estraction completed")

        LOGGER.debug(f"Starting matching...")
        # perform matching between the given pairs
        for pair in self.pairs:
            src, dst = pair

            # Perform descriptor matching using the original image keypoints and descirptors
            matches = self.__match_dess( des1 = self.rich_imgs_src[src].rich_base.kps_des, des2 = self.rich_imgs_dst[dst].rich_base.kps_des)
            match = Match(img_a=self.rich_imgs_src[src], img_b=self.rich_imgs_dst[dst], matches=matches)
            self.matches.append(match)  
            
            LOGGER.debug(f" - performed: {match}, mathces: {len(match.matches)}")
            SAVER.save_match(match=match)
        LOGGER.debug(f"Matching completed")

    def __match_dess(self, des1, des2):
        """ Perform descriptor matching with Lowe ratio test bewteen two sets of descriptors.
        """
        # Match descriptors.
        matches = self.matcher.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < self.test_config.ratio_test_th * n.distance:
                good.append([m])

        return good

    def evaluate_matches(self, inlier_ths: List[int] = [0.01, 0.05, 0.1]):
        LOGGER.debug("Starting evaluation...")
        self.gt_evaluator.evaluate_matches(matches = self.matches, inlier_ths=inlier_ths)
        LOGGER.debug("Saving evaluation...")
        for match in self.matches:
            SAVER.save_match_gt(match=match)
        LOGGER.debug("Evaluation completed")

def gen_df_from_matches(matches: List[Match], test_name:str, metric_fun ,match_prefix: str = "", df_name = ""):
    df = None
    for m in matches:
        ths = [str(metric.error_th) for metric in m.gt_metrics]
        vals = [metric_fun(metric) for metric in m.gt_metrics]

        df_ = pd.DataFrame(
            data = np.array([vals]),
            index = pd.Index([f"{match_prefix} {m}"], name="Pair:"),
            columns = pd.MultiIndex.from_product([[test_name],ths], names=["Config:","Thresholds:"]),
            dtype = float)

        df = df_ if df is None else pd.concat([df, df_])

    df.name = df_name
    return df

def run_configs(test : MatchingBenchmark, test_configs: List[TestConfig], thresholds):
    with tqdm(test_configs, leave=False) as t_bar:
        for config in t_bar:
            LOGGER.info(f"Config {config.name} [{t_bar.n}/{t_bar.total}] starting...")
            t_bar.set_description(f"Conf: {config.name}")

            test.start(config)
            test.evaluate_matches(inlier_ths=thresholds)
            df_ratio = gen_df_from_matches(
                test.matches, 
                test_name=config.name,
                metric_fun= lambda x : x.inlier_ratio,
                match_prefix=test.name,
                df_name="in_ratio")

            df_count = gen_df_from_matches(
                test.matches, 
                test_name=config.name,
                metric_fun= lambda x : x.inlier_count,
                match_prefix=test.name,
                df_name="in_count")
            
            SAVER.save_df_results(df=df_ratio)
            SAVER.save_df_results(df=df_count)

def run_tests_configs(tests : List[MatchingBenchmark], test_configs: List[TestConfig], thresholds):
    with tqdm(tests) as t_bar:
        for test in t_bar:
            LOGGER.info(f"Test {test.name} [{t_bar.n}/{t_bar.total}] starting...")
            t_bar.set_description(f"Test: {test.name}")

            run_configs(test = test, test_configs = test_configs, thresholds = thresholds)
            SAVER.save_on_disk()
            LOGGER.debug(f"Test {test.name} completed and stored")
            del test


if __name__ == "__main__":
    print(build_tests(data_folder="data/matching/perspective",bench_config_file="benchmark/bench_conf.json", data_conf_file="conf_small.json"))