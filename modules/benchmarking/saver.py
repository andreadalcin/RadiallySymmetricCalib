from dataclasses import dataclass, field
from typing import Dict
from datatypes.consts import *
import os
from utils.util import join_paths
from datatypes.richtypes import Match, RichImage
import cv2 as cv
import json
import pandas as pd

@dataclass
class Saver():

    bmk_config:dict = field(init=False)

    run_data_conf: str = field(init=False)
    run_test_name: str = field(init=False)
    run_data_name: str = field(init=False)
    run_method_name:str = field(init=False)

    results_dfs:Dict[str,Dict[str,pd.DataFrame]] = field(init=False, default_factory=lambda: {})

    def __post_init__(self):
        with open(BMK_CONFIG_FILE) as bmk_jf:
            self.bmk_config = json.load(bmk_jf)

        for folder, _, files in os.walk(self.bmk_config[BENCH_BASE_PATH]):
            for file in files:
                if RESULT_DF_FILE in file:
                    metric = file.replace(RESULT_DF_FILE,"")
                    self.results_dfs[join_paths(folder,metric)] = pd.read_csv(join_paths(folder,file), header=[0,1], index_col=0)

    def get_exp_dir(self):
        return join_paths(
            self.bmk_config[BENCH_BASE_PATH],
            self.run_test_name,
            self.run_data_conf)

    def save_match(self, match: Match):
        if not self.bmk_config[SAVE_MATCHES]:
            return
        img_match_o, img_match_t = match.get_match_imgs()

        dest_dir = join_paths(
            self.get_exp_dir(),
            self.run_data_name,
            self.run_method_name,
            self.bmk_config[BENCH_MATCHES])

        os.makedirs(dest_dir, exist_ok=True)

        cv.imwrite(join_paths(dest_dir, f"{self.run_data_name}_{match}_o{EXTENSION}"), img_match_o)
        cv.imwrite(join_paths(dest_dir, f"{self.run_data_name}_{match}_t{EXTENSION}"), img_match_t)

    def show_img_keypoints(self, rich_img: RichImage):

        transf_out = cv.drawKeypoints(rich_img.img_transf.img, keypoints=rich_img.rich_transf.kps, outImage=None, color=(0, 0, 255),
                                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        transf_out2 = cv.drawKeypoints(rich_img.img_transf.img_rot90, keypoints=rich_img.rich_transf.kps_rot90, outImage=None, color=(0, 0, 255),
                                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


        base_out = cv.drawKeypoints(rich_img.img_base.img, keypoints=rich_img.rich_base.kps, outImage=None, color=(0, 0, 255),
                                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        base_out2 = cv.drawKeypoints(rich_img.img_base.img_rot90, keypoints=rich_img.rich_base.kps_rot90, outImage=None, color=(0, 0, 255),
                                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        cv.imshow("Base Image",base_out)
        cv.imshow("Base Image Rot",base_out2)
        cv.imshow("Transformed Image",transf_out)
        cv.imshow("Transformed Image Rot",transf_out2)
        cv.waitKey(0)
        cv.destroyAllWindows() 

    def save_match_gt(self, match: Match):
        if not self.bmk_config[SAVE_MATCHES_GT]:
            return

        for i in range(len(match.gt_metrics)):
            self.save_match_gt_th(match, i)
            if self.bmk_config[SAVE_ONLY_LOW_TH]:
                break
        
    def save_match_gt_th(self, match: Match, th_index):
        img_match_o, img_match_t = match.get_gt_match_imgs(th_index=th_index)
        th = match.gt_metrics[th_index].error_th
        
        dest_dir = join_paths(
            self.get_exp_dir(),
            self.run_data_name,
            self.run_method_name,
            self.bmk_config[BENCH_MATCHES_GT])

        os.makedirs(dest_dir, exist_ok=True)

        cv.imwrite(join_paths(dest_dir, f"{self.run_data_name}_{match}_o_th_{th}{EXTENSION}"), img_match_o)
        cv.imwrite(join_paths(dest_dir, f"{self.run_data_name}_{match}_t_th_{th}{EXTENSION}"), img_match_t)

    def save_df_results(self, df):
        key = join_paths(self.get_exp_dir(),df.name)

        if key not in self.results_dfs.keys():
            self.results_dfs[key] = df
        else:
            self.results_dfs[key] = self.update_df(self.results_dfs[key], df)

    def update_df(self, df_base:pd.DataFrame, df_new:pd.DataFrame) -> pd.DataFrame:
        for i in list(df_new.index):
            if i in df_base.index:
                # Check if method column exists
                df_base = self.update_df_by_column(df_base, df_new, index=i)
            else:
                # Append row
                df_base = pd.concat([df_base,df_new.loc[i:i]],axis = 0)
        return df_base

    def update_df_by_column(self, df_base:pd.DataFrame, df_new:pd.DataFrame, index) -> pd.DataFrame:
        for c in list(df_new.columns):
            if c in df_base.columns:
                # Update values
                df_base.loc[index,c] = df_new.loc[index,c]
            else:
                # Append column
                df_base = pd.concat([df_base,df_new.loc[index:index,c:c]],axis = 1)
        return df_base

    def save_on_disk(self):
        for key, df in self.results_dfs.items():
            df.to_csv(key + RESULT_DF_FILE)
        #TODO save test_conf

SAVER = Saver()