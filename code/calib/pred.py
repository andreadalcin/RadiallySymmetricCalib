import os
from pathlib import Path
from frameworks.abs_framework import TestBase
from main import  collect_tupperware, get_model_classes, _root_path


def load_predictor(config: Path) -> TestBase:
    
    args = collect_tupperware(config)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices or "-1"

    args.batch_size = 1
    args.no_gt = True
    args.test_save_rectified_imgs = False
    args.test_save_calibration_params = False
    args.test_visualize_batch_details = False
    train_model_class, test_model_class = get_model_classes(args)

    return test_model_class(args)

if __name__ == '__main__':
    config_path = Path(_root_path / 'output/final/convnext_mix_feat_2023-04-08_03-09-50/models/params.yaml')
    pred = load_predictor(config=config_path)
    
    import cv2 as cv
    img = cv.imread('data/real/LectureA/LectureA_0001.png')
    pred.predict(img)