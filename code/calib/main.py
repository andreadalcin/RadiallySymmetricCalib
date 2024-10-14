from __future__ import annotations
import sys
import os
import json
from ruamel import yaml
import importlib, inspect

from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import Type, List, Union, Tuple

# Prpject path setup
_root_path = Path(__file__).resolve() / "../../../"
_root_path = _root_path.resolve()
sys.path.append(str(_root_path / "code/calib"))
sys.path.append(str(_root_path / "modules"))

from frameworks.abs_framework import TrainBase, TestBase
from misc.misc import Tupperware

TENSORBOARD_SCRIPT_FILE = "tboard.sh"

def printj(dic):
    return print(json.dumps(dic, indent=4))

def collect_args(args_location) -> Namespace:
    """Set command line arguments"""
    parser = ArgumentParser()
    parser.add_argument('--test', action='store_true', dest='is_test', \
        help="If it is testing phase")
    parser.add_argument('--config', type=Path,\
        help="Config file", default=Path(__file__).parent / "data/params.yaml")
    parser.add_argument('--saved', type=Path, \
        help="Saved model folder")
    parser.add_argument('--flog', action='store_true', dest='force_new_log_path', \
        help="Force new log folder")
    parser.add_argument('--train', action='store_true', dest='is_train', \
        help="If it is training phase")
    parser.add_argument('-d','--dataset', type=str, dest='dataset', \
        help="Which preconfigured dataset to use. If unspecified the one in the config is used.")
    parser.add_argument('-s','--suff', type=str, default=None, dest='suffix', \
        help="Sets the test suffix")
    parser.add_argument('-e','--eval', action='store_true', dest='eval', \
        help="Combined with test args, not gt, no batch, and save images.")

    parser.set_defaults(is_test=False)
    parser.set_defaults(is_train=False)
    parser.set_defaults(eval=False)
    parser.set_defaults(force_new_log_path=False)
    args = parser.parse_args(args_location)
    configure_saved_args(args)
    return args

def configure_saved_args(config: Namespace):
    if config.saved is not None:
        config.config = config.saved / "models" / "params.yaml"

def collect_tupperware(config: Path) -> Tupperware:
    params = yaml.safe_load(open(config))
    args = Tupperware(params)
    return args

DATASETS = [
    'new_kitti360', 
    #'new_sun360',
    'ds_sun360',
    'new_woodscape',
    'new_silda_test',
    ]

DATASETS_E2 = [
    'WS-E2', 
    'SILDa-E2',
    'Kitti-E2',
    'final_sun360',
    ]

def preset_dataset(dataset:str, args:Tupperware):
    if dataset is None:
        return
        
    if dataset not in DATASETS + DATASETS_E2:
        raise ValueError(f'The dataset configuration {dataset} does not exist!')
    
    c = dict(
            dataset_dir= f'data/{dataset}/',
            train_file= f'data/{dataset}/train.txt',
            val_file= f'data/{dataset}/val.txt',
            test_file= f'data/{dataset}/test.txt',
            test_img_out_path= f'benchmark/calibration/{dataset}/images/',
            test_param_out_path= f'benchmark/calibration/{dataset}/params/',
            output_directory= f'output/{dataset}/',
        )

    for key, val in c.items():
        args[key] = val

def get_train_test_model_classes(model_name:str, model_classes:List[Type[TestBase]]) -> Tuple[Type[TrainBase], Type[TestBase]]:
    test_suff = "Test" 
    train_suff = "Train"

    train_model = None
    test_model = None

    query_model_name = "".join(model_name.split("_")).lower()
    
    for model_class in model_classes:
        model_class_name_split = model_class.__name__.split("_")
        model_class_name = "".join(model_class_name_split[:-1])
        model_suffix = model_class_name_split[-1]

        if query_model_name == model_class_name.lower():
            if model_suffix == test_suff:
                test_model = model_class
            elif model_suffix == train_suff:
                train_model = model_class

    if train_model is None or test_model is None:
        raise ValueError(f"Invalid model name. The model {model_name} is not defined.")
    return train_model, test_model

def get_classes_from_module(module_name):
    module_classes = inspect.getmembers(importlib.import_module(module_name),inspect.isclass)
    # Filter classes not defined in that module
    module_classes = [clss for _,clss in module_classes if clss.__module__ == module_name ]
    return module_classes

def get_model_classes(args: Tupperware) -> Tuple[Type[TrainBase], Type[TestBase] ]:
    models_classes = get_classes_from_module("impl_models") 
    model_classes = get_train_test_model_classes(args.model_name, models_classes)
    return model_classes

def gen_tensorboard_script(log_path:Path):
    command = f"tensorboard --logdir {str(log_path)}"
    with open(Path(TENSORBOARD_SCRIPT_FILE),"w") as file:
        file.write(command)

def main(args_location = sys.argv[1:]):
    config = collect_args(args_location)
    
    args = collect_tupperware(config.config)


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices or "-1"

    # TODO DEBUG
    # import torch
    # torch.autograd.set_detect_anomaly(True)


    train_model_class, test_model_class = get_model_classes(args)

    if config.suffix is not None:
        args.test_suff = config.suffix
        
    if config.is_train:
        preset_dataset(config.dataset, args)
        printj(args)
        model = train_model_class(args, force_new_log_path=config.force_new_log_path)
        gen_tensorboard_script(model.log_path)
        model.start_train()

    if config.is_test:
        args.batch_size = 1
        args.num_workers =1
        if config.eval:
            args.no_gt = True
            args.test_save_rectified_imgs = True
            args.test_save_calibration_params = False
            args.test_visualize_batch_details = False
            
        if config.dataset == 'all':
            datasets = DATASETS
        elif config.dataset == 'e2':
            datasets = DATASETS_E2
        else:
            datasets = [config.dataset]

        for dataset_name in datasets:
            preset_dataset(dataset_name, args)
            printj(args)
            model = test_model_class(args)
            gen_tensorboard_script(model.log_path)
            model.start_test()

if __name__ == "__main__":
    main()

    # python code/calib/main.py --config code/calib/params/ --train
