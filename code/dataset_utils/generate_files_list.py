import os
import numpy as np
import kitti360.kitti360_utils as ku
import sun360.sun360_utils as su
import woodscape.woodscape_util as wu
import silda.silda_util as si
from tqdm.auto import tqdm
from pathlib import Path
from typing import Union

def main(gen_name_parser: ku.GenNameParser, n_test = 200, n_val = 200, n_train = 1000,
    data_dir : Union[Path,str] = None,
    train_file:Union[Path,str] = None, val_file:Union[Path,str] = None, test_file:Union[Path,str] = None):
    image_list = os.listdir(data_dir)
    
    if isinstance(n_test, int):
        if n_train is None:
            n_samples = len(image_list)
            p_train = None
        else:
            n_samples = n_test + n_val + n_train
            p_train = n_train / n_samples
        p_test  = n_test  / n_samples
        p_val   = n_val   / n_samples
    else:
        p_test  = n_test
        p_train = n_train
        p_val   = n_val
        n_test = p_test*len(image_list)
        n_val = p_val*len(image_list)
        if p_train is not None:
            n_train = p_train*len(image_list)

    idxs = set(map(lambda x: gen_name_parser.set_filename(x).unique_name, image_list))
    n_idxs = len(idxs)

    idxs_test  = set(np.random.choice(list(idxs), int(p_test * n_idxs), replace=False))
    idxs_val   = set(np.random.choice(list(idxs - idxs_test) , int(p_val * n_idxs), replace=False))
    if p_train is None:
        idxs_train = list(idxs - idxs_test - idxs_val)
    else:
        idxs_train = set(np.random.choice(list(idxs - idxs_test - idxs_val), int(p_train * n_idxs), replace=False))

    name_list  = list(map(lambda x: gen_name_parser.set_filename(x).filename, image_list))
    name_train = list(filter(lambda x: gen_name_parser.set_filename(x).unique_name in idxs_train, name_list))
    name_val   = list(filter(lambda x: gen_name_parser.set_filename(x).unique_name in idxs_val, name_list))
    name_test  = list(filter(lambda x: gen_name_parser.set_filename(x).unique_name in idxs_test, name_list))

    if p_train is not None:
        s_val   = np.random.choice(name_val, n_val, replace=False)
        s_test  = np.random.choice(name_test, n_test, replace=False)
        s_train = np.random.choice(name_train, n_train, replace=False)
    else:
        s_val   = name_val
        s_test  = name_test
        s_train = name_train

    with open(train_file,"w") as f:
        f.write("\n".join(s_train))

    with open(val_file,"w") as f:
        f.write("\n".join(s_val))

    with open(test_file,"w") as f:
        f.write("\n".join(s_test))

    print("Done!")


def kitti360_main(n_test = 1087, n_val = 0, n_train = 0, data_path = None):
    if data_path is None:
        data_path = ku.base_path / "data/gen_kitti360"
    data_dir   = data_path / 'rgb_images'
    train_file = data_path / "train.txt"
    val_file   = data_path / "val.txt"
    test_file  = data_path / "test.txt"

    main(
        gen_name_parser=ku.KittiGenNameParser(),
        n_test=n_test,
        n_train=n_train,
        n_val=n_val,
        data_dir=data_dir,
        train_file=train_file,
        test_file=test_file,
        val_file=val_file,
    )

def sun360_main(n_test = 1036, n_val = 0, n_train = 0, data_path = None):
    if data_path is None:
        data_path = ku.base_path / "data/gen_sun360"
    data_dir   = data_path / 'rgb_images'
    train_file = data_path / "train.txt"
    val_file   = data_path / "val.txt"
    test_file  = data_path / "test.txt"

    main(
        gen_name_parser=su.SunGenNameParser(),
        n_test=n_test,
        n_train=n_train,
        n_val=n_val,
        data_dir=data_dir,
        train_file=train_file,
        test_file=test_file,
        val_file=val_file,
    )

def woodscape_main(n_test = 1036, n_val = 0, n_train = 0, data_path = None):
    if data_path is None:
        data_path = ku.base_path / "data/gen_woodscape"
    data_dir   = data_path / 'rgb_images'
    train_file = data_path / "train.txt"
    val_file   = data_path / "val.txt"
    test_file  = data_path / "test.txt"

    main(
        gen_name_parser=wu.WoodscapeGenNameParser(),
        n_test=n_test,
        n_train=n_train,
        n_val=n_val,
        data_dir=data_dir,
        train_file=train_file,
        test_file=test_file,
        val_file=val_file,
    )

def generic_main(gen_name_parser, data_path, n_test = 0, n_val = 0, n_train = 0):
    data_dir   = data_path / 'rgb_images'
    train_file = data_path / "train.txt"
    val_file   = data_path / "val.txt"
    test_file  = data_path / "test.txt"

    main(
        gen_name_parser=gen_name_parser,
        n_test=n_test,
        n_train=n_train,
        n_val=n_val,
        data_dir=data_dir,
        train_file=train_file,
        test_file=test_file,
        val_file=val_file,
    )

if __name__ == "__main__" :
    n_test = 0.0
    n_val = 0.1
    n_train = None
    generic_main(
        gen_name_parser=wu.WoodscapeGenNameParser(),
        n_test = n_test, 
        n_val = n_val, 
        n_train = n_train, 
        data_path = si.base_path / "data/WS-T1",
        )