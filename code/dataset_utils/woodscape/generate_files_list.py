import os
import json
from woodscape_util import *
from tqdm.auto import tqdm

def parse_path(path):
    if isinstance(path, str):
        file_name = path.split("/")[-1]
    elif isinstance(path, Path):
        file_name = path.name
    else:
        raise ValueError("Not supported")

    seq_idx_side = file_name.split(".")[0]
    seq_idx,side = seq_idx_side.split('_')
    seq,idx = seq_idx.split('-')
    return seq_idx_side, int(seq), int(idx), side

OUT_DIR = Path("data/gen_woodscape_barrel")
OUT_IMG_DIR   = OUT_DIR / 'rgb_images'
TRAIN_FILE = OUT_DIR / "train.txt"
VAL_FILE   = OUT_DIR / "val.txt"
TEST_FILE  = OUT_DIR / "test.txt"

def main(n_test = 200, n_val = 200, n_train = 1000):
    image_list = os.listdir(OUT_IMG_DIR)
    
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

    idxs = set(map(lambda x: parse_path(x)[2], image_list))
    n_idxs = len(idxs)

    idxs_test  = set(np.random.choice(list(idxs), int(p_test * n_idxs), replace=False))
    idxs_val   = set(np.random.choice(list(idxs - idxs_test) , int(p_val * n_idxs), replace=False))
    if p_train is None:
        idxs_train = list(idxs - idxs_test - idxs_val)
    else:
        idxs_train = set(np.random.choice(list(idxs - idxs_test - idxs_val), int(p_train * n_idxs), replace=False))

    name_list  = list(map(lambda x: parse_path(x)[0], image_list))
    name_train = list(filter(lambda x: parse_path(x)[2] in idxs_train, name_list))
    name_val   = list(filter(lambda x: parse_path(x)[2] in idxs_val, name_list))
    name_test  = list(filter(lambda x: parse_path(x)[2] in idxs_test, name_list))

    if p_train is not None:
        s_val   = np.random.choice(name_val, n_val, replace=False)
        s_test  = np.random.choice(name_test, n_test, replace=False)
        s_train = np.random.choice(name_train, n_train, replace=False)
    else:
        s_val   = name_val
        s_test  = name_test
        s_train = name_train

    with open(TRAIN_FILE,"w") as f:
        f.write("\n".join(s_train))

    with open(VAL_FILE,"w") as f:
        f.write("\n".join(s_val))

    with open(TEST_FILE,"w") as f:
        f.write("\n".join(s_test))

    print("Done!")

if __name__ == "__main__" :
    n_test = .1
    n_val = .1
    n_train = None
    main(n_test = n_test, n_val = n_val, n_train = n_train)