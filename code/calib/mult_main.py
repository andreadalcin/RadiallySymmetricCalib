import os
import subprocess
from pathlib import Path
import itertools
from typing import Dict, List, Tuple
from ruamel import yaml
from tqdm.auto import tqdm
import shutil

base_path = Path(__file__).parent / "../../"
base_path = base_path.resolve()
params_path = base_path / "code" / "calib" / "params"

def build_configurations() -> Dict[str,List]:
    return dict(
        #input_height = [1000, 800],
        #input_width = [1000, 800],
        # polar_width = [
        #     800,
        #     600
        # ],
        model_name = [
            'resize_2step_polar_huber',
            'resize_gap_polar_huber',
            ],

        network_fe_layers = [3,4],
        network_fc_layers = [3,4],
       # loss_huber_delta =  [0.3, 0.03]
    )

def get_configurations_dict() -> Tuple[List[str], List[Tuple]]:
    configs = build_configurations()
    keys = list(configs.keys())
    linekd = {}#'input_height': 'input_width'}
    key_names = list(set(keys) - set(linekd.values()))

    params_tuple_iter = list(itertools.product(*[configs[c] for c in key_names]))
    params_tuple_list = []
    for params_tuple in params_tuple_iter:
        for link_s, link_c in linekd.items():

            # source elem index in tuple
            idx_s = key_names.index(link_s)
            # source value in current tuple
            value_s = params_tuple[idx_s]
            # index of value in source list
            idx_sv = configs[link_s].index(value_s)
            # corresponding copy value
            value_c = configs[link_c][idx_sv]
            params_tuple += (value_c,)

        params_tuple_list.append(params_tuple)

    key_names.extend(linekd.values())
    return key_names, params_tuple_list

def main(base_params: Path, params_path:Path):
    key_names, params_tuple_list = get_configurations_dict()
    base_params = yaml.safe_load(open(params_path / base_params))

    gen_params_dir = params_path / "gen"
    shutil.rmtree(gen_params_dir, ignore_errors=True)
    os.makedirs(gen_params_dir, exist_ok=True)

    param_file_name = "gen_param_{}.yaml"
    for i, params_tuple in tqdm(enumerate(params_tuple_list), total=len(params_tuple_list)):

        for key, value in zip(key_names, params_tuple):
            base_params[key] = value

        param_file = gen_params_dir / param_file_name.format(i)
        with open(param_file, 'w') as f:
            yaml.dump(base_params, f)

        subprocess.run([
            "python", 
            str(base_path / "code" / "calib"/ "main.py"),
            "--config",
            str(param_file),
            "--train",
            "--test",
            "--flog"
            ])

if __name__ == "__main__":
    base_params = "gap_polar.yaml"
    main(base_params, params_path)