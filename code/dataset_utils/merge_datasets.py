import os
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from typing import Union, List
import warnings
import shutil

start_alph = 65 + 10

def main(datasets:List[Path], out_dir = Path):
    os.makedirs(out_dir)
    out_dir_images = out_dir/'rgb_images'
    out_dir_calibration = out_dir/'calibration'
    os.makedirs(out_dir_images)
    os.makedirs(out_dir_calibration)
    for i,dataset in enumerate(datasets):
        dataset_images = dataset /'rgb_images'
        dataset_calibration = dataset /'calibration'
        images = os.listdir(dataset_images )
        print(dataset, len(images))
        
        for image in images:
            name = image.split('.')[0]
            new_name= name + chr(start_alph + i)
            
            os.symlink((dataset_images / image).absolute(), (out_dir_images / (new_name + '.png')).absolute())
            os.symlink((dataset_calibration / (name + '.json')).absolute(), (out_dir_calibration / (new_name + '.json')).absolute())
            
    print(len(os.listdir(out_dir_images )))
        
        
def main_files(datasets:List[Path], out_dir = Path, filename='train.txt'):
    # if out_dir.exists():
    #     warnings.warn('The folder exists, continue?')
    #     input()
    
    os.makedirs(out_dir, exist_ok=True)
    out_dir_images = out_dir/'rgb_images'
    out_dir_calibration = out_dir/'calibration'
    os.makedirs(out_dir_images, exist_ok=True)
    os.makedirs(out_dir_calibration, exist_ok=True)
    new_files = []
    
    for i,dataset in enumerate(datasets):
        dataset_images = dataset /'rgb_images'
        dataset_calibration = dataset /'calibration'
        
        images = os.listdir(dataset_images )
        with open(dataset / filename) as f:
            files = [line.rstrip('\n') for line in f]
        
        print(dataset, len(images), len(files))
        
        for name in tqdm(files):
            
            #name = image.split('.')[0]
            new_name= name + chr(start_alph + i)
            new_files.append(new_name)
            
            shutil.copy((dataset_images / (name + '.png')).absolute(), (out_dir_images / (new_name + '.png')).absolute())
            shutil.copy((dataset_calibration / (name + '.json')).absolute(), (out_dir_calibration / (new_name + '.json')).absolute())
            # os.symlink((dataset_images / (name + '.png')).absolute(), (out_dir_images / (new_name + '.png')).absolute())
            # os.symlink((dataset_calibration / (name + '.json')).absolute(), (out_dir_calibration / (new_name + '.json')).absolute())
            
    with open(out_dir / filename, 'a') as f:
        f.write("\n".join(new_files))

    print(len(os.listdir(out_dir_images )))
        
if __name__ == '__main__':
    main_files([
        # Path('data','sub00_kitti360'),
        # Path('data','sub06_kitti360'),
        # Path('data','sub09_kitti360'),
        # Path('data','final_silda_train'),
        # Path('data','final_sun360'),
        Path('data','WS-T1'),
    ],
    Path('data','final'),
    filename='val.txt',
    )
    
    # images = set(map(lambda x:x[:-4], os.listdir('data/final/rgb_images')))
    # print(len(images))
    
    # vals= set([line.rstrip('\n') for line in open('data/final/val.txt')])
    # print(len(vals))
    
    
    # print(len(images-vals))
    
    # with open('data/final/train.txt', 'w') as f:
    #     f.write("\n".join(images-vals))
    