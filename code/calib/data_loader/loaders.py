from data_loader.generic_loader import GenericDataset
from data_loader.fast_loader import FixedSizeDataset
import torch.utils.data as data

def get_dataset(data_path=None, path_file=None, is_train=False, config=None) -> data.Dataset:
    if config.loader == "fast":
        return FixedSizeDataset(
            data_path=data_path,
            config=config,
            is_train=is_train,
            path_file=path_file,
        )

    else:
        return GenericDataset(
            data_path=data_path,
            config=config,
            is_train=is_train,
            path_file=path_file,
        )