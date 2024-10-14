from __future__ import annotations
import sys
import os, json
import time
import datetime
import numpy as np
import pandas as pd
import shutil
import torch

from torch.utils.data import DataLoader
from colorama import Fore, Style
from ruamel import yaml
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from PIL import Image
from typing import List, Dict, Tuple, Optional, Type, ClassVar
from distutils.util import strtobool

import data_loader.loaders as lds
from misc import tools
from misc.misc import Tupperware, build_c2p_grid
from metrics.metric_utils import RunningMean
from projections import calibration as cal
from projections import mappings as mp

TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"



# TODO COMMENTS!

class EarlyStopper:
    def __init__(self, patience:int=5, min_delta:float=1e-4, starting_epoch:int=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.starting_epoch = starting_epoch 

    def early_stop(self, validation_loss:float):
        if self.starting_epoch > 0:
            self.starting_epoch -= 1
            return False

        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter > self.patience:
                return True
        return False


@dataclass
class TrainTestBase(ABC):
    args:Tupperware
    log_path:Path = field(init=False)
    pretrained_weights_path:Path = field(init=False)
    model_name:str = field(init=False)
    device:str = field(init=False)
    models:Dict[str,torch.nn.Module]  = field(init=False, default_factory=lambda : {})
    start_time:float  = field(init=False, default=time.time())
    writers: Dict[str, SummaryWriter] = field(init=False, default_factory=lambda :{})
    num_total_steps: int = field(init=False)
    criterions_definiton: Dict[str, torch.nn.Module] = field(init=False)
    cart2polar_grid: torch.Tensor = field(init=False)
    polar2cart_grid: torch.Tensor = field(init=False)

    MAX_AFOV: ClassVar[float] = 140

    def __post_init__(self):
        self.device = self.args.device
        self.model_name = self.args.model_name
        self.pretrained_weights_path = self._compose_pretrained_weights()

        self.cart2polar_grid = build_c2p_grid(
            input_height=self.args.input_height,
            input_width=self.args.input_width,
            polar_width=self.args.polar_width,
            batch_size=self.args.batch_size,
            v2=True,
        ).to(self.device)

        self._init_model()
        self._init_dataloaders()
        self._init_losses()

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def _init_dataloaders(self):
        pass

    @abstractmethod
    def _init_losses(self):
        pass

    def _inputs_to_device(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

    def _set_train(self):
        """Convert all models to training mode"""
        for m in self.models.values():
            m.train()

    def _set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for m in self.models.values():
            m.eval()

    def _load_model(self):
        """Load model(s) from disk"""

        assert self.pretrained_weights_path.exists(), f"Cannot find folder {self.pretrained_weights_path}"
        print(f"=> Loading model from folder {str(self.pretrained_weights_path)}")

        discard_param_group = self.args.discard_param_group

        def to_discard(k:str):
            for discard in discard_param_group:
                if k.startswith(discard):
                    return True
            return False

        for n in self.args.models_to_load:
            print(f"Loading {n} weights...")
            path = self.pretrained_weights_path / f"{n}.pth"
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() \
                               if k in model_dict and not to_discard(k) }
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def _compose_pretrained_weights(self) -> Optional[Path]:
        try:
            pretrained_weights_path = Path(self.args.pretrained_weights)
        except TypeError:
            return None

        assert pretrained_weights_path.exists(), "The pretrain weigths path does not exist!"

        # If the given folder already contains model files, it is the right folder.
        for file in pretrained_weights_path.iterdir():
            if file.name.endswith(".pth"):
                return pretrained_weights_path

        try:
            # If higher directory is given, try to get the models for the given epoch
            pretrained_weights_path = pretrained_weights_path / "models"

            epochs_weights = os.listdir(pretrained_weights_path)
            last_epoch = max([int(epoch_weights.split("_")[-1]) \
                for epoch_weights in epochs_weights \
                    if epoch_weights.startswith("weights_")])

            if not self.args.epoch_to_load:
                self.args.epoch_to_load = last_epoch

            pretrained_weights_path = pretrained_weights_path / f"weights_{self.args.epoch_to_load}"
            pretrained_weights_path = pretrained_weights_path / os.listdir(pretrained_weights_path)[0]

        except (FileNotFoundError, IndexError):
            print(f"Weights path not found : \'{str(pretrained_weights_path)}\'! \n" \
                "If a higher directory for the pretrained weights is given, it should be structured like: "\
                ".\\models\\weigths_$epoch$\\$steps$\\checkpoint_files")
            raise FileNotFoundError(filename = str(pretrained_weights_path))
    
        return pretrained_weights_path

    def sec_to_hm(self, t):
        """Convert time in seconds to time in hours, minutes and seconds
        e.g. 10239 -> (2, 50, 39)
        """
        t = int(t)
        s = t % 60
        t //= 60
        m = t % 60
        t //= 60
        return t, m, s

    def sec_to_hm_str(self, t):
        """Convert time in seconds to a nice string
        e.g. 10239 -> '02h50m39s'
        """
        h, m, s = self.sec_to_hm(t)
        return f"{h:02d}h{m:02d}m{s:02d}s"

@dataclass
class TrainBase(TrainTestBase):
    parameters_to_train:List  = field(init=False, default_factory=lambda : [])
    optimizer: torch.optim.Optimizer = field(init=False)
    lr_scheduler:object = field(init=False)
    epoch: int  = field(init=False, default=0)
    step: int  = field(init=False, default=0)
    early_stopper: EarlyStopper = field(init=False)

    val_metrics: Dict[str, RunningMean] = field(init=False, default_factory = lambda : {})
    best_val_metric : float = field(init=False, default=None)

    reference_val_metric : str = field(init=False)
    reference_train_metric : str = field(init=False)

    train_loader: DataLoader = field(init=False)
    val_loader: DataLoader = field(init=False)
    force_new_log_path: bool = False

    criterions_definiton = {
        "mse_loss":torch.nn.MSELoss(),
       # "huber_loss":torch.nn.HuberLoss(),
        }

    def __post_init__(self):
        super().__post_init__()

        self.log_path = self.__compose_log_path()

        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(self.log_path / mode)

        # Number of batches times number of epochs.
        self.num_total_steps = len(self.train_loader) * self.args.epochs

        print(f"=> Training on the {self.args.dataset_dir.upper()} dataset \n"
              f"=> Training model named: {self.args.model_name} \n"
              f"=> Models and tensorboard events files are saved to: {str(self.log_path)} \n"
              f"=> Training is using the cuda device id: {self.args.cuda_visible_devices} \n"
              )

        print(f"=> Total number of training examples: {len(self.train_loader)*self.train_loader.batch_size} \n"
              f"=> Total number of validation examples: {len(self.val_loader)*self.val_loader.batch_size}")

        self._configure_optimizers()

        if self.pretrained_weights_path is not None:
            if strtobool(input(f"=> Load weights from ({self.pretrained_weights_path})?")):
                self._load_model()
                self.__load_optimizer()

        # Store the currently trained weights
        self.args.pretrained_weights = str(self.log_path)

        # LOSSES
        self.reference_val_metric = self.args.reference_val_metric
        self.reference_train_metric = self.args.reference_train_metric

        self.early_stopper = EarlyStopper(
            patience=self.args.patience, 
            min_delta=self.args.min_delta,
            starting_epoch=self.args.starting_epoch)

        self.__save_args()

        for val_metric_name in self.criterions_definiton.keys():
            self.val_metrics[f"{val_metric_name}_val"] = RunningMean()

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def _init_losses(self):
        pass
        #self.criterions_definiton["huber_loss"].delta = self.args.loss_huber_delta

    def _init_dataloaders(self):
        # --- Load Data ---
        train_dataset = lds.get_dataset(data_path=self.args.dataset_dir,
                                            path_file=self.args.train_file,
                                            is_train=True,
                                            config=self.args)

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       num_workers=self.args.num_workers,
                                       pin_memory=True,
                                       drop_last=True,
                                       )

        val_dataset = lds.get_dataset(data_path=self.args.dataset_dir,
                                          path_file=self.args.val_file,
                                          is_train=False,
                                          config=self.args)

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=True,
                                     num_workers=self.args.num_workers,
                                     pin_memory=True,
                                     drop_last=True,
                                     collate_fn=val_dataset.collate_fn)
                                     

    def __compose_log_path(self) -> Path:
        log_path = Path(self.args.output_directory) / self.model_name
        
        # Training time: choose to overwrite the last log direcotry or to create a new one.
        time_stamp = datetime.datetime.now().strftime(TIME_FORMAT)
        new_log_path = log_path.parent / f"{log_path.name}_{time_stamp}"
        
        last_log_path = self.__get_last_log_path()
        if last_log_path is not None:
            if not self.force_new_log_path and strtobool(input(f"=> Re-use last log dir? ({last_log_path.name})")):
                shutil.rmtree(last_log_path, ignore_errors=False, onerror=None)
                os.makedirs(last_log_path, exist_ok=True)
                print("=> Cleaned up the logs!")
                return last_log_path
            
            print(f"=> Creating a new log dir! ({new_log_path.name})")
            os.makedirs(new_log_path, exist_ok=True)
            return new_log_path
        
        print(f"=> No pre-existing directories found for this experiment. \n"
            f"=> Creating a new one! ({new_log_path.name})")
        os.makedirs(new_log_path, exist_ok=True)
        return new_log_path

    def __get_last_log_path(self) -> Path:
        logs_dir = Path(self.args.output_directory)
        last_dt = None
        last_folder = None

        if not logs_dir.exists():
            return None

        for folder in logs_dir.iterdir():
            f_split = folder.name.split("_")
            model_name = "_".join(f_split[:-2])
            time_stamp = "_".join(f_split[-2:])
            date_time = datetime.datetime.strptime(time_stamp, TIME_FORMAT)

            if model_name == self.model_name and (last_dt is None or last_dt < date_time):
                last_folder = folder
                last_dt = date_time
        
        return last_folder

    def __load_optimizer(self):
        """ Loading optimizer state """
        if not self.args.freeze_encoder:
            optimizer_load_path = self.pretrained_weights_path / f"{self.args.optimizer}.pth"
            if optimizer_load_path.exists():
                print(f"=> Loading {self.args.optimizer} weights")
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.args.device)
                self.optimizer.load_state_dict(optimizer_dict)
            else:
                print(f"Cannot find {self.args.optimizer} weights so {self.args.optimizer} is randomly initialized")


    @abstractmethod
    def _train_feed_model(self, inputs):
        pass

    @abstractmethod
    def _train_compute_loss(self, inputs, outputs) -> Dict[str, float]:
        pass

    def _train_optimize(self, losses):
        self.optimizer.zero_grad()
        losses[self.reference_train_metric].backward()
        self.optimizer.step()

    def _train_batch_statistics(self, inputs, outputs, losses): 
        mode="train"
        writer = self.writers[mode]
        for loss, value in losses.items():
            writer.add_scalar(f"{loss}", value.mean(), self.step)

        writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], self.step)

    def _train_update_lr_scheduler(self, val_metrics):
        # old_lr = self.optimizer.param_groups[0]['lr']
        # self.lr_scheduler.step(val_metrics[self.reference_val_metric])
        # new_lr = self.optimizer.param_groups[0]['lr']
        # if old_lr != new_lr:
        #     pretrained_w = self._compose_pretrained_weights()
        #     old_w = self.pretrained_weights_path
        #     self.pretrained_weights_path = pretrained_w
            
        #     self._load_model()
            
        #     self.pretrained_weights_path = old_w
        self.lr_scheduler.step()

    def start_train(self):
        """Train the model
        """
        for self.epoch in range(self.args.epochs):
            # switch to train mode
            self._set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()

            for batch_idx, inputs in enumerate(self.train_loader):
                current_time = time.time()
                data_loading_time += (current_time - before_op_time)
                before_op_time = current_time

                # -- FEED MODEL AND COMPUTE LOSSES --
                outputs = self._train_feed_model(inputs)
                losses = self._train_compute_loss(inputs, outputs)

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self._train_optimize(losses)

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx!=0 and batch_idx % self.args.log_frequency == 0:
                    self._log_time(batch_idx, self.args.log_frequency, losses, data_loading_time, gpu_time)
                    self._train_batch_statistics(inputs, outputs, losses)
                    data_loading_time = 0
                    gpu_time = 0

                self.step += 1
                before_op_time = time.time()

            # Validate on each step, save model on improvements
            val_metrics = self.val_step()
            self._print_val_metrics(val_metrics)

            if self.best_val_metric is None or val_metrics[self.reference_val_metric] <= self.best_val_metric:
                print(f"=> Saving model weights with {self.reference_val_metric} of {val_metrics[self.reference_val_metric]:.3e} "
                      f"at step {self.step} on {self.epoch} epoch.")
                self.best_val_metric = val_metrics[self.reference_val_metric]
            self._save_model()

            self._train_update_lr_scheduler(val_metrics)
            if self.args.do_early_stopping:
                if self.early_stopper.early_stop(val_metrics[self.reference_val_metric]):
                    print(f"Early stopping entered at epoch {self.epoch}!")
                    break

        print(f"=> Training of {self.log_path} complete!")

    def _print_val_metrics(self, val_metrics):
        to_print =  f"{Fore.GREEN}epoch {self.epoch:>3}{Style.RESET_ALL} " \
                    f"| current lr {self.optimizer.param_groups[0]['lr']:.2e} " \
                    f"| step {self.step} " 
        for metric_name, metric_value in val_metrics.items():
            to_print += f"| {Fore.RED}{metric_name}: {metric_value:.3e}{Style.RESET_ALL} "
        print(to_print)

    def _val_feed_model(self, inputs:dict) -> dict:
        metadata = tools.extract_subset(inputs, inputs['metadata'])
        outputs = self._train_feed_model(inputs)
        inputs.update(metadata)
        return outputs

    @abstractmethod
    def _val_compute_loss(self, inputs, outputs) -> Dict[str, float]:
        pass

    def _val_statistics(self, val_metrics):
        mode = "val"
        writer = self.writers[mode]
        for loss, value in val_metrics.items():
            writer.add_scalar(f"{loss}", value, self.step)

    def _val_visualize_batch_outputs(self, inputs, outputs, losses, batch_idx):
        pass

    @torch.no_grad()
    def val_step(self):
        """Validate the model"""
        print(f"=> Starting validation...")
        self._set_eval()
        for batch_idx,inputs in enumerate(self.val_loader):
            
            outputs = self._val_feed_model(inputs)
            losses = self._val_compute_loss(inputs, outputs)

            if batch_idx % self.args.log_frequency == 0 and self.args.test_visualize_batch_details:
                self._val_visualize_batch_outputs(inputs, outputs, losses, batch_idx)

            for val_loss_name, val_loss_value in losses.items():
                self.val_metrics[val_loss_name].update(val_loss_value)

        val_metrics = dict()
        for val_metric_name, val_metric_mean in self.val_metrics.items():
            val_metrics[val_metric_name] = val_metric_mean.average()

        # Compute stats for the tensorboard
        self._val_statistics(val_metrics)

        for val_metric in self.val_metrics.values():
            val_metric.reset()

        del inputs, losses, outputs
        self._set_train()

        return val_metrics

    def _configure_optimizers(self):
        """Default optimizer and scheduler implementation. Using Adam and ReduceLROnPlateau.
        """
        # self.optimizer = torch.optim.Adam(self.parameters_to_train, self.args.learning_rate)
        self.optimizer = torch.optim.AdamW(self.parameters_to_train, lr=self.args.learning_rate, weight_decay=5e-3)
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer=self.optimizer, 
        #     mode="min",
        #     # threshold_mode="abs",
        #     # threshold=self.args.min_delta,
        #     patience= self.args.scheduler_patience,
        #     factor=self.args.scheduler_factor,
        #     min_lr=self.args.scheduler_min_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer, 
            T_max=self.args.epochs)

    def _log_time(self, batch_idx, num_batches, losses, data_time, gpu_time):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.args.batch_size * num_batches / (gpu_time + data_time)
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        to_print = f"{Fore.GREEN}epoch {self.epoch:>3}{Style.RESET_ALL} " \
              f"| batch {batch_idx:>6} " \
              f"| current lr {self.optimizer.param_groups[0]['lr']:.2e} " \
              f"| examples/s: {samples_per_sec:5.1f} " 
        
        for loss_name, loss_vaue in losses.items():
            to_print += f"| {Fore.RED}{loss_name}: {loss_vaue.cpu().data:.3e}{Style.RESET_ALL} "

        to_print += f"| {Fore.BLUE}time elapsed: {self.sec_to_hm_str(time_sofar)}{Style.RESET_ALL} " \
              f"| {Fore.CYAN}time left: {self.sec_to_hm_str(training_time_left)}{Style.RESET_ALL} " \
              f"| CPU/GPU time: {data_time:0.1f}s/{gpu_time:0.1f}s" 

        print(to_print)
        
    def _save_model(self):
        """Save model weights to disk"""
        save_folder = self.log_path / "models" / f"weights_{self.epoch}" / str(self.step)
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = save_folder / f"{model_name}.pth"
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = save_folder / "adam.pth"
        if self.epoch > 10:  # Optimizer file is quite large! Sometimes, life is a compromise.
            torch.save(self.optimizer.state_dict(), save_path)

    def __save_args(self):
        """Save arguments to disk so we know what we ran this experiment with"""

        models_dir = self.log_path / "models"
        os.makedirs(models_dir, exist_ok=True)
        
        to_save = self.args.copy()

        # # TODO remove when stable
        # for name,arg in to_save.items():
        #     print(name,type(arg))

        with open(models_dir / 'params.yaml', 'w') as f:
            yaml.dump(to_save, f)

@dataclass
class TestBase(TrainTestBase):
    test_loader: DataLoader = field(init=False)
    test_weights_name:str = field(init=False)
    test_metrics: Dict[str, RunningMean] = field(init=False, default_factory = lambda : {})
    test_metrics_df_path : Path = field(init=False)
    test_metrics_df : pd.DataFrame = field(init=False)
    test_ouput_img_path : Path = field(init=False)
    modes : List[str] = field(init=False)

    criterions_definiton = {
        "mse_loss_test":torch.nn.MSELoss(),
        "huber_loss_test":torch.nn.HuberLoss(),
        }

    def _init_losses(self):
        self.criterions_definiton["huber_loss_test"].delta = self.args.loss_huber_delta

    def __post_init__(self):
        super().__post_init__()

        self.modes = ["test"]

        assert self.pretrained_weights_path is not None, \
            "At testing time the path for the pretrained weights must be specified!"

        self.test_weights_name = Path(self.args.pretrained_weights).name

        self.log_path = self.__compose_log_path()
        self.__clear_log_path()

        for mode in self.modes:
            self.writers[mode] = SummaryWriter(self.log_path / mode)

        # Number of batches.
        self.num_total_steps = len(self.test_loader)

        print(f"=> Testing on the {self.args.dataset_dir.upper()} dataset \n"
              f"=> Testing model named: {self.args.model_name} \n"
              f"=> Tensorboard events files are saved in: {str(self.log_path)} \n"
              f"=> Testing is using the cuda device id: {self.args.cuda_visible_devices} \n"
              )

        print(f"=> Total number of testing examples: {len(self.test_loader)*self.test_loader.batch_size} \n")
 
        self._load_model()

        for test_metric_name in self.criterions_definiton.keys():
            self.test_metrics[test_metric_name] = RunningMean()

        if 'cuda' in self.device:
            torch.cuda.synchronize()

        self.test_metrics_df_path = self.args.test_metrics_df_path
        if self.test_metrics_df_path:
            self.test_metrics_df_path = Path(self.test_metrics_df_path)
            if not self.test_metrics_df_path.exists():
                print(f"=> No existing metrics found!")
                os.makedirs(self.test_metrics_df_path.parent, exist_ok=True)
                self.test_metrics_df = None
            else:
                self.test_metrics_df = pd.read_csv(self.test_metrics_df_path, index_col=0, header=0)
                
            print(f"=> Metrics are stored in {self.test_metrics_df_path}")
            

        test_suff = self.args.test_suff
        test_suff = '' if test_suff is None else test_suff
        self.save_rectified_imgs:bool = self.args.test_save_rectified_imgs
        self.save_calibration_params:bool = self.args.test_save_calibration_params
        self.test_ouput_img_path = Path(self.args.test_img_out_path) / (self.log_path.name + test_suff)
        self.test_param_out_path = Path(self.args.test_param_out_path) / (self.log_path.name + test_suff)
        if self.save_rectified_imgs:
            os.makedirs( self.test_ouput_img_path, exist_ok=True)
        if self.save_calibration_params:
            os.makedirs( self.test_param_out_path, exist_ok=True)
        
    
    def __compose_log_path(self) -> Path:
        return Path(self.args.pretrained_weights)

    def __clear_log_path(self):
        for mode in self.modes:
            shutil.rmtree(self.log_path/mode, ignore_errors=True)

    def _init_dataloaders(self):
        # --- Load Data ---
        test_dataset = lds.get_dataset(data_path=self.args.dataset_dir,
                                          path_file=self.args.test_file,
                                          is_train=False,
                                          config=self.args)

        self.test_loader = DataLoader(test_dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     num_workers=self.args.num_workers,
                                     pin_memory=True,
                                     drop_last=False,
                                     collate_fn=test_dataset.collate_fn)

    @torch.no_grad()
    def start_test(self):
        print(f"\n=> Testing started! \n")
        
        self._set_eval()
        losses = dict()
        for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=self.num_total_steps):

            outputs = self._test_feed_model(inputs)
            losses = self._test_compute_loss(inputs, outputs)

            for test_loss_name, test_loss_value in losses.items():
                self.test_metrics[test_loss_name].update(test_loss_value)

            if self.save_rectified_imgs or self.save_calibration_params:
                self._test_save_batch_outputs(inputs, outputs, batch_idx)
            
            if batch_idx % self.args.log_frequency == 0 and self.args.test_visualize_batch_details:
                self._test_visualize_batch_outputs(inputs, outputs, losses, batch_idx)

        output_metrics = dict()
        for test_metric_name, test_metric_mean in self.test_metrics.items():
                output_metrics[test_metric_name] = test_metric_mean.average()

        self._print_test_metrics(output_metrics)
        # Compute stats for the tensorboard
        self._test_statistics(output_metrics)
        self._save_test_metrics_to_df(output_metrics)

        for test_metric in self.test_metrics.values():
            test_metric.reset()

        del inputs, losses

    def _print_test_metrics(self, test_metrics:Dict[str, float]):
        to_print = f"Testing metircs: "
        for metric_name, metric_value in test_metrics.items():
            to_print += f"| {Fore.RED}{metric_name}: {metric_value:.3e}{Style.RESET_ALL} "
        print(to_print)

    def _save_test_metrics_to_df(self, test_metrics:Dict[str, float]):
        if not self.test_metrics_df_path:
            print("=> Test metrics have not been saved!")
            return

        metric_names = list(test_metrics.keys())
        metric_values = list(test_metrics.values())
        metric_values = [value.cpu().numpy() for value in metric_values]

        df = pd.DataFrame([metric_values], columns=metric_names, index=[self.test_weights_name])
        if self.test_metrics_df is not None:
            if self.test_weights_name not in self.test_metrics_df.index:
                df = pd.concat([self.test_metrics_df, df])

        df.to_csv(self.test_metrics_df_path)


    @abstractmethod
    def _test_feed_model(self, inputs:dict) -> Dict[str,object]:
        pass

    @abstractmethod
    def _test_compute_loss(self, inputs, outputs) -> Dict[str, float]:
        pass

    def _test_statistics(self, test_metrics):
        mode = "test"
        writer = self.writers[mode]
        for metric, value in test_metrics.items():
            writer.add_scalar(f"{metric}", value, 10)

    def _test_save_batch_outputs(self, inputs, outputs, losses):
        files = inputs["file"]
        for j in range(self.args.batch_size):

            self.calibrate_and_reproject(
                img_name= files[j],
                va_vec= outputs["va_vec"][j].cpu().numpy(),
                save_params= self.save_calibration_params,
                save_rectified= self.save_rectified_imgs,
            )

    @torch.no_grad()
    def predict(self, inputs:Dict[str,object], has_bactch=True) -> Dict[str,object]:
        self._set_eval()
        if 'metadata' not in inputs:
            inputs['metadata'] = ['metadata']
        if not has_bactch: # Add batch dimension
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key][None,...]

        return self._test_feed_model(inputs)

    @abstractmethod
    def _test_visualize_batch_outputs(self, inputs, outputs, losses, batch_idx):
        pass

    def calibrate_and_reproject(self, img_name:str, va_vec:np.ndarray, sigma:np.ndarray=None,
        save_rectified:bool=False, save_params:bool=False, calib_model:Type[cal.ImageDescription]=cal.FisheyeDS_Description) -> Image:
        
        rgb_path = Path(self.args.dataset_dir) / "rgb_images" / f"{img_name}.png"
        rgb_image = Image.open(rgb_path).convert('RGB')

        height, width  = rgb_image.height, rgb_image.width
        calib = cal.Calibration.from_va_vec(va_vec, camera=calib_model, sigma=sigma)
        estimated = calib.run()

        est_des_resized = calib_model(
            height=self.args.input_height ,
            width=self.args.input_width ,
            intrinsics=dict(estimated))

        if self.args.do_resize:
            estimated["f"] *= width / self.args.input_width 

        est_des = calib_model(
            height=height,
            width=width,
            intrinsics=estimated)

        pred_afov = min(est_des.get_afov(), self.MAX_AFOV)

        persp_des = cal.Perspective_Description(
            width=self.args.test_img_out_width,
            height=self.args.test_img_out_height,
            intrinsics=dict(afov=pred_afov)
        )

        rgb_reproj,_ = mp.map_img(np.asarray(rgb_image), [est_des, persp_des])
        rgb_reproj = mp.circular_mask(rgb_reproj)

        repr_img = Image.fromarray(rgb_reproj)

        if save_rectified:
            out_img_path = self.test_ouput_img_path / rgb_path.name
            repr_img.save(out_img_path)

        if save_params:
            out_params_path = self.test_param_out_path / f'{img_name}.json'
            with open(out_params_path,'w') as f:
                json.dump(est_des.to_dict(),f,indent=3)
                
        return repr_img, rgb_image, est_des_resized
    
    
    def calibrate_and_reproject_full(self, img_name:str, angles:np.ndarray, rhos:np.ndarray,  sigma:np.ndarray=None,
        save_rectified:bool=False, save_params:bool=False, calib_model:Type[cal.ImageDescription]=cal.FisheyeDS_Description) -> Image:
        
        rgb_path = Path(self.args.dataset_dir) / "rgb_images" / f"{img_name}.png"
        rgb_image = Image.open(rgb_path).convert('RGB')

        height, width  = rgb_image.height, rgb_image.width
        
        if len(rhos >= 10):

            calib = cal.Calibration.from_rho_angles(rhos, angles, camera=calib_model, sigma=sigma)
            estimated = calib.run(method='ls', loss='soft_l1')

            if self.args.do_resize:
                estimated["f"] *= width / self.args.input_width 

            est_des = calib_model(
                height=height,
                width=width,
                intrinsics=estimated)

            # TODO comments, explaination
            pred_afov = min(est_des.get_afov(), self.MAX_AFOV)

            persp_des = cal.Perspective_Description(
                width=self.args.test_img_out_width,
                height=self.args.test_img_out_height,
                intrinsics=dict(afov=pred_afov)
            )

            rgb_reproj,_ = mp.map_img(np.asarray(rgb_image), [est_des, persp_des])
            rgb_reproj = mp.circular_mask(rgb_reproj)

            repr_img = Image.fromarray(rgb_reproj)
        
        else:
            print(f'BAD: {img_name}')
            repr_img = rgb_image.resize((
                self.args.test_img_out_width,
                self.args.test_img_out_height,
            ))
            est_des = calib_model(width=width, height=height, 
                                intrinsics={'f':500, 'a':0, 'xi':0})

        if save_rectified:
            out_img_path = self.test_ouput_img_path / rgb_path.name
            repr_img.save(out_img_path)

        if save_params:
            out_params_path = self.test_param_out_path / f'{img_name}.json'
            with open(out_params_path,'w') as f:
                json.dump(est_des.to_dict(),f,indent=3)
                
        return repr_img, rgb_image, est_des