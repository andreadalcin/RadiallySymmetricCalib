from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from typing import Dict, Tuple, Type
from pathlib import Path
from ruamel import yaml
import cv2 as cv
import numpy as np
import warnings

from misc import misc, tools
from frameworks.abs_framework import TrainBase, TestBase
from data_loader.mixed_loader import MixedDataset
from losses import  calibration_loss

from projections import calibration as cal
from PIL import Image
from projections import mappings as mp
import json

# TODO comments

#########################################
### -- VARIANCE CALIBRATION MODELS -- ###
#########################################

class AbsVarCalTrain(TrainBase):

    def __post_init__(self):
        super().__post_init__()
        self.cart2polar_grid = misc.build_c2p_grid(
            input_height=self.args.input_height,
            input_width=self.args.input_width,
            polar_width=self.args.polar_width,
            batch_size=self.args.batch_size,
            v2 = True,
        ).to(self.device)

        self.task = self.args.task

    def _init_model(self):
        super()._init_model()
        self.task = self.args.task

    def _init_losses(self):
        self.criterions_definiton = {}
        b = self.args.loss_weightedvar_beta
        self.criterions_definiton['varcal_loss'] = calibration_loss.CalibVarLoss(b)

    # -- TRAINING STEP DEFINITIONS --

    def _train_feed_model(self, inputs):
        
        self._inputs_to_device(inputs)

        ds_s = self.args.va_downsample_start
        if ds_s is not None:
            va_s, idx = misc.sample_va_vec(inputs['va_vec'], self.cell_size, ds_s)
            inputs['va_vec_full'] = inputs['va_vec']
            inputs['va_vec'] = va_s
            inputs['va_idx'] = idx

        outputs = dict()
        inputs["img_polar"] = misc.polar_grid_sample(
            x= inputs["color"],
            grid=self.cart2polar_grid,
            border_remove=0,
            )
        
        outputs["logvar"], outputs["calib"] = self.models["net"](inputs["color"], inputs["img_polar"])
        
        return outputs

    def _train_compute_loss(self, inputs, outputs) -> Dict[str, float]: 
        losses = dict()

        citerion = self.criterions_definiton[self.reference_train_metric]
        losses[self.reference_train_metric] = citerion(outputs, inputs)
            
        return losses

    # -- VALIDATION STEP DEFINITIONS --    

    def _val_compute_loss(self, inputs, outputs) -> Dict[str, float]:
        losses = dict()

        for loss_name, citerion in self.criterions_definiton.items():
            losses[f"{loss_name}_val"] = citerion(outputs, inputs)
        return losses

    def _val_visualize_batch_outputs(self, inputs, outputs, losses, batch_idx):
        writer = self.writers["val"]

        batch_idx = self.epoch * 1000 + batch_idx

        for j in range(min(1, self.args.batch_size)):  # write maximum of 1 image per batch
            cart_img = inputs['color'][j]
            writer.add_image(f"polar_img/{j}", inputs['img_polar'][j], batch_idx )
            writer.add_image(f"cart_img/{j}", cart_img, batch_idx )

            if self.task in ['calib', 'calib_full', 'joint']:
                # CALIBRATION
                va_map_pred = misc.depth_to_space(outputs["calib"])[j]
                writer.add_image(f"pred_va_map/{j}", misc.tensor_to_heatmap(va_map_pred), batch_idx )
                
                va_map_gt = misc.flat_tensor_to_3d(inputs["va_vec"][j], height=va_map_pred.shape[-1], cell_size=1).permute((0,2,1)) # Exchange W and H
                writer.add_image(f"gt_va_map/{j}", misc.tensor_to_heatmap(va_map_gt), batch_idx )

                comp_va_maps = torch.cat([va_map_pred, va_map_gt], dim=-1)
                writer.add_image(f"va_map_comp/{j}", misc.tensor_to_heatmap(comp_va_maps), batch_idx )

                res_va_maps = torch.abs(va_map_pred-va_map_gt)
                writer.add_image(f"va_map_res/{j}", misc.tensor_to_heatmap(res_va_maps), batch_idx )


                writer.add_image(f"var_map/{j}", misc.tensor_to_heatmap(torch.exp(outputs['logvar'][j])), batch_idx )

class AbsVarCalTest(TestBase):
    def __post_init__(self):
        super().__post_init__()
        self.cart2polar_grid = misc.build_c2p_grid(
            input_height=self.args.input_height,
            input_width=self.args.input_width,
            polar_width=self.args.polar_width,
            batch_size=self.args.batch_size,
            v2 = True,
        ).to(self.device)

        self.no_gt = self.args.no_gt
        self.eval_method = self.args.eval_method

    def _init_losses(self):
        self.criterions_definiton = {}
        b = self.args.loss_weightedvar_beta
        self.criterions_definiton['varcal_loss'] = calibration_loss.CalibVarLoss(b)

    def _init_dataloaders(self):
        # --- Load Data ---

        test_dataset =  MixedDataset(data_path=self.args.dataset_dir,
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


    # -- TESTING STEP DEFINITIONS -- 

    def _test_feed_model(self, inputs:Dict) -> Tuple[torch.Tensor, Dict[str,object]]:
        metadata = tools.extract_subset(inputs, inputs['metadata'])
        self._inputs_to_device(inputs)


        outputs = dict()
        inputs["img_polar"] = misc.polar_grid_sample(
            x= inputs["img"],
            grid=self.cart2polar_grid,
            border_remove=0,
            )
        
        outputs["logvar"], outputs["calib"] = self.models["net"](inputs["img"], inputs["img_polar"])

        ds_s = self.args.va_downsample_start
        if ds_s is not None:
            if self.no_gt:
                inputs['va_idx'] = torch.arange(ds_s, self.args.input_height//2, self.cell_size)
            else:
                va_s, idx = misc.sample_va_vec(inputs['va_vec'], self.cell_size, ds_s)
                inputs['va_vec_full'] = inputs['va_vec']
                inputs['va_vec'] = va_s
                inputs['va_idx'] = idx
        
        inputs.update(metadata)
        return outputs
    
    def _test_save_batch_outputs(self, inputs, outputs, losses):
        files = inputs["file"]

        calib = misc.depth_to_space(outputs["calib"])

        if self.eval_method == 'va_vec':
            va_vec = calib.mean(dim=-1)
            for j in range(self.args.batch_size):
                
                super().calibrate_and_reproject(
                    img_name= files[j],
                    va_vec= va_vec[j][0].cpu().numpy(),
                    save_params= self.save_calibration_params,
                    save_rectified= self.save_rectified_imgs,
                )
            return

        if self.eval_method == 'sup':
            probs = misc.polar_grid_sample(
                (outputs['prob'] * inputs['valid_mask'])[:,None,...], 
                self.cart2polar_grid,
                border_remove=0,
                )[:,0,...]
            
            # Exp
            k_size = 5
            pad = k_size // 2
            conv = torch.ones((calib.shape[0],1,1,k_size), device=calib.device) / k_size

            va_map_pred_conv = misc.mixed_padding(calib, (0,pad))
            va_map_pred_conv = torch.nn.functional.conv2d(va_map_pred_conv, conv)
            
            for j in range(self.args.batch_size):
                nms_size = 8
                min_prob = 0.001
                keep_top_k = 300

                prob = misc.box_nms(probs[j], size=nms_size, min_prob=min_prob, keep_top_k=keep_top_k)

                va_map = va_map_pred_conv[j][0].cpu().numpy()
                prob = prob.cpu().numpy()

                angles = va_map[prob > 0] # (n,)
                rhos =  np.where(prob>0)[0] # (n,)

                self.calibrate_and_reproject(
                    img_name= files[j],
                    angles= angles,
                    rhos = rhos,
                    save_params= self.save_calibration_params,
                    save_rectified= self.save_rectified_imgs,
                )

            return

        if self.eval_method == 'va_sub':
            for j in range(self.args.batch_size):
                # angles = outputs["calib"][j][0]
                # rhos = inputs['va_idx'][:,None].expand_as(angles)
                angles = outputs["calib"][j][0].mean(-1)
                rhos = inputs['va_idx'].expand_as(angles)
                
                self.calibrate_and_reproject(
                    img_name= files[j],
                    angles = angles.cpu().numpy().flatten(),
                    rhos = rhos.cpu().numpy().flatten(),
                    save_params= self.save_calibration_params,
                    save_rectified= self.save_rectified_imgs,
                )
            return
        
        if self.eval_method == 'var':
            for j in range(self.args.batch_size):
                var = torch.exp(outputs['logvar'][j])
                sigma = torch.sqrt(var)

                va_map_pred = calib[j][0]

                rhos = inputs['va_idx'][:,None].expand_as(va_map_pred)
                self.calibrate_and_reproject(
                    img_name= files[j],
                    angles=va_map_pred.cpu().numpy().flatten(),
                    rhos=rhos.cpu().numpy().flatten(),
                    sigma=sigma[0].cpu().numpy().flatten(),
                    save_params= self.save_calibration_params,
                    save_rectified= self.save_rectified_imgs,
                )
            return

            
    
    def _test_compute_loss(self, inputs, outputs) -> Dict[str, float]:
        losses = dict()
        if self.no_gt:
            return {}
        
        for loss_name, citerion in self.criterions_definiton.items():
            losses[loss_name] = citerion(outputs, inputs)
        return losses
    
    def _test_visualize_batch_outputs(self, inputs, outputs, losses, batch_idx):
        files = inputs["file"]
        dess = inputs["des"]
        writer = self.writers["test"]
    
        for j in range(min(4, self.args.batch_size)):  # write maximum of four images
            
            cart_img = inputs['img'][j]
            writer.add_image(f"polar_img/{j}", inputs['img_polar'][j], batch_idx )
            writer.add_image(f"cart_img/{j}", cart_img, batch_idx )

            cart_img = misc.tensor_nCH_to_3CH(cart_img)

            # CALIBRATION
            va_map_pred = misc.depth_to_space(outputs["calib"])[j]
            writer.add_image(f"pred_va_map/{j}", misc.tensor_to_heatmap(va_map_pred), batch_idx )
            
            va_map_gt = misc.flat_tensor_to_3d(inputs["va_vec"][j], height=va_map_pred.shape[-1], cell_size=1).permute((0,2,1)) # Exchange W and H
            writer.add_image(f"gt_va_map/{j}", misc.tensor_to_heatmap(va_map_gt), batch_idx )

            comp_va_maps = torch.cat([va_map_pred, va_map_gt], dim=-1)
            writer.add_image(f"va_map_comp/{j}", misc.tensor_to_heatmap(comp_va_maps), batch_idx )

            res_va_maps = torch.abs(va_map_pred-va_map_gt)
            writer.add_image(f"va_map_res/{j}", misc.tensor_to_heatmap(res_va_maps), batch_idx )

            var = torch.exp(outputs['logvar'][j])
            writer.add_image(f"var_map/{j}", misc.tensor_to_heatmap(var), batch_idx )
            sigma = torch.sqrt(var)
            writer.add_image(f"sigma_map/{j}", misc.tensor_to_heatmap(sigma), batch_idx )

            # va_vec = va_map_pred[0].mean(dim=-1)
            # cals = super().calibrate_and_reproject(
            #     img_name= files[j],
            #     va_vec= va_vec.cpu().numpy(),
            #     save_params= self.save_calibration_params,
            #     save_rectified= self.save_rectified_imgs,
            # )

            rhos = inputs['va_idx'][:,None].expand_as(va_map_pred[0])
            cals = self.calibrate_and_reproject(
                img_name= files[j],
                angles=va_map_pred[0].cpu().numpy().flatten(),
                rhos=rhos.cpu().numpy().flatten(),
                sigma=sigma[0].cpu().numpy().flatten(),
                save_params= self.save_calibration_params,
                save_rectified= self.save_rectified_imgs,
            )


            est_va_vec = cals[2].get_va_vector()
            ds_s = self.args.va_downsample_start
            if ds_s is not None:
                est_va_vec, idx = misc.sample_va_vec(est_va_vec, self.cell_size, ds_s)
            gt_va_vec = inputs['va_vec'][j].cpu().numpy()

            import matplotlib.pyplot as plt
            x = np.arange(0,gt_va_vec.shape[0])
            fig = plt.figure()
            for i in range(va_map_pred.shape[2]):
                plt.plot(x, va_map_pred[0,:,i].cpu().numpy(), 'r+', label='points' if i==0 else None)
            plt.plot(x, gt_va_vec, label='curve')
            plt.plot(x, est_va_vec, label='estimated', color = 'g')
            plt.legend()

            writer.add_figure(f"calibration_plot/{j}", fig, batch_idx)


    def predict(self, inputs: Dict[str, object], has_bactch=True) -> Dict[str, object]:
        shape = inputs['img'].shape
        if has_bactch:
            assert len(shape) == 4
            c_dim, h_dim, w_dim = 1, 2, 3
        else:
            assert len(shape) == 3
            c_dim, h_dim, w_dim = 0, 1, 2
        
        assert shape[c_dim] == 1
        assert shape[h_dim] % 16 == 0
        assert shape[w_dim] % 8 == 0

        warnings.warn("Avoid cropping grid!")
        self.cart2polar_grid = misc.crop_c2p_grid(self.cart2polar_grid, shape[h_dim]//2)

        return super().predict(inputs, has_bactch)

    def calibrate_and_reproject(self, img_name:str, angles:np.ndarray, rhos:np.ndarray,  sigma:np.ndarray=None,
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