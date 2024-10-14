from __future__ import annotations

import torch
import numpy as np

from typing import Dict, Tuple

from misc import misc, tools
from frameworks.abs_framework import TrainBase, TestBase
import losses.calibration_loss as losses
from torchvision import transforms
from misc.misc import get_radial_mask
from projections import calibration as cal

# TODO comments

#########################################
### -- ABSTRACT CALIBRATION MODELS -- ###
#########################################

class AbsMixedCalibTrain(TrainBase):
    # -- TRAINING STEP DEFINITIONS --

    def _train_feed_model(self, inputs):
        self._inputs_to_device(inputs)
        outputs = dict()
        inputs["color_polar"] = misc.polar_grid_sample(
            x= inputs["color"],
            grid=self.cart2polar_grid,
            border_remove=0,
            )
        outputs["va_vec"] = self.models["net"](inputs["color"], inputs["color_polar"])#, inputs['va_vec'])
        if self.reference_train_metric == "mse_relative_loss":
            outputs["va_vec_rel"] = outputs["va_vec"]
            outputs["va_vec"] = torch.cumsum(outputs["va_vec_rel"], dim=1)
        return outputs
    
    def _init_losses(self):
        super()._init_losses()
        #self.criterions_definiton["mse_balanced_loss"] = losses.CalibMSEBalanced()
        #self.criterions_definiton["mse_balanced2_loss"] = losses.CalibMSEBalanced2()
        #self.criterions_definiton["mse_balanced3_loss"] = losses.CalibMSEBalanced3()
        #self.criterions_definiton["reprojection_loss"] = losses.CalibRE_DS(input_height=self.args.input_height, va_downsample=self.args.va_downsample, device=self.device)
        self.criterions_definiton["reprojection2_loss"] = losses.CalibRE2_DS(input_height=self.args.input_height, va_downsample=self.args.va_downsample, device=self.device)
        #self.criterions_definiton["mse_relative_loss"] = RelvativeVaVecMSE()

    def _train_compute_loss(self, inputs, outputs) -> Dict[str, float]: 
        losses = dict()
        citerion = self.criterions_definiton[self.reference_train_metric]

        if self.reference_train_metric == "mse_relative_loss":
            losses[self.reference_train_metric] = citerion(outputs["va_vec_rel"], inputs["va_vec"])
        elif 'reprojection' in self.reference_train_metric:
            losses[self.reference_train_metric] = citerion(outputs["va_vec"], inputs["params"])
        else:
            losses[self.reference_train_metric] = citerion(outputs["va_vec"], inputs["va_vec"])
        return losses

    # -- VALIDATION STEP DEFINITIONS --    
    def _val_compute_loss(self, inputs, outputs) -> Dict[str, float]:
        losses = dict()
        for loss_name, citerion in self.criterions_definiton.items():
            if loss_name == "mse_relative_loss":
                if self.reference_train_metric == "mse_relative_loss":
                    losses[f"{loss_name}_val"] = citerion(outputs["va_vec_rel"], inputs["va_vec"])
                    
            elif 'reprojection' in loss_name:
                losses[f"{loss_name}_val"] = citerion(outputs["va_vec"], inputs["params"])
            else:
                losses[f"{loss_name}_val"] = citerion(outputs["va_vec"], inputs["va_vec"])
                
        return losses
    
    def _val_visualize_batch_outputs(self, inputs, outputs, losses, batch_idx):
        writer = self.writers["val"]

        batch_idx_base = self.epoch * 1000 + batch_idx
        for j in range(min(4, self.args.batch_size)):  # write maximum of four images
            batch_idx = batch_idx_base + j
            pred_va_vec = misc.flat_tensor_to_3d(outputs['va_vec'][j])
            gt_va_vec = misc.flat_tensor_to_3d(inputs['va_vec'][j])
            compare_va_vec = torch.cat([pred_va_vec, gt_va_vec], dim = 1)
            compare_va_vec = compare_va_vec / (torch.pi / 180 * 230)
            compare_va_vec = misc.tensor_to_heatmap(compare_va_vec, colormap='viridis', rescale=False)
            va_diff = misc.tensor_to_heatmap(torch.abs(pred_va_vec - gt_va_vec), colormap='inferno') # L1 norm
            compare_all = torch.cat([compare_va_vec, va_diff], dim = 1)

            writer.add_image(f"polar_img/{j}", inputs["color_polar"][j], batch_idx )
            writer.add_image(f"comparison_pred_gt_diff/{j}", compare_all, batch_idx )

class AbsMixedCalibTest(TestBase):
    # -- TESTING STEP DEFINITIONS -- 

    def _test_feed_model(self, inputs:dict) -> dict:
        if self.args.test_visualize_batch_details:
            self.models["net"].writer = self.writers["test"]
        
        metadata = tools.extract_subset(inputs, inputs['metadata'])

        self._inputs_to_device(inputs)
        outputs = dict()
        inputs["color_polar"] = misc.polar_grid_sample(
            x= inputs["color"],
            grid=self.cart2polar_grid,
            border_remove=0,
            )
        outputs["va_vec"] = self.models["net"](inputs["color"], inputs["color_polar"])

        if self.args.reference_train_metric == "mse_relative_loss":
            outputs["va_vec_rel"] = outputs["va_vec"]
            outputs["va_vec"] = torch.cumsum(outputs["va_vec_rel"], dim=1)
            
        inputs.update(metadata)

        return outputs
    
    def _test_compute_loss(self, inputs, outputs) -> Dict[str, float]:
        losses = dict()
        for loss_name, citerion in self.criterions_definiton.items():
            losses[loss_name] = citerion(outputs["va_vec"], inputs["va_vec"])
        return losses
    
    def _test_save_batch_outputs(self, inputs, outputs, losses):
        
        files = inputs["file"]
        for j in range(self.args.batch_size):

            if self.args.va_downsample is None:
                self.calibrate_and_reproject(
                    img_name= files[j],
                    va_vec= outputs["va_vec"][j].cpu().numpy(),
                    save_params= self.save_calibration_params,
                    save_rectified= self.save_rectified_imgs,
                )
            else:
                step = self.args.input_height // 2 // self.args.va_downsample
                rhos = np.arange((step-1),(self.args.input_height // 2),step)
                self.calibrate_and_reproject_full(
                    img_name= files[j],
                    angles=outputs["va_vec"][j].cpu().numpy(),
                    rhos=rhos,
                    save_params= self.save_calibration_params,
                    save_rectified= self.save_rectified_imgs,
                )
    
    def _test_visualize_batch_outputs(self, inputs, outputs, losses, batch_idx):
        writer = self.writers["test"]

        

        d = inputs['va_vec'].clone()
        d[:,:d.shape[1]//2] = d[:,[d.shape[1]//2]]
        residual = (outputs['va_vec'] - inputs['va_vec']) ** 2 / (d) ** 2
        residual = torch.repeat_interleave(residual, 2, dim=-1)
        residual = torch.repeat_interleave(residual, 10, dim=-2)
        print(torch.rad2deg(inputs['va_vec'][:,-1])*2)
        writer.add_image(f"residuals", misc.tensor_to_heatmap(residual, colormap='inferno'), batch_idx )

        residual = (outputs['va_vec'] - inputs['va_vec']) ** 2 / (inputs['va_vec'] + torch.deg2rad(torch.tensor(1))) ** 2
        residual = torch.repeat_interleave(residual, 2, dim=-1)
        residual = torch.repeat_interleave(residual, 10, dim=-2)
        writer.add_image(f"residuals2", misc.tensor_to_heatmap(residual, colormap='inferno'), batch_idx )

        d = inputs['va_vec'].clone()
        d[:,:150] = d[:,[150]]
        residual = (outputs['va_vec'] - inputs['va_vec']) ** 2 / (d) ** 2
        residual = torch.repeat_interleave(residual, 2, dim=-1)
        residual = torch.repeat_interleave(residual, 10, dim=-2)
        writer.add_image(f"residuals3", misc.tensor_to_heatmap(residual, colormap='inferno'), batch_idx )

        residual = (outputs['va_vec'] - inputs['va_vec']) ** 2
        residual = torch.repeat_interleave(residual, 2, dim=-1)
        residual = torch.repeat_interleave(residual, 10, dim=-2)
        writer.add_image(f"residuals_mse", misc.tensor_to_heatmap(residual, colormap='inferno'), batch_idx )
        
        residual = (outputs['va_vec'] - inputs['va_vec']) ** 2 / inputs['va_vec'][:,[-1]] ** 2
        residual = torch.repeat_interleave(residual, 2, dim=-1)
        residual = torch.repeat_interleave(residual, 10, dim=-2)
        writer.add_image(f"residuals4", misc.tensor_to_heatmap(residual, colormap='inferno'), batch_idx )


        d = torch.cat([inputs['va_vec'],d], dim=0)
        d = torch.repeat_interleave(d, 2, dim=-1)
        d = torch.repeat_interleave(d, 10, dim=-2)
        writer.add_image(f"d", misc.tensor_to_heatmap(d, colormap='inferno'), batch_idx )

        self.visualize_activations(writer, inputs, batch_idx, selected_layers=[1,3,7])

        for j in range(min(4, self.args.batch_size)):  # write maximum of four images
            pred_va_vec = misc.flat_tensor_to_3d(outputs['va_vec'][j])
            gt_va_vec = misc.flat_tensor_to_3d(inputs['va_vec'][j])
            compare_va_vec = torch.cat([pred_va_vec, gt_va_vec], dim = 1)
            compare_va_vec = torch.cat([pred_va_vec, gt_va_vec], dim = 1)
            compare_va_vec = compare_va_vec / (torch.pi)
            compare_va_vec = misc.tensor_to_heatmap(compare_va_vec, colormap='viridis', rescale=False)
            va_diff = misc.tensor_to_heatmap(torch.abs(pred_va_vec - gt_va_vec), colormap='inferno') # L1 norm
            compare_all = torch.cat([compare_va_vec, va_diff], dim = 1)

            writer.add_image(f"polar_img/{j}", inputs["color_polar"][j], batch_idx )
            writer.add_image(f"comparison_pred_gt_diff/{j}", compare_all, batch_idx )
            
    
    def visualize_activations(self, writer, inputs, batch_idx, selected_layers=[]):
        vis = inputs['color_polar'] if self.args.do_polar_mapping else inputs['color']
        index = 0
        for layer in self.models['net'].net.features:            
            index +=1
            
            # Forward pass layer by layer
            # x is not used after this point because it is only needed to trigger
            # the forward hook function
            vis = layer(vis)
            # Only need to forward until the selected layer is reached
            if index in selected_layers:
                for i in range(3):
                    # (forward hook function triggered)
                    writer.add_image(f"fe_activation_{index}/{i}",  misc.tensor_to_heatmap(vis[0,i,...], colormap='gray'), batch_idx)
    
    
    
    def predict(self, img:np.ndarray, has_bactch=False, calib_model=cal.FisheyeDS_Description, loss='cauchy') -> Dict[str, object]:
        base_img= img.astype(np.uint8)
        img = transforms.ToTensor()(base_img)
        crop_size = min(base_img.shape[:2])
        
        img = transforms.CenterCrop(size=crop_size)(img)
        
        assert img.shape[1] == img.shape[2]
        mask = ~get_radial_mask((self.args.input_height,self.args.input_width))
        mask = torch.from_numpy(mask[None,...]).expand(3,-1,-1)
        resize = transforms.Resize((self.args.input_height, self.args.input_width),
                                        interpolation=transforms.InterpolationMode.BICUBIC)
        img = resize(img)
        img[mask] = 0
        inputs = dict(color = img)
        

        va_vec = super().predict(inputs, has_bactch)["va_vec"][0]
        
        step = self.args.input_height // 2 // self.args.va_downsample
        rhos = np.arange((step-1),(self.args.input_height // 2),step)
        
        height, width  = crop_size, crop_size
        
        angles=va_vec.cpu().numpy()
        calib = cal.Calibration.from_rho_angles(rhos, angles, camera=calib_model)
        estimated = calib.run(method='ls', loss=loss)

        estimated["f"] *= width / self.args.input_width 

        est_des = calib_model(
            height=height,
            width=width,
            intrinsics=estimated)
        return angles,rhos, est_des