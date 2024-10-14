from __future__ import annotations

import torch

from typing import Dict, Tuple

from misc import misc, tools
from frameworks.abs_framework import TrainBase, TestBase

# TODO comments

#########################################
### -- ABSTRACT CALIBRATION MODELS -- ###
#########################################

class AbsCalibTrain(TrainBase):
    # -- TRAINING STEP DEFINITIONS --

    def _train_feed_model(self, inputs):
        self._inputs_to_device(inputs)
        outputs = dict()
        inputs["color_polar"] = misc.polar_grid_sample(
            x= inputs["color"],
            grid=self.cart2polar_grid,
            border_remove=0,
            )
        outputs["va_vec"] = self.models["net"](inputs["color_polar"])
        return outputs

    def _train_compute_loss(self, inputs, outputs) -> Dict[str, float]: 
        losses = dict()
        citerion = self.criterions_definiton[self.reference_train_metric]
        losses[self.reference_train_metric] = citerion(outputs["va_vec"], inputs["va_vec"])
        return losses

    # -- VALIDATION STEP DEFINITIONS --    
    def _val_compute_loss(self, inputs, outputs) -> Dict[str, float]:
        losses = dict()
        for loss_name, citerion in self.criterions_definiton.items():
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
            compare_va_vec = misc.tensor_to_heatmap(compare_va_vec, colormap='viridis')
            va_diff = misc.tensor_to_heatmap(torch.abs(pred_va_vec - gt_va_vec), colormap='inferno') # L1 norm
            compare_all = torch.cat([compare_va_vec, va_diff], dim = 1)

            writer.add_image(f"polar_img/{j}", inputs["color_polar"][j], batch_idx )
            writer.add_image(f"comparison_pred_gt_diff/{j}", compare_all, batch_idx )

class AbsCalibTest(TestBase):
    # -- TESTING STEP DEFINITIONS -- 

    def _test_feed_model(self, inputs:dict) -> dict:
        metadata = tools.extract_subset(inputs, inputs['metadata'])

        self._inputs_to_device(inputs)
        outputs = dict()
        inputs["color_polar"] = misc.polar_grid_sample(
            x= inputs["color"],
            grid=self.cart2polar_grid,
            border_remove=0,
            )
        outputs["va_vec"] = self.models["net"](inputs["color_polar"])

        inputs.update(metadata)

        return outputs
    
    def _test_compute_loss(self, inputs, outputs) -> Dict[str, float]:
        losses = dict()
        for loss_name, citerion in self.criterions_definiton.items():
            losses[loss_name] = citerion(outputs["va_vec"], inputs["va_vec"])
        return losses
    
    def _test_visualize_batch_outputs(self, inputs, outputs, losses, batch_idx, input_info):
        files = input_info["file"]
        dess = input_info["des"]
        writer = self.writers["test"]

        for j in range(min(4, self.args.batch_size)):  # write maximum of four images
            pred_va_vec = misc.flat_tensor_to_3d(outputs['va_vec'][j])
            gt_va_vec = misc.flat_tensor_to_3d(inputs['va_vec'][j])
            compare_va_vec = torch.cat([pred_va_vec, gt_va_vec], dim = 1)
            compare_va_vec = misc.tensor_to_heatmap(compare_va_vec, colormap='viridis')
            va_diff = misc.tensor_to_heatmap(torch.abs(pred_va_vec - gt_va_vec), colormap='inferno') # L1 norm
            compare_all = torch.cat([compare_va_vec, va_diff], dim = 1)

            writer.add_image(f"polar_img/{j}", inputs["color_polar"][j], batch_idx )
            writer.add_image(f"comparison_pred_gt_diff/{j}", compare_all, batch_idx )

