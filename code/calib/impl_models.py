import torch
from torchinfo import summary

from frameworks.base_framework import *
from frameworks import mixed_framework
from models import cal_models

#########################################
###  -- IMPLEMENTED CALIB MODELS --   ###
#########################################

"""
SWING transformer
"""     

class SwigV1_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3

        self.models["net"] = cal_models.SwingV1(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=self.args.input_height//2,
            input_nc= input_nc,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )

class SwigV1_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        # --- Init model ---
        self.models["net"] = cal_models.SwingV1(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth = self.args.input_height//2,
            input_nc = input_nc,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
"""
ConvNext network
"""  

class ConvNextV1_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3

        self.models["net"] = cal_models.ConvNextV1(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=self.args.input_height//2,
            input_nc= input_nc,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNextV1_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        # --- Init model ---
        self.models["net"] = cal_models.ConvNextV1(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth = self.args.input_height//2,
            input_nc = input_nc,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        

class ConvNextV2_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3

        self.models["net"] = cal_models.ConvNextV2(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=self.args.input_height//2,
            input_nc= input_nc,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNextV2_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        # --- Init model ---
        self.models["net"] = cal_models.ConvNextV2(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth = self.args.input_height//2,
            input_nc = input_nc,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNextV3_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3

        self.models["net"] = cal_models.ConvNextV3(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=self.args.input_height//2,
            input_nc= input_nc,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNextV3_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        # --- Init model ---
        self.models["net"] = cal_models.ConvNextV3(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth = self.args.input_height//2,
            input_nc = input_nc,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_mini_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3

        self.models["net"] = cal_models.ConvNext_mini(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=self.args.input_height//2,
            input_nc= input_nc,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_mini_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        # --- Init model ---
        self.models["net"] = cal_models.ConvNext_mini(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth = self.args.input_height//2,
            input_nc = input_nc,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_mini2_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3

        self.models["net"] = cal_models.ConvNext_mini2(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=self.args.input_height//2,
            input_nc= input_nc,
            polar=self.args.do_polar_mapping,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_mini2_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        # --- Init model ---
        self.models["net"] = cal_models.ConvNext_mini2(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth = self.args.input_height//2,
            input_nc = input_nc,
            polar=self.args.do_polar_mapping,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_mini3_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample

        self.models["net"] = cal_models.ConvNext_mini3(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc= input_nc,
            polar=self.args.do_polar_mapping,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_mini3_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample
            
        # --- Init model ---
        self.models["net"] = cal_models.ConvNext_mini3(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc = input_nc,
            polar=self.args.do_polar_mapping,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
        
class ConvNext_mix_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample

        self.models["net"] = cal_models.ConvNext_mix(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc= input_nc,
            polar=self.args.do_polar_mapping,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_mix_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample
            
        # --- Init model ---
        self.models["net"] = cal_models.ConvNext_mix(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc = input_nc,
            polar=self.args.do_polar_mapping,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_mix_feat_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample

        self.models["net"] = cal_models.ConvNext_mix_feat(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc= input_nc,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_mix_feat_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample
            
        # --- Init model ---
        self.models["net"] = cal_models.ConvNext_mix_feat(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc = input_nc,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
        
class ConvNext_mix_feat_small_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample

        self.models["net"] = cal_models.ConvNext_mix_feat_small(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc= input_nc,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_mix_feat_small_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample
            
        # --- Init model ---
        self.models["net"] = cal_models.ConvNext_mix_feat_small(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc = input_nc,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_polar_feat_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample

        self.models["net"] = cal_models.ConvNext_polar_feat(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc= input_nc,
            polar=self.args.do_polar_mapping,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_polar_feat_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample
            
        # --- Init model ---
        self.models["net"] = cal_models.ConvNext_polar_feat(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc = input_nc,
            polar=self.args.do_polar_mapping,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
        
class ConvNext_mix_feat_deep_Train(mixed_framework.AbsMixedCalibTrain):
    def _init_model(self):
        super()._init_model()
        # --- Init model ---
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample

        self.models["net"] = cal_models.ConvNext_mix_feat_deep(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc= input_nc,
            ).to(self.device)
        self.parameters_to_train += list(self.models["net"].parameters())

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )
        
class ConvNext_mix_feat_deep_Test(mixed_framework.AbsMixedCalibTest):
    def _init_model(self):
        input_nc = 1 if self.args.grayscale else 3
        
        if self.args.va_downsample is None:
            calib_out_depth=self.args.input_height//2
        else:
            calib_out_depth=self.args.va_downsample
            
        # --- Init model ---
        self.models["net"] = cal_models.ConvNext_mix_feat_deep(
            n_layers_fe=self.args.network_fe_layers,
            n_layers_fc=self.args.network_fc_layers,
            kernel_size=self.args.kernel_size,
            calib_out_depth=calib_out_depth,
            input_nc = input_nc,
            ).to(self.device)

        summary(self.models["net"], 
            input_size=[(self.args.batch_size,input_nc,self.args.input_height,self.args.input_width),
            (self.args.batch_size,input_nc,self.args.input_height//2,self.args.polar_width)],
            )