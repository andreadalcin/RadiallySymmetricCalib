import torch
from torch import nn
import numpy as np
#from misc.misc import space_to_depth, polar_grid_sample

def weighted_mse_loss(input, target, weight, eps = 1e-10):
    if weight is None:
        return ((input - target) ** 2).mean()
    return (weight * (input - target) ** 2).sum() / (weight.sum() + eps)

    
class CalibMSEBalanced(nn.Module):
    def __init__(self):
        """_summary_
        """
        super().__init__()

    def forward(self, pred, target):

        residual = (pred - target) ** 2

        residual = residual / (target + np.radians(1)) ** 2
        
        return residual.mean()
    
class RelvativeVaVecMSE(nn.Module):
    def __init__(self):
        """_summary_
        """
        super().__init__()

    def forward(self, pred:torch.Tensor, target:torch.Tensor):
        prev_va = torch.zeros_like(target)
        prev_va[:,1:] = target[:,:-1]
        
        delta_va = target - prev_va

        residual = (pred - delta_va) ** 2
        
        return residual.mean()
    
class CalibRE_DS(nn.Module):
    def __init__(self,input_height=400,va_downsample=None,device=None,penality=1e+7):
        """_summary_
        """
        super().__init__()
        self.penality = penality
        if va_downsample is None:
            step = 1
        else:
            step = input_height // 2 // va_downsample
        self.target = torch.arange((step-1),(input_height // 2),step,device=device)
        
    def ds_radial_projection(self, cos_thetas:torch.Tensor, thetas:torch.Tensor, f:torch.Tensor, a:torch.Tensor, xi:torch.Tensor):
        # params = [f,a,xi]
        sin_thetas = torch.sin(thetas)
        
        e_1_2 = xi *xi + 2*xi*cos_thetas + torch.ones_like(cos_thetas)
        
        e_1 = torch.sqrt(e_1_2)
        return f * sin_thetas / (a*e_1 + (1-a)*(xi+cos_thetas))

    def forward(self, thetas:torch.Tensor, params:torch.Tensor):
    
        
        cos_thetas = torch.cos(thetas)        
        f = params[:,[0]]
        a = params[:,[1]]
        xi =params[:,[2]]
        
        w1 = torch.where(a <= 0.5, a / (1 - a), (1 - a) / a)
        #w1 = params[:,1] / (1 - params[:,1]) if params[:,1] <= 0.5 else (1 - params[:,1]) / params[:,1]
        w2 = (w1 + xi ) / torch.sqrt(2*w1*xi + xi*xi + 1)
        
        # out_mask = cos_thetas > - w2
        low_theta = thetas * (-self.penality)
        high_theta = (thetas - torch.pi) * self.penality
        out_theta = (cos_thetas + w2) * (-self.penality)
        low_mask = low_theta > 0
        high_mask = high_theta > 0 
        out_mask = out_theta >= 0
        #print(thetas[out_mask].shape)
        
        ok_mask = (~low_mask) & (~high_mask) & (~out_mask)
        
        pred = self.ds_radial_projection(cos_thetas[ok_mask], thetas[ok_mask], 
                                         f=f.expand_as(thetas)[ok_mask],
                                         a=a.expand_as(thetas)[ok_mask],
                                         xi=xi.expand_as(thetas)[ok_mask])
        
        mse = (pred - self.target.expand(thetas.shape[0],-1)[ok_mask]) ** 2
        
        residuals = torch.zeros_like(thetas)
        residuals[low_mask] = low_theta[low_mask] + self.penality
        residuals[high_mask] = high_theta[high_mask] + self.penality
        residuals[out_mask] = out_theta[out_mask] + self.penality
        residuals[ok_mask] = mse
        return residuals.mean()
    
class CalibRE2_DS(nn.Module):
    def __init__(self,input_height=400,va_downsample=None,device=None,penality=1e+7):
        """_summary_
        """
        super().__init__()
        self.penality = penality
        if va_downsample is None:
            step = 1
        else:
            step = input_height // 2 // va_downsample
        self.target = torch.arange((step-1),(input_height // 2),step,device=device)
        
    def ds_radial_projection(self, cos_thetas:torch.Tensor, thetas:torch.Tensor, f:torch.Tensor, a:torch.Tensor, xi:torch.Tensor):
        # params = [f,a,xi]
        sin_thetas = torch.sin(thetas)
        
        e_1_2 = xi *xi + 2*xi*cos_thetas + torch.ones_like(cos_thetas)
        
        e_1 = torch.sqrt(e_1_2)
        return f * sin_thetas / (a*e_1 + (1-a)*(xi+cos_thetas))

    def forward(self, thetas:torch.Tensor, params:torch.Tensor):
    
        
        cos_thetas = torch.cos(thetas)        
        f = params[:,[0]]
        a = params[:,[1]]
        xi =params[:,[2]]
        
        w1 = torch.where(a <= 0.5, a / (1 - a), (1 - a) / a)
        #w1 = params[:,1] / (1 - params[:,1]) if params[:,1] <= 0.5 else (1 - params[:,1]) / params[:,1]
        w2 = (w1 + xi ) / torch.sqrt(2*w1*xi + xi*xi + 1)
        
        # out_mask = cos_thetas > - w2
        low_theta = thetas * (-self.penality)
        high_theta = (thetas - torch.pi) * self.penality
        
        # out_theta = (cos_thetas + w2) * (-self.penality)
        
        # out_mask = out_theta >= 0
        
        low_mask = low_theta > 0
        high_mask = high_theta > 0 
        out_mask = cos_thetas <= - w2
        _cos_thetas = cos_thetas.clone()
        _cos_thetas[out_mask] = - w2.expand_as(cos_thetas)[out_mask] + 1e-4
        _thetas = thetas.clone()
        _thetas[out_mask] = torch.arccos(_cos_thetas[out_mask])
        #print(thetas[out_mask].shape)
        
        ok_mask = (~low_mask) & (~high_mask) #& (~out_mask)
        
        pred = self.ds_radial_projection(_cos_thetas[ok_mask], _thetas[ok_mask], 
                                         f=f.expand_as(thetas)[ok_mask],
                                         a=a.expand_as(thetas)[ok_mask],
                                         xi=xi.expand_as(thetas)[ok_mask])
        
        mse = (pred - self.target.expand(thetas.shape[0],-1)[ok_mask]) ** 2
        
        residuals = torch.zeros_like(thetas)
        residuals[low_mask] = low_theta[low_mask] + self.penality
        residuals[high_mask] = high_theta[high_mask] + self.penality
        #residuals[out_mask] = out_theta[out_mask] + self.penality
        residuals[ok_mask] = mse
        return residuals.mean()
    
    
    
class CalibRE2_DS_new(nn.Module):
    def __init__(self,input_height=400,va_downsample=None,device=None,penality=1e+7, shrink=np.deg2rad(1)):
        """_summary_
        """
        super().__init__()
        self.penality = penality
        self.shrink =  shrink
        if va_downsample is None:
            step = 1
        else:
            step = input_height // 2 // va_downsample
        self.target = torch.arange((step-1),(input_height // 2),step,device=device)
        
    def ds_radial_projection(self, thetas:torch.Tensor, f:torch.Tensor, a:torch.Tensor, xi:torch.Tensor):
        # params = [f,a,xi]
        sin_thetas = torch.sin(thetas)
        cos_thetas = torch.cos(thetas)
        
        e_1_2 = xi *xi + 2*xi*cos_thetas + torch.ones_like(cos_thetas)
        
        e_1 = torch.sqrt(e_1_2)
        return f * sin_thetas / (a*e_1 + (1-a)*(xi+cos_thetas))

    def forward(self, thetas:torch.Tensor, params:torch.Tensor):
         
        f = params[:,[0]]
        a = params[:,[1]]
        xi =params[:,[2]]
        
        w1 = torch.where(a <= 0.5, a / (1 - a), (1 - a) / a)
        #w1 = params[:,1] / (1 - params[:,1]) if params[:,1] <= 0.5 else (1 - params[:,1]) / params[:,1]
        w2 = torch.arccos(- (w1 + xi ) / torch.sqrt(2*w1*xi + xi*xi + 1)) - self.shrink
    
        out_mask = thetas >= w2
        _thetas = thetas.clone()
        _thetas[out_mask] = w2.expand_as(thetas)[out_mask]
        
        pred = self.ds_radial_projection(_thetas, 
                                         f=f.expand_as(thetas),
                                         a=a.expand_as(thetas),
                                         xi=xi.expand_as(thetas))
        
        mse = (pred - self.target.expand(thetas.shape[0],-1)) ** 2
        
        residuals = mse
        return residuals.mean()
    