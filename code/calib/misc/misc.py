import numpy as np
from typing import Tuple, List
import cv2 as cv
import json
import torch
from torchvision.ops import nms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib as mpl
import projections.mappings as mp
import projections.cameras as cams
from projections.proj2d import Cart2Polar, Cart2Polar_v2

class Tupperware(dict):
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Tupperware):
            value = Tupperware(value)
        super(Tupperware, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, Tupperware.MARKER)
        if found is Tupperware.MARKER:
            found = Tupperware()
            super(Tupperware, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__

def keypoint_detection(image:np.ndarray, polar_width:int) -> np.ndarray:
    # orb = cv.ORB_create()
    # kp = orb.detect(image,None)

    imagegray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    features = cv.SIFT_create()
  
    keypoints = features.detect(imagegray, None)

    c2p = mp.Cart2Polar()
    c2p.set_sizes_from_src_w(src_size=imagegray.shape[:2], d_width=polar_width)

    # Column vectors: (2, n)
    kps = cv.KeyPoint_convert(keypoints).T
    
    kps_polar = c2p.project([kps[0,:], kps[1,:]])
    kps_polar = np.vstack(kps_polar[:2])

    # # Visualization only
    # kps_polar = cv.KeyPoint_convert(points2f=kps_polar.T)

    # output_image = cv.drawKeypoints(imagegray, keypoints, 0, (255, 0, 0),
    #                              flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    
    # polar_img = mp.map_img_2d(img=imagegray, des=c2p)
    # polar_img_kps = cv.drawKeypoints(polar_img, kps_polar, 0, (255, 0, 0),
    #                              flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    # cv.imwrite("polar.png",polar_img)
    # cv.imwrite("out.png",output_image)
    # cv.imwrite("polar_key.png",polar_img_kps)

    return kps_polar

def keypoint_va_weighting(image:np.ndarray, img_width:int, polar_width:int) -> np.ndarray:
    
    image_resized = cv.resize(image, dsize=(img_width, img_width))
    kps_polar = keypoint_detection(image_resized, polar_width)
    rhos = kps_polar[1,:] # only rho ravlues are interesting

    # Distribution of the features vary based on the rho we are considering: the lenght of the circle should be considered
    counts = np.zeros((image_resized.shape[0]//2+1,))
    for rho in rhos:
        rho_int = int(np.round(rho,decimals=0))
        if rho_int < counts.shape[0]:
            counts[rho_int] += 1

    counts_scaled = counts.copy()
    counts_scaled[1:] /= (2*np.pi*np.arange(1,img_width//2+1))
    weights = counts_scaled
    return weights


def tensor_to_heatmap(heatmap: torch.Tensor, colormap = 'jet', rescale = True) -> torch.Tensor:
    """Generate heatmap

    Parameters
    ----------
    heatmap : torch.Tensor
        Shape (1, W, H), conent will be resized between 0 and 1.

    Returns
    -------
    torch.Tensor
        Output heatmap with color jet in RGBA, shape (4, W, H)
    """
    if len(heatmap.shape) == 2:
        heatmap = heatmap[None, ...] 
    
    if rescale:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)

    x = heatmap.detach().cpu().numpy()
    x = np.squeeze(x, 0)
    
    x_transformed = mpl.colormaps[colormap](x)
    x_transformed = np.transpose(x_transformed, (-1,0,1))
    return torch.from_numpy(x_transformed)


def build_c2p_grid(input_height:int, input_width:int, polar_width:int, batch_size:int, inverse:bool=False, v2=False, polar_pad = 2) -> torch.Tensor:
    c2p = Cart2Polar_v2() if v2 else Cart2Polar()
    c2p.set_sizes_from_src_w(
        src_size=(input_height, input_width),
        d_width=polar_width,
        )

    if inverse:
        H, W = c2p.s_height, c2p.s_width
        H_ref, W_ref = c2p.d_height, c2p.d_width
    else:
        H, W = c2p.d_height, c2p.d_width
        H_ref, W_ref = c2p.s_height, c2p.s_width
    u,v = np.meshgrid(np.arange(0,int(W)),np.arange(0,int(H)),indexing="xy")

    map_x, map_y, _ = c2p.project([u,v], inverse= not inverse)# theta, rho, mask
    
    grid = torch.stack(
        (torch.tensor(map_x, dtype=torch.float32), torch.tensor(map_y, dtype=torch.float32)), dim=-1) # H x W x 2
    grid = grid.tile((batch_size, 1, 1, 1)) # N x H x W x 2

    # scale grid to [-1,1]

    w_offset = 1.0
    if inverse:
        mult = W_ref / (W_ref + 2*polar_pad)
        w_offset = 1.0 * mult
        W_ref = W_ref + 2*polar_pad # TO THEN USE POLAR CIRCULAR PADDING AVOIDING ARTIFACTS
    grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / W_ref - w_offset
    grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / H_ref - 1.0

    return grid

def polar_grid_sample(x:torch.Tensor, grid:torch.Tensor, border_remove:int = 3, mode='bilinear') -> torch.Tensor:
    # heatmap: N x 1 x H x W
    # grid: N x H x W x 2
    polar_x = F.grid_sample(x, grid, mode=mode, align_corners=True) # N x 1 x Hp x Wp
    # min_value = polar_x.min()
    # temp = polar_x.detach().clone()
    # temp[:,:,-border_remove:,:] = min_value
    # return temp
    if border_remove > 0:
        polar_x[:,:,-border_remove:,:] = 0

    return polar_x


def cartesian_grid_sample(x:torch.Tensor, grid:torch.Tensor, mode='bilinear', pad=2) -> torch.Tensor:
    # x: N x 1 x H x W
    # grid: N x H x W x 2

    x_padded = F.pad(x, (pad,pad,0,0), mode='circular')

    cart_x = F.grid_sample(x_padded, grid, mode=mode, align_corners=False) # N x 1 x Hp x Wp

    return cart_x


def build_weights(polar_heatmap:torch.Tensor) -> torch.Tensor:
    # heatmap: N x 1 x Hp x Wp
    weights = torch.sum(polar_heatmap, dim=-1) # N x 1 x Wp
    
    norm = torch.max(weights, dim=-1, keepdim=True)[0] + 0.000001
    weights_normalized = weights / norm # N x 1 x Wp
    weights_normalized = weights_normalized.squeeze(dim=1) # N x Wp

    return weights_normalized

def flat_tensor_to_3d(x: torch.Tensor, height:int = 30, cell_size:int=2) -> torch.Tensor:
    # input shape: (N,)
    # out shape: (1,H,N*cell_size)
    x = x[None,None,:]  # shape: (1,1,N)
    x = torch.repeat_interleave(x, cell_size, dim=-1) # shape: (1,1,N*cell_size)
    x = x.expand((-1,height,-1)) # shape: (1,H,N*cell_size)
    return x

def tensor_nCH_to_3CH(x: torch.Tensor) -> torch.Tensor:
    # input shape: (n, ...)
    # out shape: (3, ...)
    n = x.shape[0]
    dims = len(x.shape)
    if n ==1:
        return x.expand((3,) + (-1,)*(dims-1))
    if n==3:
        return x
    if n==4:
        return x[:3,...]
    
    raise NotImplementedError()

def read_woodscape_description_from_json(path):
    """generates a Camera object from a json file"""
    with open(path) as f:
        config = json.load(f)

    intrinsic = config['intrinsic']
    coefficients = [intrinsic['k1'], intrinsic['k2'], intrinsic['k3'], intrinsic['k4']]

    return cams.FisheyePOLY_Description(
        height=intrinsic['height'],
        width=intrinsic['width'],
        principle_point=(intrinsic['cx_offset'], intrinsic['cy_offset']),
        intrinsics=dict(distortion_coeffs=coefficients)
    )

def space_to_depth(x:torch.Tensor, block_size:int) -> torch.Tensor:
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

def depth_to_space(x: torch.Tensor) -> torch.Tensor:
    # x: N x cell_size^2 x Hc x Wc
    # out: N x 1 x Hc*cell_size x Wc*cell_size
    cell_size = int(np.sqrt(x.shape[1]))
    assert cell_size**2 == x.shape[1]
    Hc = x.shape[2]
    Wc = x.shape[3]
    x = x.permute((0, 2, 3, 1)) # N x Hc x Wc x cell_size^2
    x = x.reshape((-1, Hc, Wc, cell_size, cell_size)) # N x H/8 x W/8 x 8 x 8
    x = x.permute((0, 1, 3, 2, 4)) # N x H/8 x 8 x W/8 x 8
    x = x.reshape((-1, 1, Hc*cell_size, Wc*cell_size)) # N x 1 x H x W
    return x

def logits_to_probabilities(x: torch.Tensor) -> torch.Tensor:
    # x: N x cell_size^2+1 x Hc x Wc
    prob = torch.softmax(x, dim=1)
    # Strip the extra “no interest point” dustbin
    prob = prob[:, :-1, :, :]
    prob = depth_to_space(prob)
    prob = torch.squeeze(prob, dim=1)
    return prob 
    # out: N x Hc*cell_size x Wc*cell_size

def crop_c2p_grid(grid:torch.Tensor, polar_height:int):
    # grid: N x Hp x Wp x 2
    # Grid should be top cropped to: N x polar_height x Wp x 2
    
    ratio = grid.shape[1] / polar_height
    cropped_grid = grid.clone()
    cropped_grid = cropped_grid[:,:polar_height,:,:] # N x polar_height x Wp x 2

    # scale grid to [-1,1]
    scaled_grid = ratio * cropped_grid
    return scaled_grid

def crop_p2c_grid(grid:torch.Tensor, cart_radius:int):
    # grid: N x Hc x Wc x 2
    # Grid should be center cropped to: N x cart_radius*2 x cart_radius*2 x 2

    pad_h = grid.shape[1] // 2 - cart_radius
    rem_h = grid.shape[1] % 2
    pad_w = grid.shape[2] // 2 - cart_radius
    rem_w = grid.shape[2] % 2

    ratio = float(min(grid.shape[1], grid.shape[1]) // 2) / float(cart_radius)
    cropped_grid = grid.clone()
    cropped_grid = cropped_grid[:,pad_h:-(pad_h+rem_h),pad_w:-(pad_w+rem_w),:] # N x H?*2 x H?*2 x 2

    # scale Y grid to [-1,1], keeping top at -1
    cropped_grid[:,:,:,1] = (ratio) * (cropped_grid[:,:,:,1] + 1) -1
    return cropped_grid


def box_nms(prob:torch.Tensor, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.

    Arguments:
        prob: the probability heatmap, with shape `[H, W]`.
        size: a scalar, the size of the bouding boxes.
        iou: a scalar, the IoU overlap threshold.
        min_prob: a threshold under which all probabilities are discarded before NMS.
        keep_top_k: an integer, the number of top scores to keep.
    """
    pts_x, pts_y = torch.where(torch.greater_equal(prob, min_prob))
    pts = torch.stack([pts_x, pts_y], -1).type(torch.float32)
    size = size/2.
    boxes = torch.cat([pts-size, pts+size], dim=1)
    
    scores = prob[pts_x, pts_y]

    indices = nms(boxes, scores, iou)
    pts_x, pts_y = pts_x[indices], pts_y[indices]
    scores = scores[indices]

    if keep_top_k:
        k = min(scores.shape[0], keep_top_k)
        scores, indices = torch.topk(scores, k)
        pts_x, pts_y = pts_x[indices], pts_y[indices]

    prob = torch.zeros_like(prob)
    prob[pts_x, pts_y] = scores
    
    return prob

def get_radial_mask(size, border:int = 2) -> np.ndarray:
    rad = (min(size)) / 2 - border
    grid = np.mgrid[:size[0],:size[1]].astype(np.float32)
    grid -= np.array([(size[0]-1)/2, (size[1]-1)/2]).reshape(-1,1,1)
    r_2 = grid[0,...] ** 2 + grid[1,...] ** 2
    mask = r_2 <= rad**2
    return mask

def draw_keypoints(img, corners=None, kps_map=None, color=(0, 255, 0)):
    if kps_map is not None:
        corners = np.stack(kps_map[::-1]).T
        
    corners = corners.astype(np.uint16)
    keypoints = cv.KeyPoint_convert(corners)
    return cv.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)

def mixed_padding(x:torch.Tensor, padding):
    """Circular padding on the width, zero on height"""
    # pad: (padding_left, padding_right, padding_top, padding_bottom)
    # padding: (H,W)
    # Circular left, right
    x = torch.nn.functional.pad(x, (padding[1], padding[1], 0, 0), mode='circular')

    # Pad top, bottom
    x = torch.nn.functional.pad(x, (0, 0, padding[0], 0), mode='constant', value=0)

    x = torch.nn.functional.pad(x, (0, 0, 0, padding[0]), mode='replicate')
    return x


def sample_va_vec(va_vec:torch.Tensor, cell_size = 8, start=4):
    idxs = torch.arange(start, va_vec.shape[-1], cell_size)
    vals = va_vec[...,idxs]
    return vals, idxs


if __name__ == '__main__':
    a = torch.rand((100,100))
    box_nms(a, 4, min_prob=0.8)

