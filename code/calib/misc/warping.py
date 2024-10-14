import numpy as np
import sys
from pathlib import Path
import cv2 as cv
from misc.tools import dict_update
from misc import misc
import copy

# Prpject path setup
_root_path = Path(__file__).resolve().parent / "../../../"
_root_path = _root_path.resolve()
sys.path.append(str(_root_path / "modules"))
import projections.mappings as mp

def sample_warp(input_size, input_afov=90, config=None):

    input_des = mp.Perspective_Description(
        width=input_size[1],
        height=input_size[0],
        intrinsics=dict(afov=input_afov),
    )

    warp = random_warp(**config)


def warp_image(image:np.ndarray, warp:mp.ImageDescription, input_des):
    warped = mp.map_img(image, [input_des, warp])
    return warped

def warp_kps(kps:np.ndarray, warp:mp.ImageDescription, input_des):
    kps = (kps[:,0], kps[:,1])
    warped_kps = mp.map_points(kps, [input_des, warp])
    mask = warped_kps[2]
    warped_kps = np.stack(warped_kps[:2]).T
    return warped_kps, mask

def random_warp(out_size=(500,500), range_f = (200,500), range_a = (0,1), 
        range_xi = (-0.5, 0.5), range_r = (180,10,10)):
    
    f = np.random.default_rng().uniform(*range_f)

    while True:
        a = np.random.default_rng().uniform(*range_a)
        xi = np.random.default_rng().uniform(*range_xi)
        if xi < 0 and a < 0.4: # Avoid pincushion distortion
            continue
        break

    max_r = np.asarray(range_r)
    min_r = -max_r

    r = np.random.default_rng().uniform(min_r, max_r)


    return mp.FisheyeDS_Description(
        width=out_size[1],
        height=out_size[0],
        intrinsics=dict(
            f = f,
            a = a,
            xi = xi,
        ),
        extrinsic_rot=r,
    )

def valid_mask(warp:mp.ImageDescription, input_des:mp.ImageDescription, erosion_radius):
    base_mask = np.ones((input_des.height, input_des.width), dtype = np.uint8) 

    mask = np.zeros((warp.height + 2, warp.width + 2), dtype = np.uint8)
    mask[1:-1,1:-1] = mp.map_img(base_mask, [input_des, warp], interpolation= cv.INTER_NEAREST)[0]

    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_radius*2,)*2)
    
    mask = cv.erode(mask, element) 
    mask = mask[1:-1,1:-1]

    return mask


def filter_kps(kps:np.ndarray, mask:np.ndarray):
    if len(kps.shape) != 2 or kps.shape[0] ==0:
        return kps

    bounds = np.asarray(mask.shape[:2])
    kps = np.round(kps).astype(np.int32)

    inliers = (kps[:,0]>=0) & (kps[:,0]<bounds[1]) & (kps[:,1]>=0) & (kps[:,1]<bounds[0]) 
    kps_in_bounds = kps[inliers]
    in_mask = mask[kps_in_bounds[:,1], kps_in_bounds[:,0]] > 0 # Row, col - y,x
    return kps_in_bounds[in_mask]



warping_adaptation_default_config = {
    'num': 1,
    'aggregation': 'sum',
    'valid_border_margin': 3,
    'warps': { 
        'out_size' : (500,500), # Unused, same as des, H,W
        'range_f' : (-50, 50), # Changes relative to the original des
        'range_a' : (-0.1, .1), # Changes relative to the original des
        'range_xi' : (-0.1, 0.1), # Changes relative to the original des
        'range_r' : (180,10,10),  # Relative Roll, Pitch, Yaw
    },
    'filter_counts': 0
}

def update_relative_warp(relative_warp:mp.FisheyeDS_Description, base_warp:mp.FisheyeDS_Description):
    relative_warp.width = base_warp.width
    relative_warp.height = base_warp.height
    relative_warp.a_ += base_warp.a_
    relative_warp.a_ = np.clip(relative_warp.a_, 0, 1)
    relative_warp.xi_ += base_warp.xi_
    relative_warp.xi_ = np.clip(relative_warp.xi_, -.5, .5)
    relative_warp.f += base_warp.f

def warping_adaptation(image:np.ndarray, des:mp.FisheyeDS_Description, net, config):
    """Perfoms homography adaptation.
    Inference using multiple random warped patches of the same input image for robust
    predictions.
    Arguments:
        image: A `Tensor` with shape `[N, H, W, 1]`.
        net: A function that takes an image as input, performs inference, and outputs the
            prediction dictionary.
        config: A configuration dictionary containing optional entries such as the number
            of sampled homographies `'num'`, the aggregation method `'aggregation'`.
    Returns:
        A dictionary which contains the aggregated detection probabilities.
    """
    # Remove batch dim
    probs = net(image)['prob'][0].cpu().detach().numpy()

    counts = valid_mask(des, des, config['valid_border_margin'])

    probs = np.expand_dims(probs * counts, axis=-1)
    counts = np.expand_dims(counts, axis=-1)
    images = np.expand_dims(image, axis=-1)

    config = dict_update(copy.deepcopy(warping_adaptation_default_config), config)

    des.extrinsic_rot = (0,0,0) # Generated warps are relative to the image

    def step(probs, counts, images):
        # Sample image patch
        warp = random_warp(**config['warps']) 
        # This warp intrinsics are relative to the input des intrinsics
        update_relative_warp(warp, des)

        warped = mp.map_img(image, [des, warp])[0]

        mask = valid_mask(warp, des, config['valid_border_margin'])
        count = counts[..., 0] & mp.map_img(mask, [warp, des], interpolation=cv.INTER_NEAREST)[0]

        
        # count is the reprojection of the valid mask in the original frame, 
        # thus contains 1 if the pixel in the original image is a valid pixel in 
        # the warped one.

        # Predict detection probabilities
        prob = net(warped)['prob'][0].cpu().detach().numpy()
        prob = prob * mask  #TODO useless?
        prob_proj = mp.map_img(prob, [warp, des])[0]
        # reproject probabilities in the base frame
        prob_proj = prob_proj * count

        probs = np.concatenate([probs, np.expand_dims(prob_proj, -1)], axis=-1)
        counts = np.concatenate([counts, np.expand_dims(count, -1)], axis=-1)
        images = np.concatenate([images, np.expand_dims(warped, -1)], axis=-1)
        return probs, counts, images

    for _ in range(config['num']):
        probs, counts, images = step(probs, counts, images)

    H_counts = counts
    counts = np.sum(counts, axis=-1)
    max_prob = np.max(probs, axis=-1)
    mean_prob = np.sum(probs, axis=-1) / (counts + 1e-5)

    if config['aggregation'] == 'max':
        prob = max_prob
    elif config['aggregation'] == 'sum':
        prob = mean_prob
    else:
        raise ValueError('Unkown aggregation method: {}'.format(config['aggregation']))

    if config['filter_counts']:
        prob = np.where(counts >= config['filter_counts'],
                        prob, np.zeros_like(prob))

    return {'prob': prob, 'counts': counts,
            'mean_prob': mean_prob, 'input_images': images, 'H_probs': probs, 'H_counts':H_counts}  # debug

