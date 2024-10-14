import numpy as np
import misc.photometric_augmentation as photaug
import misc.warping as warping
import cv2 as cv

def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
            else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p


def add_dummy_valid_mask(data):
    valid_mask = np.ones(data['img'].shape[:2], dtype=np.int32)
    return {**data, 'valid_mask': valid_mask}

def add_keypoint_map(data):
    image_shape = np.asarray(data['img'].shape[:2])
    kp = np.minimum(np.round(data['kps']).astype(np.int32), image_shape-1)
    kmap = np.zeros(image_shape)
    if len(kp.shape) != 2 or kp.shape[0] ==0:
        return {**data, 'kps_map': kmap}
    kmap[kp[:,1],kp[:,0]] = 1
    return {**data, 'kps_map': kmap}


def photometric_augmentation(data, **config):
    primitives = parse_primitives(config['primitives'], photaug.augmentations)
    prim_configs = [config['params'].get(
                            p, {}) for p in primitives]

    indices = np.arange(len(primitives))
    if config['random_order']:
        np.random.default_rng().shuffle(indices)

    image = data['img']
    for i in indices:
        p, c = primitives[i], prim_configs[i]
        image = getattr(photaug, p)(image, **c)

    return {**data, 'img': image}


def warp_augmentation(data, **config):
    
    warp = warping.random_warp(**config['params'])

    data['warp'] = warp

    image = data['img']

    data['base_img'] = image
    data['base_kps'] = data['kps']

    input_des = warping.mp.Perspective_Description(
        width=image.shape[1],
        height=image.shape[0],
        intrinsics=dict(afov=config['input_afov']),
    )

    warped_img, _ = warping.warp_image(
        image= image,
        warp=warp,
        input_des=input_des,
    )

    warped_kps, _ = warping.warp_kps(
        kps=data['kps'],
        warp=warp,
        input_des=input_des,
    )
    
    
    data['valid_mask'] = warping.valid_mask(
        warp=warp,
        input_des=input_des,
        erosion_radius= config['valid_border_margin'],
    )

    warped_kps = warping.filter_kps(warped_kps, data['valid_mask'])
    
    return {**data, 'img': warped_img, 'kps':warped_kps}


def mask_and_resize(image: np.ndarray, **config):
    """Inscribe the image in a circle and resize it to the given size. """
    size = config['resize']
    valid_pad = config['valid_border_margin']

    shape = image.shape
    max_dim = max(shape)
    diam = int(np.ceil(max_dim * np.sqrt(2)))
    diam += (diam - max_dim) % 2
    mask = np.zeros((diam, diam), dtype = image.dtype)
    pad = (diam - max_dim) // 2

    out_img = mask.copy()
    mask[pad + valid_pad: pad + shape[0] - valid_pad*2, pad + valid_pad : pad + shape[1] - valid_pad*2] = 1
    out_img[pad:pad+shape[0], pad:pad+shape[1]] = image
    
    scale =  max(size) / diam 
    out_img = cv.resize(out_img, size)
    mask = cv.resize(mask, size, interpolation=cv.INTER_NEAREST)

    h_t = np.eye(3)
    h_t[:2,2] = pad

    h_s = np.diag([scale,scale,1])

    h = h_s @ h_t

    return out_img, mask, h
