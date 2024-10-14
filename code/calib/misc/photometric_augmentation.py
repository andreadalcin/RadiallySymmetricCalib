import cv2 as cv
import numpy as np


augmentations = [
        'additive_gaussian_noise',
        'additive_speckle_noise',
        'additive_shade',
        'motion_blur'
]


def additive_gaussian_noise(image:np.ndarray, stddev_range=[5, 50]):
    stddev = np.random.default_rng().uniform(*stddev_range)
    noise = np.random.default_rng().normal(size=image.shape, scale=stddev)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image


def additive_speckle_noise(image:np.ndarray, prob_range=[0.0, 0.005]):
    prob = np.random.default_rng().uniform(*prob_range)
    sample = np.random.default_rng().uniform(size= image.shape)
    noisy_image = np.where(sample <= prob, np.zeros_like(image), image)
    noisy_image = np.where(sample >= (1. - prob), 255.*np.ones_like(image), noisy_image)
    return noisy_image


def additive_shade(image:np.ndarray, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                   kernel_size_range=[250, 350]):

    min_dim = min(image.shape[:2]) / 4
    mask = np.zeros(image.shape[:2], np.uint8)
    for i in range(nb_ellipses):
        ax = int(max(np.random.rand() * min_dim, min_dim / 5))
        ay = int(max(np.random.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = np.random.randint(max_rad, image.shape[1] - max_rad)  # center
        y = np.random.randint(max_rad, image.shape[0] - max_rad)
        angle = np.random.rand() * 90
        cv.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

    transparency = np.random.uniform(*transparency_range)
    kernel_size = np.random.randint(*kernel_size_range)
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1
    mask = cv.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    shaded = image * (1 - transparency * mask[..., np.newaxis]/255.)
    shaded = np.clip(shaded, 0, 255)

    return np.reshape(shaded, image.shape)


def motion_blur(image:np.ndarray, max_kernel_size=10):

    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
    center = int((ksize-1)/2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    blurred = cv.filter2D(image, -1, kernel)
    
    return np.reshape(blurred, image.shape)
