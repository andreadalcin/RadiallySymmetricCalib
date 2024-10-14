import numpy as np

def expand_image(img, margin):
    size = img.shape[:2]
    new_size = [int(size[0] + margin*2) , int(size[1] + margin*2)]

    new_img = np.zeros(new_size + [3], dtype=np.uint8)
    new_img[margin:-margin,margin:-margin,:] = img
    return new_img