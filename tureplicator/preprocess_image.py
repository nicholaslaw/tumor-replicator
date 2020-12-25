import cv2
import numpy as np
from skimage.morphology import erosion

def fill_holes(img):
    im = img.copy().astype("uint8")
    h, w = img.shape[:2]
    im_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im, im_mask, (0, 0), 255)
    im_inv = cv2.bitwise_not(im)
    
    return img | im_inv

def filter_coords(tumor_mid_coords, spawn_coords, tol=0.6):
    """
    PARAMS
    ==========
    tumor_mid_coords: list or tuple
        coordinate of tumor's mid point
    spawn_coords: list or tuple
        each element is a candidate pair of coordinates to spawn new tumors
    tol: float
        range between 0 and 1

    RETURNS
    ===========
    list of coordinates which has euclidean distance above tol-th percentile
    """

    result = []
    dist_list = []

    for x, y in spawn_coords:
        euclidean_distance = np.linalg.norm(np.array(tumor_mid_coords) - np.array([x, y]))
        result.append([x, y, euclidean_distance])
        dist_list.append(euclidean_distance)

    distance_min = min(dist_list)
    distance_diff = max(dist_list) - distance_min
    distance_threshold = distance_min + tol * distance_diff

    return list(map(lambda i: i[:-1], filter(lambda j: j[-1] > distance_threshold, result)))