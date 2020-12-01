import numpy as np
import SimpleITK as sitk
import cv2
from PIL import Image

from .preprocess_image import fill_holes, get_boundary, filter_coords

class TumorReplicator:
    def __init__(self, tumor_box_size=50, skull_threshold=190):
        self.tumor_box_size = tumor_box_size
        self.skull_threshold = skull_threshold

    def fit(self, tumor, mask):
        tumor_only = self._extract_tumor(tumor, mask)
        tumor_box = self._extract_tumor(tumor_only)
        tumor_box = self._rotate_tumor_box(tumor_box)

    def _extract_tumor(self, tumor, mask):
        """
        PARAMS
        ===========
        tumor: 2D numpy array
            original tumor image as an array
        mask: 2D numpy array
            mask highlighting the tumor, only has values of 1s and 0s

        RETURNS
        ==========
        array with just the tumor from the original image
        """
        return tumor * mask

    def _extract_tumor_box(self, tumor_only):
        nonzero_px = np.argwhere(tumor_only!=0) # coordinates of nonzero pixels
        x_max, y_max = nonzero_px.max(axis=0) # max coordinates of x and y
        x_min, y_min = nonzero_px.min(axis=0) # min coordinates of x and y

        x_min -= self.tumor_box_size
        x_max += self.tumor_box_size

        y_min -= self.tumor_box_size
        y_max += self.tumor_box_size

        return tumor_only[x_min:x_max, y_min:y_max].copy()
        tumor_box = np.array(Image.fromarray(tumor_box).rotate(np.random.randint(low=0, high=360))) # random rotation of tumor

    def _rotate_tumor_box(self, tumor_box):
        return np.array(Image.fromarray(tumor_box).rotate(np.random.randint(low=0, high=360)))