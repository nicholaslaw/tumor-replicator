import numpy as np
import SimpleITK as sitk
import cv2
from PIL import Image
from skimage.morphology import erosion
from .preprocess_image import fill_holes, filter_coords
from typing import Union

class TumorReplicator:
    def __init__(self, tumor_box_size: int=50, skull_threshold: int=190, outer_erosion: int=50, inner_erosion: int=5,
                spawn_tol: float=0.6, overlap_range: Union[list, tuple]=(0.4,0.5), dist_transform_mask_size: int=5, dist_transform_scalar: float=1.5, seed: int=42):
        """
        PARAMS
        ===========
        tumor_box_size: int
            length/breadth of box containing extracted tumor from original image
        skull_threshold: int
            threshold used to obtain skull mask
        outer_erosion: int
            number of loops to obtain outer boundary mask through erosion
        inner_erosion: int
            number of loops to obtain inner boundary mask through erosion
        spawn_tol: float
            float between 0 and 1 such that the new tumor will be spawned at a particular percentile of distance away
        dist_transform_mask_size: int
            maskSize argument of cv2.distanceTransform
        dist_transform_scalar: float
            weight of new tumor in merging with original image
        seed: int
            random seed for reproducibility
        """
        self.tumor_box_size = tumor_box_size
        self.skull_threshold = skull_threshold
        self.outer_erosion_loops = outer_erosion
        self.inner_erosion_loops = inner_erosion

        self.box = {"width": None, "height": None, "midpt": None} # this is filled up in self._extract_tumor_box
        self.spawn_tol = spawn_tol
        self.overlap_range = {"min": overlap_range[0], "max": overlap_range[1]}

        self.dist_transform_mask_size = dist_transform_mask_size
        self.dist_transform_scalar = dist_transform_scalar

        self.seed = seed

    def generate(self, tumor: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.seed:
            np.random.seed(self.seed)
        tumor_only = self._extract_tumor(tumor, mask)
        tumor_box = self._extract_tumor_box(tumor_only) # box containing tumor
        tumor_box = self._rotate_tumor_box(tumor_box) # randomly rotate box containing tumor
        skull_mask = self._extract_skull(tumor) # skull mask with holes filled
        boundary_mask = self._extract_boundary(skull_mask, self.outer_erosion_loops) # outer boundary
        boundary_mask, inner_coords = self._get_spawn_area(skull_mask, boundary_mask) # spawning reference for mid pt of tumor
        spawn_coords = self._get_spawn_coords(boundary_mask) # candidate spawn coordinates for new tumor
        new_tumor = self.generate_tumor_only(tumor_only, spawn_coords, tumor_box, inner_coords) # newly generated tumor (just the tumor)
        new_image = self.merge(skull_mask, new_tumor, tumor) # new image with 2 tumors

        return new_image

    def _extract_tumor(self, tumor: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        PARAMS
        ===========
        tumor: 2D numpy array
            original tumor image as an array
        mask: 2D numpy array
            mask highlighting the tumor, only has values of 1s and 0s

        RETURNS
        ==========
        2D array with just the tumor from the original image
        """
        return tumor * mask

    def _extract_tumor_box(self, tumor_only: np.ndarray) -> np.ndarray:
        """
        PARAMS
        ===========
        tumor_only: numpy array
            2D array with just the tumor from the original image
        
        RETURNS
        ===========
        2D array which is a box containing the tumor with some space
        
        """
        nonzero_px = np.argwhere(tumor_only!=0) # coordinates of nonzero pixels
        x_max, y_max = nonzero_px.max(axis=0) # max coordinates of x and y
        x_min, y_min = nonzero_px.min(axis=0) # min coordinates of x and y

        x_min -= self.tumor_box_size
        x_max += self.tumor_box_size

        y_min -= self.tumor_box_size
        y_max += self.tumor_box_size

        self.box["width"] = x_max - x_min
        self.box["height"] = y_max - y_min
        self.box["midpt"] = ((x_max + x_min) // 2, (y_max + y_min) // 2) # coordinates of tumor's midpt

        return tumor_only[x_min:x_max, y_min:y_max].copy()

    def _rotate_tumor_box(self, tumor_box: np.ndarray) -> np.ndarray:
        """
        PARAMS
        ===========
        tumor_box: numpy array
            2D array which is a box containing the tumor with some space

        RETURNS
        ===========
        2D array which is a rotated box containing the tumor with some space
        """
        return np.array(Image.fromarray(tumor_box).rotate(np.random.randint(low=0, high=360))) # random rotation of tumor

    def _extract_skull(self, tumor: np.ndarray) -> np.ndarray:
        """
        PARAMS
        ===========
        tumor: 2D numpy array
            original tumor image as an array

        RETURNS
        ===========
        2D array which is a skull mask with holes filled
        """
        skull_mask = 1 * (tumor > self.skull_threshold) # still has holes
        skull_mask = fill_holes(skull_mask)
        skull_mask[skull_mask > 1] = 1
        return skull_mask

    def _extract_boundary(self, skull_mask: np.ndarray, reps: int) -> np.ndarray:
        """
        PARAMS
        ===========
        skull_mask: numpy array
            2D array which is a skull mask with holes filled
        reps: int
            number of loops for erosion

        RETURNS
        ===========
        2D array which is the difference between the skull mask and the output of erosion
        a visual analogy for this would be a moat
        """
        inner = skull_mask.copy()
        for _ in range(reps):
            inner = erosion(inner)

        return skull_mask - inner

    def _get_spawn_area(self, skull_mask: np.ndarray, boundary_mask: np.ndarray):
        """
        PARAMS
        ===========
        skull_mask: numpy array
            2D array which is a skull mask with holes filled
        boundary_mask: numpy array
            2D array which is the difference between the skull mask and the output of erosion

        RETURNS
        ===========
        2D array which is a (nearly) circular boundary which is used as a reference for the
        mid point of the tumor for spawning

        Also, there will be a set of tuples whereby each tuple represents coordinates of nonzero pixels, and this is
        used to check whether the newly spawned tumour has a certain amount of overlap with the inner skull mask
        """
        inner_boundary_mask = skull_mask - boundary_mask # like an inner skull
        boundary_mask = self._extract_boundary(inner_boundary_mask, self.inner_erosion_loops)
        boundary_mask[boundary_mask > 1] = 1

        inner_coords = np.argwhere(inner_boundary_mask != 0)
        inner_coords = set(map(lambda i: tuple(i), inner_coords))

        return boundary_mask, inner_coords

    def _get_spawn_coords(self, boundary_mask: np.ndarray) -> list:
        """
        PARAMS
        ===========
        boundary_mask: numpy array
            2D array which is a (nearly) circular boundary which is used as a reference for the
            mid point of the tumor for spawning

        RETURNS
        ===========
        list of lists where each element is a candidate pair of coordinates for new tumor to be spawned at
        """
        spawn_coords = np.argwhere(boundary_mask != 0)
        spawn_coords = filter_coords(self.box["midpt"], spawn_coords, tol=self.spawn_tol)
        return spawn_coords

    def generate_tumor_only(self, tumor_only: np.ndarray, spawn_coords: Union[list, tuple], tumor_box: np.ndarray, inner_coords: Union[set, list, tuple]) -> np.ndarray:
        """
        PARAMS
        ===========
        tumor_only: numpy array
            2D array with just the tumor from the original image
        spawn_coords: list of lists
            list of lists where each element is a candidate pair of coordinates for new tumor to be spawned at
        tumor_box: numpy array
            2D array which is a rotated box containing the tumor with some space
        inner_coords: set of tuples
            each tuple represents coordinates of nonzero pixel for the inner skull mask

        RETURNS
        ===========
        2D numpy array with just the new tumor
        """
        new_tumor = np.zeros_like(tumor_only)

        while True:
            new_tumor = np.zeros_like(tumor_only)
            
            spawn_x_y = spawn_coords[np.random.randint(low=0, high=len(spawn_coords))]

            new_tumor[int(spawn_x_y[0] - self.box["width"] // 2):int(spawn_x_y[0] - self.box["width"] // 2) + tumor_box.shape[0],
                    int(spawn_x_y[1] - self.box["height"] // 2):int(spawn_x_y[1] - self.box["height"] // 2) + tumor_box.shape[1]] = tumor_box
                    
            new_tumor_coords = np.argwhere(new_tumor != 0)
            new_tumor_coords = set(map(lambda i: tuple(i), new_tumor_coords))
            
            if self.overlap_range["min"] < len(inner_coords & new_tumor_coords) / len(new_tumor_coords) < self.overlap_range["max"]:
                break

        return new_tumor

    def merge(self, skull_mask: np.ndarray, new_tumor: np.ndarray, tumor: np.ndarray) -> np.ndarray:
        """
        PARAMS
        ===========
        skull_mask: numpy array
            2D array which is a skull mask with holes filled
        new_tumor: numpy array
            2D numpy array with just the new tumor
        """
        new_mask = skull_mask * new_tumor
        new_mask[new_mask != 0] = 1
        new_mask = new_mask.astype("uint8")
        dist_transform = cv2.distanceTransform(new_mask, cv2.DIST_L2, self.dist_transform_mask_size)
        dist_transform /= dist_transform.max()

        result = (1 - dist_transform) * tumor + dist_transform * skull_mask * new_tumor * self.dist_transform_scalar
        result = result.astype(int)

        return result