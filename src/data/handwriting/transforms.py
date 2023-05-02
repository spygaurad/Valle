import itertools
import numpy as np
from skimage import transform as stf
from numpy import random, floor
from PIL import Image, ImageOps
from cv2 import erode, dilate
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from torchvision.transforms import RandomPerspective
import torchvision

"""
Each transform class defined here takes as input a PIL Image and returns the modified PIL Image
"""


class SignFlipping:
    """
    Color inversion
    """

    def __init__(self):
        pass

    def __call__(self, x):
        return ImageOps.invert(x)


class DPIAdjusting:
    """
    Resolution modification
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        w, h = x.size
        self.factor_w = self.factor
        self.factor_h = self.factor

        return x.resize((int(np.ceil(w * self.factor)), int(np.ceil(h * self.factor))), Image.BILINEAR)

        usage_prob = random.random() > 0.5
        height_prob = random.random() > 0.5

        if usage_prob:
            return x.resize((int(np.ceil(w * self.factor)), int(np.ceil(h * self.factor))), Image.BILINEAR)
        else:
            if height_prob:
                self.factor_h = self.factor + random.uniform(0, 0.25)
                return x.resize((int(np.ceil(w * self.factor_w)), int(np.ceil(h * self.factor_h))), Image.BILINEAR)
            else:
                self.factor_w = self.factor + random.uniform(0, 0.25)
                return x.resize((int(np.ceil(w * self.factor_w)), int(np.ceil(h * self.factor_h))), Image.BILINEAR)


class Dilation:
    """
    OCR: stroke width increasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(dilate(np.array(x), self.kernel, iterations=self.iterations))


class Erosion:
    """
    OCR: stroke width decreasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(erode(np.array(x), self.kernel, iterations=self.iterations))


class ElasticDistortion:
    """
    Elastic Distortion adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """
    def __init__(self, grid, magnitude, min_sep):

        self.grid_width, self.grid_height = grid
        self.xmagnitude, self.ymagnitude = magnitude
        self.min_h_sep, self.min_v_sep = min_sep

    def __call__(self, x):
        w, h = x.size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []
        shift = [[(0, 0) for x in range(horizontal_tiles)] for y in range(vertical_tiles)]

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

                sm_h = min(self.xmagnitude, width_of_square - (self.min_h_sep + shift[vertical_tile][horizontal_tile - 1][
                    0])) if horizontal_tile > 0 else self.xmagnitude
                sm_v = min(self.ymagnitude, height_of_square - (self.min_v_sep + shift[vertical_tile - 1][horizontal_tile][
                    1])) if vertical_tile > 0 else self.ymagnitude

                dx = random.randint(-sm_h, self.xmagnitude)
                dy = random.randint(-sm_v, self.ymagnitude)
                shift[vertical_tile][horizontal_tile] = (dx, dy)

        shift = list(itertools.chain.from_iterable(shift))

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for id, (a, b, c, d) in enumerate(polygon_indices):
            dx = shift[id][0]
            dy = shift[id][1]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        self.generated_mesh = generated_mesh

        return x.transform(x.size, Image.MESH, self.generated_mesh, resample=Image.BICUBIC)


class RandomTransform:
    """
    Random Transform adapted from https://github.com/IntuitionMachines/OrigamiNet
    Used in "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold",
        Yousef, Mohamed and Bishop, Tom E., The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020
    """
    def __init__(self, val):

        self.val = val

    def __call__(self, x):
        w, h = x.size

        dw, dh = (self.val, 0) if random.randint(0, 2) == 0 else (0, self.val)

        def rd(d):
            return random.uniform(-d, d)

        def fd(d):
            return random.uniform(-dw, d)

        # generate a random projective transform
        # adapted from https://navoshta.com/traffic-signs-classification/
        tl_top = rd(dh)
        tl_left = fd(dw)
        bl_bottom = rd(dh)
        bl_left = fd(dw)
        tr_top = rd(dh)
        tr_right = fd(min(w * 3 / 4 - tl_left, dw))
        br_bottom = rd(dh)
        br_right = fd(min(w * 3 / 4 - bl_left, dw))

        tform = stf.ProjectiveTransform()
        tform.estimate(np.array((
            (tl_left, tl_top),
            (bl_left, h - bl_bottom),
            (w - br_right, h - br_bottom),
            (w - tr_right, tr_top)
        )), np.array((
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        )))

        # determine shape of output image, to preserve size
        # trick take from the implementation of skimage.transform.rotate
        corners = np.array([
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ])

        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.around((out_rows, out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = stf.SimilarityTransform(translation=translation)
        tform = tform4 + tform
        # normalize
        tform.params /= tform.params[2, 2]

        x = stf.warp(np.array(x), tform, output_shape=output_shape, cval=255, preserve_range=True)
        x = stf.resize(x, (h, w), preserve_range=True).astype(np.uint8)

        return Image.fromarray(x)


def apply_data_augmentation(img):
    aug = {
                "dpi": {
                    "proba": 0.2,
                    "min_factor": 0.75,
                    "max_factor": 1,
                },
                "perspective": {
                    "proba": 0.2,
                    "min_factor": 0,
                    "max_factor": 0.3,
                },
                "elastic_distortion": {
                    "proba": 0.2,
                    "max_magnitude": 20,
                    "max_kernel": 3,
                },
                "random_transform": {
                    "proba": 0.2,
                    "max_val": 125,
                },
                "dilation_erosion": {
                    "proba": 0.2,
                    "min_kernel": 1,
                    "max_kernel": 3,
                    "iterations": 1,
                },
                "brightness": {
                    "proba": 0.2,
                    "min_factor": 0.01,
                    "max_factor": 1,
                },
                "contrast": {
                    "proba": 0.2,
                    "min_factor": 0.01,
                    "max_factor": 1,
                },
                "sign_flipping": {
                    "proba": 0.2,
                },
            }
    # Apply data augmentation
    if "dpi" in aug.keys() and np.random.rand() < aug["dpi"]["proba"]:
        factor = np.random.uniform(aug["dpi"]["min_factor"], aug["dpi"]["max_factor"])
        img = DPIAdjusting(factor)(img)
    if "perspective" in aug.keys() and np.random.rand() < aug["perspective"]["proba"]:
        scale = np.random.uniform(aug["perspective"]["min_factor"], aug["perspective"]["max_factor"])
        img = RandomPerspective(distortion_scale=scale, p=1, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=255)(img)
    elif "elastic_distortion" in aug.keys() and np.random.rand() < aug["elastic_distortion"]["proba"]:
        magnitude = np.random.randint(1, aug["elastic_distortion"]["max_magnitude"] + 1)
        kernel = np.random.randint(1, aug["elastic_distortion"]["max_kernel"] + 1)
        magnitude_w, magnitude_h = (magnitude, 1) if np.random.randint(2) == 0 else (1, magnitude)
        img = ElasticDistortion(grid=(kernel, kernel), magnitude=(magnitude_w, magnitude_h), min_sep=(1, 1))(
            img)
    elif "random_transform" in aug.keys() and np.random.rand() < aug["random_transform"]["proba"]:
        img = RandomTransform(aug["random_transform"]["max_val"])(img)
    if "dilation_erosion" in aug.keys() and np.random.rand() < aug["dilation_erosion"]["proba"]:
        kernel_h = np.random.randint(aug["dilation_erosion"]["min_kernel"],
                                        aug["dilation_erosion"]["max_kernel"] + 1)
        kernel_w = np.random.randint(aug["dilation_erosion"]["min_kernel"],
                                        aug["dilation_erosion"]["max_kernel"] + 1)
        if np.random.randint(2) == 0:
            img = Erosion((kernel_w, kernel_h), aug["dilation_erosion"]["iterations"])(img)
        else:
            img = Dilation((kernel_w, kernel_h), aug["dilation_erosion"]["iterations"])(img)
    if "contrast" in aug.keys() and np.random.rand() < aug["contrast"]["proba"]:
        factor = np.random.uniform(aug["contrast"]["min_factor"], aug["contrast"]["max_factor"])
        img = adjust_contrast(img, factor)
    if "brightness" in aug.keys() and np.random.rand() < aug["brightness"]["proba"]:
        factor = np.random.uniform(aug["brightness"]["min_factor"], aug["brightness"]["max_factor"])
        img = adjust_brightness(img, factor)
    # if "sign_flipping" in aug.keys() and np.random.rand() < aug["sign_flipping"]["proba"]:
    #     img = SignFlipping()(img)
    # # convert to numpy array

    # img = np.array(img)
    
    return img
