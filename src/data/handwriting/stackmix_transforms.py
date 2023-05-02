import albumentations as A
from augmixations import HandWrittenBlot
import cv2
import numpy as np


class AlbuHandWrittenBlot(A.DualTransform):
    def __init__(self, hwb, always_apply=False, p=0.5):
        super(AlbuHandWrittenBlot, self).__init__(always_apply, p)
        self.hwb = hwb

    def apply(self, image, **params):
        return self.hwb(image)


def get_blot_transforms():
    min_h, max_h = 50, 100
    config = {'blot': {
                'params': {
                    'min_h': min_h,
                    'min_w': 10 * 2,
                    'max_h': max_h,
                    'max_w': 50 * 2,
                    'min_shift': 10,
                    'max_shift': 50,
                }
            },}
    bp = config['blot']['params']
    return A.OneOf([
        AlbuHandWrittenBlot(HandWrittenBlot(
            {
                'x': (None, None),
                'y': (None, None),
                'h': (bp['min_h'], bp['max_h']),
                'w': (bp['min_w'], bp['max_w']),
            }, {
                'incline': (bp['min_shift'], bp['max_shift']),
                'intensivity': (0.75, 0.75),
                'transparency': (0.05, 0.4),
                'count': i,
            }), p=1) for i in range(1, 11)
    ], p=0.5)

def resize_if_need(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    coef = 1 if img_h <= max_h and img_w <= max_w else max(img_h / max_h, img_w / max_w)
    h = int(img_h / coef)
    w = int(img_w / coef)
    img = cv2.resize(img, (w, h))
    return img, coef


def make_img_padding(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    bg = np.zeros((max_h, max_w, img_c), dtype=np.uint8)
    x1 = 0
    y1 = (max_h - img_h) // 2
    x2 = x1 + img_w
    y2 = y1 + img_h
    bg[y1:y2, x1:x2, :] = img.copy()
    return bg


def get_stackmix_image(img, augment=True):
    img, coef = resize_if_need(img, 128, 2048)
    img = make_img_padding(img, 128, 2048)

    if augment:
        transforms = A.Compose([
                    get_blot_transforms(),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.25, always_apply=False),
                    A.Rotate(limit=3, interpolation=1, border_mode=0, p=0.5),
                    A.JpegCompression(quality_lower=75, p=0.5),
        ], p=1.0)
        # image = self.transforms(image=image)['image']
        img = transforms(image=img)['image']
        return img
    else:
        return img


