import glob
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageFont
import torch
import html
import copy
import random
from torch.nn import functional as f
from torch.utils.data import Dataset
import torchaudio
# from torchvision.transforms import InterpolationMode, RandomPerspective
# from torchvision.transforms.functional import (adjust_brightness,
#                                                adjust_contrast, gaussian_blur,
#                                                rotate)
import albumentations as A
# from augmixations import HandWrittenBlot
from hwb import HandWrittenBlot
import sys
sys.path.append('/Users/spygaurad/multimodal_valle/Valle/src/generate_dataset')
from src.generate_dataset.encodec_wrapper import EncodecWrapper
from src.data.handwriting.batch_output import BatchOutput
# from src.data.handwriting.transforms import apply_data_augmentation
# from src.data.handwriting.transforms import Dilation, ElasticDistortion, Erosion, DPIAdjusting, apply_data_augmentation
# from src.data.handwriting.stackmix_transforms import  get_stackmix_image


TXT_LABEL = 0
IMG_LABEL = 1
IMG_TXT_LABEL = 2


aug = {"dpi": {
    "proba": 0.2,
    "min_factor": 0.75,
    "max_factor": 1.25,
},
    "perspective": {
    "proba": 0.2,
    "min_factor": 0,
    "max_factor": 0.2,
},
    "elastic_distortion": {
    "proba": 0.2,
    "max_magnitude": 10,
    "max_kernel": 3,
},
    "random_transform": {
    "proba": 0.2,
    "max_val": 125,
},
    "dilation_erosion": {
    "proba": 0.2,
    "min_kernel": 1,
    "max_kernel": 2,
    "iterations": 1,
},
    "brightness": {
    "proba": 0.2,
    "min_factor": 0.4,
    "max_factor": 1,
},
    "contrast": {
    "proba": 0.2,
    "min_factor": 0.5,
    "max_factor": 1,
},
    "sign_flipping": {
    "proba": 0.2,
},
}

ROTATION_MIN_MAX = 10  # it will rotate the image between -5 and 5
BLUR_SIGMA = 2
BLUR_KERNEL = 3 #5
MIN_PATCH_X = 5
MAX_PATCH_X = 20
MIN_PATCH_Y = 5
MAX_PATCH_Y = 60

# class AlbuHandWrittenBlot(A.DualTransform):
    # def __init__(self, hwb, always_apply=False, p=0.5):
    #     super(AlbuHandWrittenBlot, self).__init__(always_apply, p)
    #     self.hwb = hwb

    # def apply(self, image, **params):
    #     return self.hwb(image)


# def get_blot_transforms(img):
#     # h, w, c = img.shape
#     # min_h, max_h = int(0.4 * h), int(0.8 * h)
#     min_h, max_h = 50, 100
#     config = {'blot': {
#                 'params': {
#                     'min_h': min_h,
#                     'min_w': 10 * 2,
#                     'max_h': max_h,
#                     'max_w': 50 * 2,
#                     'min_shift': 10,
#                     'max_shift': 50,
#                 }
#             },}
#     bp = config['blot']['params']
#     return A.OneOf([
#         AlbuHandWrittenBlot(HandWrittenBlot(
#             {
#                 'x': (None, None),
#                 'y': (None, None),
#                 'h': (bp['min_h'], bp['max_h']),
#                 'w': (bp['min_w'], bp['max_w']),
#             }, {
#                 'incline': (bp['min_shift'], bp['max_shift']),
#                 'intensivity': (0.75, 0.75),
#                 'transparency': (0.05, 0.4),
#                 'count': i,
#             }), p=1) for i in range(1, 11)
#     ], p=0.5)

# class BlotAugmentation:
#     def __call__(self, img):
#         img = np.array(img)
#         img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

#         h, w, c = img.shape

#         rectangle_info = {
#             # Minimum and maximum X coordinate for blot position. Can be single int value.
#             'x': (1, w-1),
#             # Minimum and maximum Y coordinate for blot position. Can be single int value.
#             'y': (1, h-1),
#             # Minimum and maximum blots Height. Can be single int value.
#             'h': (int(0.1*h), int(0.4*h)),
#             # Minimum and maximum blots Width. Can be single int value.
#             'w': (int(0.1*w), int(0.2*w)),
#         }

#         blot_params = {
#             # Incline of blots. All left or right points of blot will be shifted on this value. Can be single int value.
#             'incline': (-5, 5),
#             # Points count that will be generated for blots. Can be single float value (0, 1).
#             'intensivity': (0.1, 0.4),
#             # Blots transparency. Can be single float value (0, 1).
#             'transparency': (0.05, 0.4),
#             'count': (1, 5)  # Min Max Blots count.
#         }

#         blots = HandWrittenBlot(rectangle_info, blot_params)
#         new_img = blots(img)
#         new_img = ImageOps.grayscale(Image.fromarray(new_img))

#         return new_img

# class Augmentation:
#     def __init__(self, img):
#         self.img = img
#         self.h, self.w = self.img.shape

#     # @staticmethod
#     # def apply_rotation(img):
#     #     h, w = img.shape
#     #     rot_angle = (np.random.uniform() - 0.5) * ROTATION_MIN_MAX
#     #     rot_mat = cv2.getRotationMatrix2D((h/2, w/2), rot_angle, 1)
#     #     rot_img = cv2.warpAffine(img, rot_mat, img.shape[::-1])
#     #     return rot_img

#     @staticmethod
#     def apply_rotation(img):
#         rot_angle = (np.random.uniform() - 0.5) * ROTATION_MIN_MAX
#         rot_img = rotate(img, rot_angle)
#         return rot_img

#     # @staticmethod
#     # def apply_blur(img, kernel_size=5, sigma=BLUR_SIGMA):
#     #     blured_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
#     #     return blured_img

#     @staticmethod
#     def apply_blur(img, kernel_size=BLUR_KERNEL, sigma=BLUR_SIGMA):
#         blured_img = gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
#         return blured_img


#     # @staticmethod
#     # def apply_strikethrough(img):
#     #     h, w = img.shape
#     #     point_height = np.random.randint(10, int(h/2), 1)
#     #     point_start_width = np.random.randint(0, 10, 1)
#     #     point_end_width = np.random.randint(int(w/2), w, 1)
#     #     limg = img.copy()
#     #     limg = cv2.line(
#     #         limg, (point_start_width, point_height),
#     #         (point_end_width, point_height),
#     #         color=img.max() - img.mean(),
#     #         thickness=2)

#     #     return limg

#     @staticmethod
#     def apply_strikethrough(img):
#         # print(img.size)
#         w, h = img.size
#         point_height = np.random.randint(4, int(h/2), 1)
#         point_start_width = np.random.randint(0, 10, 1)
#         point_end_width = np.random.randint(int(w/2), w, 1)

#         img1 = ImageDraw.Draw(img)
#         img1.line([(point_start_width, point_height), (point_end_width, point_height)], fill=128, width=5)

#         return img

#     # @staticmethod
#     # def apply_cutout(
#     #         img, min_patch_size_x=MIN_PATCH_X, min_patch_size_y=MIN_PATCH_Y, max_patch_size_x=MAX_PATCH_X,
#     #         max_patch_size_y=MAX_PATCH_Y):
#     #     # resized_img, orig_h, orig_w = Augmentation.resize(img, pad=False)
#     #     resized_img = copy.deepcopy(img)
#     #     h, w = resized_img.shape

#     #     patch_size_x = np.random.randint(min_patch_size_x, max_patch_size_x)
#     #     patch_size_y = np.random.randint(min_patch_size_y, max_patch_size_y)

#     #     center_x = np.random.randint(patch_size_x, h-patch_size_x)
#     #     center_y = np.random.randint(patch_size_y, w-patch_size_y)

#     #     img1 = resized_img.copy()

#     #     border_x = int(patch_size_x/2)
#     #     border_y = int(patch_size_y/2)

#     #     img1[center_x-border_x:center_x+border_x, center_y-border_y: center_y+border_y] = img1.mean()

#     #     return img1

#     @staticmethod
#     # def apply_cutout(
#     #         img, min_patch_size_x=MIN_PATCH_X, min_patch_size_y=MIN_PATCH_Y, max_patch_size_x=MAX_PATCH_X,
#     #         max_patch_size_y=MAX_PATCH_Y):
#     #     w, h = img.size

#     #     patch_size_x = np.random.randint(min_patch_size_x, max_patch_size_x)
#     #     patch_size_y = np.random.randint(min_patch_size_y, max_patch_size_y)

#     #     center_x = np.random.randint(patch_size_x, h-patch_size_x)
#     #     center_y = np.random.randint(patch_size_y, w-patch_size_y)

#     #     border_x = int(patch_size_x/2)
#     #     border_y = int(patch_size_y/2)
#     #     img.paste(int(np.asarray(img).mean()), [center_y-border_y,
#     #               center_x-border_x, center_y+border_y, center_x+border_x])

#     #     return img
#     def apply_cutout(
#                 img, patch_proportion=0.1):
#         w, h = img.size

#         patch_size_x = int(w * patch_proportion)
#         patch_size_y = int(h * patch_proportion)

#         center_x = np.random.randint(0, w-patch_size_x)
#         center_y = np.random.randint(0, h-patch_size_y)

#         img1 = copy.deepcopy(img)
#         img1.paste(int(np.asarray(img1).mean()), [center_x,
#                                                   center_y, center_x + patch_size_x, center_y + patch_size_y])

#         return img1

#     @staticmethod
#     def apply_random_drop(img):
#         # pixels with value more than mean
#         gt_threshold = np.asarray(img) <= np.asarray(img).mean()
#         gt_pos = np.where(gt_threshold)
#         drop_prob = 0.3

#         drop_pos_x = np.random.choice(gt_pos[0], int(drop_prob*gt_pos[0].shape[0]), replace=False)
#         drop_pos_y = np.random.choice(gt_pos[1], int(drop_prob*gt_pos[1].shape[0]), replace=False)

#         img_data = np.array(img)
#         img_data[drop_pos_x, drop_pos_y] = np.asarray(img).max()

#         return Image.fromarray(img_data)


fonts={
        "architects_daughther": ["fonts/ArchitectsDaughter-Regular.ttf"],
        "caveat":["fonts/Caveat-Regular.ttf"],
        "dancing_script":["fonts/DancingScript-Regular.ttf"],
        "indie_flower":["fonts/IndieFlower-Regular.ttf"],
        "lobster":["fonts/Lobster-Regular.ttf"],
        "open_sans":["fonts/OpenSans-Regular.ttf"],
        "sacramento": ["fonts/Sacramento-Regular.ttf"],
        "shadows_into_light":["fonts/ShadowsIntoLight-Regular.ttf"],
        "times_new_roman": ["fonts/times_new_roman_regular.ttf"],
        "yanone":["fonts/YanoneKaffeesatz-Regular.ttf"]

}

# import Augmentor
from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)


def create_img(text, font):
    generator = GeneratorFromStrings(
        [text,],
        blur=1,
        random_blur=False,
        fonts = [font,],
        count = 1,
        size = 64,
        width = 1024,
        background_type = 1
    )
    img, label = list(generator)[0]
    img = ImageOps.grayscale(img)

    # if np.random.random() < 0.5:
    #     p = Augmentor.Pipeline()
    #     p.gaussian_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=6, corner="dl", method="out")
    #     transforms = torchvision.transforms.Compose([p.torch_transform(), torchvision.transforms.GaussianBlur(5, sigma=(0.1, 0.1)), torchvision.transforms.RandomAdjustSharpness(2, p=1.0)])
    #     img = transforms(img.copy())

    # img = Image.new("RGB", (1024, 64), (255, 255, 255))
    # draw = ImageDraw.Draw(img)

    # width, height = img.size

    # textwidth, textheight = draw.textsize(text)

    # if font=="roboto_mono":
    #     font="yanone"

    # # 512->8, 1024-> 16
    # # font_path = f"fonts/{font}"
    # font_path = font
    # font = ImageFont.truetype(font_path, 30)

    # x = 5
    # y = (height/2) - textheight

    # draw.text((x, y), text, (0,0,0), font=font)
    img = np.array(ImageOps.grayscale(img))
    return img


# def create_img(text, font):
#     # img = Image.new("RGB", (2048, 128), (255, 255, 255))
#     img = Image.new("RGB", (1024, 64), (255, 255, 255))
#     draw = ImageDraw.Draw(img)
# 
#     width, height = img.size
#     textwidth, textheight = draw.textsize(text)
# 
#     # if font=="roboto_mono":
#     #     font="yanone"
# 
#     # 512->8, 1024-> 16
#     font_path = font
#     font = ImageFont.truetype(font_path, 16)
# 
#     x = 5
#     y = (height/2) - textheight
# 
#     draw.text((x, y), text, (0,0,0), font=font)
#     img = np.array(ImageOps.grayscale(img))
# 
#     return img
# 

class DatasetHelper:
    def __init__(self, char_model, font_model):
        self.char_model = char_model
        self.font_model = font_model
        self.codec = EncodecWrapper()

    def generate_pad_mask(self, line):
        mask = (line == self.char_model.char2index["PAD"])
        return mask

    def indexes_from_sentence(self, sentence):
        return [self.char_model.char2index[char] for char in sentence]

    def tensor_from_sentence(self, sentence, max_char_len, src):
        indexes = self.indexes_from_sentence(sentence)

        if src:
            sentence_len_diff = max_char_len - len(sentence)
        else:
            indexes.insert(0, self.char_model.char2index["TSOS"])
            indexes.append(self.char_model.char2index["TEOS"])
            sentence_len_diff = max_char_len - len(sentence) - 2

        if sentence_len_diff > 0:
            for i in range(0, sentence_len_diff):
                indexes.append(self.char_model.char2index["PAD"])

        return torch.tensor(indexes, dtype=torch.long)

    def tensor_from_font_type(self, font_type, max_char_len=None):
        font_type = str(font_type)
        return torch.tensor(self.font_model.font2index[font_type])

    @staticmethod
    def get_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def get_mask_seq_cat(first_seq_len=128, second_seq_len=128):
        second_mask = torch.triu(torch.full((second_seq_len, second_seq_len), float('-inf')), diagonal=1)
        first_mask = torch.zeros(second_seq_len, first_seq_len)

        bottom_mask = torch.cat([first_mask, second_mask], axis=-1)
        top_mask = bottom_mask[0].unsqueeze(0).repeat(first_seq_len,1)

        mask = torch.cat([top_mask, bottom_mask], axis=0)
        return mask

class TxtDataset(DatasetHelper, Dataset):
    def __init__(self, data, char_model, max_char_len):
        super().__init__(char_model, None)
        self.data = data
        self.max_char_len = max_char_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        line = row["text"]
        line = line[1:]  # Ignore the space appended during data generation

        txt = self.tensor_from_sentence(line, self.max_char_len, True)
        txt_pad_mask = self.generate_pad_mask(txt)

        txt_tgt = self.tensor_from_sentence(line, self.max_char_len, False)

        txt_tgt_in = txt_tgt[:-1]  # Decoder input during training

        # As tgt_in is created ignoring last token
        txt_tgt_in_pad_mask = self.generate_pad_mask(txt_tgt_in)
        txt_tgt_in_mask = self.get_mask(self.max_char_len-1)

        txt_tgt_out = txt_tgt[1:]  # Used during loss calculation

        dataset_instance = {
            "txt_txt": txt,
            "txt_txt_pad_mask": txt_pad_mask,
            "txt_txt_tgt_in": txt_tgt_in,
            "txt_txt_tgt_in_pad_mask": txt_tgt_in_pad_mask,
            "txt_txt_tgt_in_mask": txt_tgt_in_mask,
            "txt_txt_tgt_out": txt_tgt_out,
            "label": torch.tensor(TXT_LABEL)
        }

        return dataset_instance


class ImgDataset(DatasetHelper, Dataset):
    def __init__(self, data, font_model, max_char_len, img_dir):
        super().__init__(None, font_model)
        self.data = data
        self.max_char_len = max_char_len
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.img_dir/row["image"]
        img = 255 - cv2.imread(str(img_path), 0)
        img = img/255

        style_imgs = []
        for i in range(0,4):
            writer = row["font_type"]
            writer_row = self.data[self.data["font_type"] == writer]
            style_row = writer_row.sample(n=1).iloc[0]


            style_img_path = self.img_dir/style_row["image"]
            style_img = 255 - cv2.imread(str(style_img_path), 0)
            style_img = style_img/255
            style_imgs.append(style_img)

        style_img = np.concatenate(style_imgs, axis=0)

        font_type = row["font_type"]
        font = self.tensor_from_font_type(font_type, self.max_char_len)

        dataset_instance = {
            "img_img": torch.from_numpy(img[np.newaxis]).to(torch.float),
            "img_img_style_img": torch.from_numpy(style_img[np.newaxis]).to(torch.float),
            "img_font": font,
            "label": torch.tensor(IMG_LABEL)
        }

        return dataset_instance

def resize_img(img, h=64, w=1024, uniform_shape=True, fill=0):

    uniform_shape = uniform_shape
    IMAGE_WIDTH = w
    IMAGE_HEIGHT = h
    img_dim = (IMAGE_WIDTH, IMAGE_HEIGHT)

    if uniform_shape:
        img_w, img_h = img.size
        max_w, max_h = img_dim
        coef = 1 if img_h <= max_h and img_w <= max_w else max(img_h / max_h, img_w / max_w)
        h = int(img_h / coef)
        w = int(img_w / coef)
        img = img.resize((w, h))
        pad_h, pad_w = max_h - h, max_w - w
        if pad_h > 0:
            pad_pixels = pad_h
            up_pad = int(pad_pixels/2)
            down_pad = pad_pixels - up_pad
            padding = (0, up_pad, 0, down_pad)
            # padding = (0, 0, 0, pad_h)
            img = ImageOps.expand(img, padding, fill=fill)
        if pad_w > 0:
            padding = (0, 0, pad_w, 0)
            img = ImageOps.expand(img, padding, fill=fill)
    else:
        if img.size[0] > IMAGE_WIDTH:
            img = img.resize(img_dim)
        else:
            padding = (0, 0, IMAGE_WIDTH-img.size[0], fill)
            img = ImageOps.expand(img, padding, fill=fill)

    return img

class ImgTxtDataset(DatasetHelper, Dataset):
    def __init__(self, data, char_model, font_model, max_char_len, img_dir, augment=True):
        super().__init__(char_model, font_model)
        self.data = data
        self.max_char_len = max_char_len
        self.img_dir = img_dir
        self.augment = augment
        # self.img_mode = False
        self.img_mode = True
        self.txt_mode = True
        self.required_height = 128
        self.required_width = 2048
        self.resize = -1
        self.resize_first = True
        self.ours_augment = True
        self.stackmix_mode = False


        if self.stackmix_mode:
            stackmix_train_csv = img_dir.parent/"dataset_v2/stackmix_train.csv"
            stackmix_eval_csv = img_dir.parent/"dataset_v2/stackmix_eval.csv"
            self.stackmix_train_data = pd.read_csv(stackmix_train_csv).sample(frac=1)
            self.stackmix_eval_data = pd.read_csv(stackmix_eval_csv)
            self.stackmix_len = 3000 * 2
        print("***********************************************")
        print(f"Augmentation Status: {self.augment}")
        print(f"Ours Augment: {self.ours_augment}")
        print(f"Txt Mode: {self.txt_mode}")
        print(f"Image Mode: {self.img_mode}")
        print(f"Stackmix Mode: {self.stackmix_mode}")
        if self.img_mode:
            print(f"Resize original image: {self.resize}")
            print(f"Resize first: {self.resize_first}")
            print(f"Image Height: {self.required_height}")
            print(f"Image Width: {self.required_width}")
        print("***********************************************\n")


    def __len__(self):
        if self.stackmix_mode and self.augment:
            return len(self.data) + self.stackmix_len
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if idx < len(self.data):
            row = self.data.iloc[idx]
        elif self.augment and self.stackmix_mode and idx >= len(self.data):
            # print(f"Train Stackmix Mode: {self.stackmix_mode}")
            idx = np.random.randint(0, len(self.stackmix_train_data))
            row = self.stackmix_train_data.iloc[idx]
        else:
            raise Exception("Data out of bound")

        # """
        if self.img_mode:

            img_path = self.img_dir/row["image"]
            img = Image.open(img_path)
            img = ImageOps.grayscale(img)

            apply_aug = np.random.uniform()
            apply_aug_threshold = 0.4
            aug_threshold = 0.5
            ours_augment = self.ours_augment
            resize_first = self.resize_first

            if resize_first:
                if self.augment and ours_augment:
                    if apply_aug >= apply_aug_threshold:
                        apply_dpi_adjust = np.random.uniform() >= aug_threshold
                        if apply_dpi_adjust:
                            factor = np.random.uniform(aug["dpi"]["min_factor"], aug["dpi"]["max_factor"])
                            img = DPIAdjusting(factor)(img)
                            # print("Scaling:", factor, img.size)
                img = resize_img(img)

            resized_img = img


            # print("Before:", img.size)

            if self.augment and ours_augment:
                # print("Augment ours")
                # resized_img = apply_data_augmentation(resized_img)
                if apply_aug >= apply_aug_threshold:

                    if resize_first:
                        apply_dpi_adjust = 0
                    else:
                        apply_dpi_adjust = np.random.uniform() >= aug_threshold

                    apply_cutout = np.random.uniform() >= aug_threshold
                    apply_rot = np.random.uniform() >= aug_threshold
                    apply_blur = np.random.uniform() >= aug_threshold
                    apply_strikethrough = np.random.uniform() >= aug_threshold
                    apply_perspective = np.random.uniform() >= aug_threshold
                    apply_elastic = np.random.uniform() >= aug_threshold
                    # apply_random = np.random.uniform() >= 10
                    apply_dilation_erosion = np.random.uniform() >= aug_threshold
                    apply_contrast = np.random.uniform() >= aug_threshold
                    apply_brightness = np.random.uniform() >= aug_threshold
                    apply_random_drop = np.random.uniform() >= aug_threshold
                    apply_blot = np.random.uniform() >= 0.8


                    try:
                        if apply_dpi_adjust:
                            factor = np.random.uniform(aug["dpi"]["min_factor"], aug["dpi"]["max_factor"])
                            resized_img = DPIAdjusting(factor)(resized_img)

                        if apply_dilation_erosion:
                            kernel_h = np.random.randint(aug["dilation_erosion"]["min_kernel"],
                                                         aug["dilation_erosion"]["max_kernel"] + 1)
                            kernel_w = np.random.randint(aug["dilation_erosion"]["min_kernel"],
                                                         aug["dilation_erosion"]["max_kernel"] + 1)
                            if np.random.randint(2) == 0:
                                resized_img = Erosion((kernel_w, kernel_h),
                                                      aug["dilation_erosion"]["iterations"])(resized_img)
                            else:
                                resized_img = Dilation((kernel_w, kernel_h),
                                                       aug["dilation_erosion"]["iterations"])(resized_img)

                        if apply_random_drop:
                            resized_img = Augmentation.apply_random_drop(resized_img)

                        if apply_cutout:
                            cutout_proportion = random.uniform(0.0, 0.2)
                            resized_img = Augmentation.apply_cutout(resized_img, patch_proportion=cutout_proportion)

                        if apply_strikethrough:
                            resized_img = Augmentation.apply_strikethrough(resized_img)

                        # if apply_blot:
                        #     blot_img = np.asarray(resized_img)
                        #     blot_img = np.repeat(blot_img[:, :, np.newaxis], 3, axis=2)
                        #     blot_transform = get_blot_transforms(blot_img)
                        #     resized_img = Image.fromarray(blot_transform(image=blot_img)['image'])
                        #     resized_img = ImageOps.grayscale(resized_img)
                        if apply_blot:
                            blot_transform = BlotAugmentation()
                            resized_img = blot_transform(resized_img)

                        if apply_rot:
                            resized_img = Augmentation.apply_rotation(resized_img)

                        if apply_perspective:
                            scale = np.random.uniform(aug["perspective"]["min_factor"], aug["perspective"]["max_factor"])
                            resized_img = RandomPerspective(distortion_scale=scale, p=1, interpolation=InterpolationMode.BILINEAR, fill=255)(resized_img)

                        if apply_elastic:
                            magnitude = np.random.randint(1, aug["elastic_distortion"]["max_magnitude"] + 1)
                            kernel = np.random.randint(1, aug["elastic_distortion"]["max_kernel"] + 1)
                            magnitude_w, magnitude_h = (magnitude, 1) if np.random.randint(2) == 0 else (1, magnitude)
                            resized_img = ElasticDistortion(grid=(kernel, kernel),
                                                            magnitude=(magnitude_w, magnitude_h),
                                                            min_sep=(1, 1))(resized_img)
                        if apply_cutout:
                            cutout_proportion = random.uniform(0.0, 0.2)
                            resized_img = Augmentation.apply_cutout(resized_img, patch_proportion=cutout_proportion)

                        if apply_blur:
                            resized_img = Augmentation.apply_blur(resized_img)

                        if apply_contrast:
                            factor = np.random.uniform(aug["contrast"]["min_factor"], aug["contrast"]["max_factor"])
                            resized_img = adjust_contrast(resized_img, factor)

                        if apply_brightness:
                            factor = np.random.uniform(aug["brightness"]["min_factor"], aug["brightness"]["max_factor"])
                            resized_img = adjust_brightness(resized_img, factor)


                        # if apply_cutout >= 0.5:
                        #     resized_img = Augmentation.apply_cutout(resized_img)

                        # if apply_strikethrough >= 0.5:
                        #     resized_img = Augmentation.apply_strikethrough(resized_img)

                        # if apply_rot >= 0.5:
                        #     resized_img = Augmentation.apply_rotation(resized_img)

                        # if apply_blur >= 0.5:
                        #     resized_img = Augmentation.apply_blur(resized_img)
                    except Exception as e:
                        resized_img = img
                        print(img_path, e)

            elif self.augment and not ours_augment:
                # print("Augment VA")
                # resized_img = apply_data_augmentation(resized_img)
                pass

            # resized_img = np.asarray(resized_img)
            # resized_img = np.repeat(resized_img[:, :, np.newaxis], 3, axis=2)
            # resized_img = get_stackmix_image(resized_img, self.augment)
            # resized_img = Image.fromarray(resized_img)
            # resized_img = ImageOps.grayscale(resized_img)

            img = resized_img

            if not self.resize_first:
                img = resize_img(img)


            # print('After: ', img.size)
            img = np.array(img)

            img = img.astype(np.float32)/255.0

            # print("image shape:", img.shape)

            # img_width, img_height = img.size
            # new_height = 64 * 2
            # ratio = img_width/img_height
            # new_width = int(ratio * new_height)
            # aspect_img_dim = (new_width, new_height)
            # img = img.resize(aspect_img_dim, Image.BILINEAR)

            # if img.size[0] >= IMAGE_WIDTH or img.size[1] >= IMAGE_HEIGHT:
            #     img = img.resize(img_dim)
            # if img.size[0] < IMAGE_WIDTH:
            #     padding = (0, 0, IMAGE_WIDTH-img.size[0], 0)
            #     img = ImageOps.expand(img, padding, fill=0)
            # if img.size[1] < IMAGE_HEIGHT:
            #     padding = (0, 0, 0, IMAGE_HEIGHT-img.size[1])
            #     img = ImageOps.expand(img, padding, fill=0)

            # img = np.array(img)

            # if np.random.rand() < 0.2 and self.augment: 
            #     img = 255 - img
            # img = np.array(img)
            # img = 255 - img
            # img = img/255

            # """

           
            # img = 255 - create_img(row["text"], row["font_type"])

            """

            style_imgs = []
            for i in range(0,4):

                writer = row["font_type"]
                writer_row = self.data[self.data["font_type"] == writer]
                style_row = writer_row.sample(n=1).iloc[0]

                # style_img_path = self.img_dir/style_row["style"]
                # style_img = 255 - cv2.imread(str(style_img_path), 0)

                style_img = 255 - create_img(style_row["text"], style_row["font_type"])

                style_img = style_img/255
                style_imgs.append(style_img)

            style_img = np.concatenate(style_imgs, axis=0)
            """
            # if self.augment:
            #     required_width =  2 * txt_tgt_len + 1
            #     if required_width > img.size[0]:
            #         padding = (0, 0, required_width - img.size[0], 0)
            #         img = ImageOps.expand(img, padding, fill=0)

            # Starts here

            # required_width = self.required_width #1096 #1136 - 5*8 #1096 #1432 # 1096
            # required_height = self.required_height #160 #192 - 3 * 32 #128 #256 # 160 - 32 #192
            # 
            # # required_width = 2568
            # # required_height = 64
            # pad = True
            # if required_width < img.size[0]:
            #     # print(row["image"], img.size)
            #     pad = False
            # if required_height < img.size[1]:
            #     # print(row["image"], img.size)
            #     pad = False

            # if pad:
            #     padding = (0, 0, required_width - img.size[0], required_height - img.size[1])
            #     # img = ImageOps.expand(img, padding, fill=255)
            #     img = ImageOps.expand(img, padding, fill=0)
            # img = np.array(img)

            #Ends here

            # img = img/255
            # img = 255 - img

            # For tanh normalization
            # https://stackoverflow.com/questions/58954799/how-to-normalize-pil-image-between-1-and-1-in-pytorch-for-transforms-compose
            # img = (img-0.5)/0.5

        # font_type = row["font_type"]
        # font = self.tensor_from_font_type(font_type, self.max_char_len)

        if self.txt_mode:
            line = row["text"]
            line = html.unescape(line)
            # line = line[1:]  # Ignore the space appended during data generation

            txt = line
            # txt = self.tensor_from_sentence(line, self.max_char_len, True)
            # txt_pad_mask = self.generate_pad_mask(txt)

            # txt_tgt = self.tensor_from_sentence(line, self.max_char_len, False)
            txt_tgt_len = torch.tensor(len(line)+2)


        # txt_tgt_in = txt_tgt[:-1]  # Decoder input during training

        # # As tgt_in is created ignoring last token
        # txt_tgt_in_pad_mask = self.generate_pad_mask(txt_tgt_in)
        # first_sequence_pad_mask = torch.zeros(130, dtype=torch.bool)
        # txt_tgt_in_pad_mask = torch.cat([first_sequence_pad_mask, txt_tgt_in_pad_mask])
    
        # txt_tgt_in_mask = self.get_mask(self.max_char_len-1)

        # txt_tgt_out = txt_tgt[1:]  # Used during loss calculation


        # img = torch.from_numpy(img[np.newaxis]).to(torch.float)
        # """
        # img = img.unsqueeze(0)
        # img = f.unfold(img, kernel_size=(32,8), stride=8)
        # img = img.permute(0, 2, 1)
        # img = img.squeeze(0)
        # """
        # src_mask = self.get_mask_seq_cat(130, self.max_char_len-1)

        """

        dataset_instance = {
            "img_txt_img": img,
            "img_txt_style_img": torch.from_numpy(style_img[np.newaxis]).to(torch.float),
            "img_txt_font": font,
            "img_txt_txt": txt,
            "img_txt_txt_pad_mask": txt_pad_mask,
            "img_txt_txt_tgt_in": txt_tgt_in,
            "img_txt_txt_tgt_in_pad_mask": txt_tgt_in_pad_mask,
            "img_txt_txt_tgt_in_mask": txt_tgt_in_mask,
            "img_txt_txt_tgt_out": txt_tgt_out,
            "src_mask": src_mask,
            "label": torch.tensor(IMG_TXT_LABEL)
        }
        """
        if self.img_mode and self.txt_mode:
            dataset_instance = {
                "char_model": self.char_model,
                "img_txt_img": img,
                "img_txt_original_txt": txt,
                # "img_txt_txt": txt,
                # "img_txt_txt_pad_mask": txt_pad_mask,
                # "img_txt_txt_tgt": txt_tgt,
                "img_txt_txt_tgt_len": txt_tgt_len,
                # "img_txt_txt_tgt_in": txt_tgt_in,
                # "img_txt_txt_tgt_in_pad_mask": txt_tgt_in_pad_mask,
                # "img_txt_txt_tgt_in_mask": txt_tgt_in_mask,
                # "img_txt_txt_tgt_out": txt_tgt_out,
                # "src_mask": src_mask,
                "label": torch.tensor(IMG_TXT_LABEL)
            }
        elif not self.img_mode and self.txt_mode:
            dataset_instance = {
                "char_model": self.char_model,
                "img_txt_original_txt": txt,
                "img_txt_txt_tgt_len": txt_tgt_len,
                "label": torch.tensor(TXT_LABEL)
            }

        return dataset_instance

class StyleDataset(DatasetHelper, Dataset):
    def __init__(self, data, char_model, font_model, max_char_len, img_dir, augment=True):
        super().__init__(char_model, font_model)
        self.data = data
        # print(self.data.head())
        self.max_char_len = max_char_len
        self.img_dir = img_dir
        self.char_model = char_model
        self.font_model = font_model

    def __len__(self):
        return len(self.data)
        # return (self.font_model.n_fonts)

    def __getitem__(self, idx):
        writers = list(set(self.data["font_type"].values.tolist()))
        # print("Writers:", writers)
        # print(len(writers))

        row = self.data.iloc[idx]
        # if idx not in writers:
        #     row = self.data.sample(n=1).iloc[0]
        # else:
        #     current_writer_df = self.data[self.data["font_type"] == idx]
        #     row = current_writer_df.sample(n=1).iloc[0]

        current_writer = row["font_type"]
        # print("Current writer:", current_writer)
        writers.remove(current_writer)
        # print("Writers Now:", writers)
        # print(len(writers))

        style_writer = random.choice(writers)
        # print("Style writer:", style_writer)
        style_writer_df = self.data[self.data["font_type"]==style_writer]
        # print("DF:", style_writer_df)
        style_writer_id = self.tensor_from_font_type(style_writer)
        # print("Style writer id:", style_writer_id)


        # print("Writer ID:", idx)

        img_path = self.img_dir/row["image"]
        img = Image.open(img_path)
        img = ImageOps.grayscale(img)
        img = resize_img(img, fill=255)
        img = np.array(img)/255

        style_img_list = []
        for i in range(2):
            style_row = style_writer_df.sample(n=1).iloc[0]
            style_img_path = self.img_dir/style_row["image"]
            # print('Style img path', style_img_path)
            style_img = Image.open(style_img_path)
            style_img = ImageOps.grayscale(style_img)
            style_img = resize_img(style_img, fill=255)
            style_img = np.array(style_img)/255
            style_img_list.append(style_img)

        line = row["text"]
        line = html.unescape(line)
        txt = line
        txt_tgt_len = torch.tensor(len(line)+2)

        dataset_instance = {
            "char_model": self.char_model,
            "img_txt_img": img,
            "img_txt_style_img": style_img_list[0],
            "img_txt_style_img_collection": np.concatenate(style_img_list, axis=0),
            "writer_id": style_writer_id,
            "img_txt_original_txt": txt,
            "img_txt_txt_tgt_len": txt_tgt_len,
            "label": torch.tensor(IMG_TXT_LABEL)
        }

        return dataset_instance

class SyntheticStyleDataset(DatasetHelper, Dataset):
    def __init__(self, data, char_model, font_model, max_char_len, img_dir, augment=True):
        super().__init__(char_model, font_model)
        self.fonts = list(glob.glob("dataset_vqvae/fonts/*.ttf"))
        print(np.random.choice(self.fonts))
        self.data = data


        self.max_char_len = max_char_len
        self.img_dir = img_dir
        self.char_model = char_model
        self.font_model = font_model

    def __len__(self):
        # return 20000
        return len(self.data)
        # return (self.font_model.n_fonts)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        # while len(text) > 64:
        #     new_idx = np.random.randint(len(self.data))
        #     row = self.data.iloc[idx]
        #     text = row["text"]

        # font_type = np.random.choice(self.fonts)
        font_type = self.fonts[5]


        style_row = self.data.sample(n=1).iloc[0]
        style_text = style_row["text"]
        # while len(style_text) > 64:
        #     style_row = self.data.sample(n=1).iloc[0]
        #     style_text = style_row["text"]
        # style_text = text

        line = style_text + " " + text
        # line = html.unescape(line)
        txt = line
        txt_tgt_len = torch.tensor(len(line)+2)

        img = 255 - create_img(txt, font_type)
        img = img/255

        style_img = 255 - create_img(style_text, font_type)
        style_img = style_img / 255

        dataset_instance = {
            "char_model": self.char_model,
            "img_txt_img": img,
            "img_txt_style_img": style_img,
            "img_txt_original_txt": txt,
            "img_txt_txt_tgt_len": txt_tgt_len,
            "label": torch.tensor(IMG_TXT_LABEL)
        }

        return dataset_instance

class LibriSpeechDataset(DatasetHelper, Dataset):
    #  data, char_model, font_model, max_char_len, img_dir, augment=True
    def __init__(self, data, char_model, font_model, max_char_len, img_dir, augment=True):
        super().__init__(char_model, font_model)

        self.audio_dir = "/home/wiseyak/suraj/everything_text_valle/Valle/audio_dataset/LibriSpeech"
        self.data = data
        self.max_aud_len = None
        self.char_model = char_model
        self.font_model = font_model
        self.max_char_len = max_char_len

    def __len__(self):
        # print('*'*100)
        # print(self.data)
        return len(self.data)

    def __getitem__(self, idx):
        # Get the audio file path and load the audio
        row = self.data.iloc[idx]
        txt = row["text"]

        # text + audio 

        txt_tgt_len = torch.tensor(len(txt)+2)

        # style text + text + eos + style hand image + prediction + eos
        # text + image 100 tokens + pred remaining token

        audio_file = os.path.join(self.audio_dir, row['audio'])
        try:
            img = torch.load(audio_file).squeeze(0)
        except:
            print('Got error in audio: ', audio_file)
            # audio_file = audio_file.split('encodec_Libri/')[-1]
            # audio_file = audio_file.replace('encodec_Libri/L', 'L').replace('.pt','.flac')
            waveform, sample_rate = torchaudio.load(audio_file)
            _, indices, _ = self.codec(waveform, return_encoded = True)
            img = indices

        #For Ar, we only use first codebook
        img = img[:,:1].squeeze(1)

        '''
        audio_length = img.shape[1] / sample_rate
        if audio_length > self.max_aud_len:
            return None
        '''

        # Return a dictionary containing the waveform, sample rate, and transcript
        # return {'waveform': waveform, 'sample_rate': sample_rate, 'transcript': transcript}
        dataset_instance = {
            "char_model": self.char_model,
            "audio_path": audio_file,
            "img_txt_img": img,
            "img_txt_style_img": img,
            "img_txt_original_txt": txt,
            "img_txt_txt_tgt_len": txt_tgt_len,
            "label": torch.tensor(IMG_TXT_LABEL)
        }

        return dataset_instance

def collate_fn(batch):
    return BatchOutput(batch)
