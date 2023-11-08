import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple


class BasePreprocessing(object):
    '''
    Base class for preprocessing from which inherit other preprocessing classes
    '''
    def __init__(self,
                 image_size=(224, 224, 3),
                 normalization=(0, 1),
                 padding_size=(0,0),
                 padding_color = (114,114,114),
                 verbose=False):
        self.image_size = image_size
        self.normalization = normalization
        self.padding_size = padding_size
        self.padding_color = padding_color

        if verbose:
            print(f'[+] {self.__class__.__name__} loaded')


    def __call__(self, image_path: Union[Path, str, np.ndarray]):
        if isinstance(image_path, Path) or isinstance(image_path, str):
            image = cv2.imread(image_path.as_posix() if isinstance(image_path, Path) else image_path)
            if image is None:
                raise Exception(f"Not an image {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_path, np.ndarray):
            image = image_path
        else:
            raise Exception("Unsupported Image Type")

        return image
    def padding(self, img: np.array):
        pad_h, pad_v = self.padding_size
        img = cv2.copyMakeBorder(img, pad_v//2, pad_v//2 ,pad_h//2, pad_h//2, cv2.BORDER_CONSTANT, value=self.padding_color)
        return img, (pad_h//2, pad_v//2)

    def normalization(self, img: np.array, normalization: tuple):
        mean, stdev = normalization
        img = img / stdev - mean
        return img


class ClassificationPreprocessing(BasePreprocessing):
    '''
    Preprocessing for classification
    '''
    def __init__(self,
                 image_size=(224, 224, 3),
                 normalization=(0,1),
                 padding_size=(0,0),
                 padding_color=(0,0,0),
                 verbose=False):
        super().__init__(image_size=image_size,
                         normalization=normalization,
                         padding_size=padding_size,
                         padding_color=padding_color,
                         verbose=verbose)


        # print(f'[+] {self.__class__.__name__} loaded')

    def __call__(self, image_path: Union[Path, str, np.ndarray]):
        image = super().__call__(image_path)

        image, _ = self.padding(image)

        image = cv2.resize(image, self.image_size[:2])
        if len(image.shape) == 3:
            image = np.expand_dims(image,0)

        return image




class YoloPreprocessing(BasePreprocessing):
    '''
    Preprocessing for YOLO
    '''
    def __init__(self,
                 image_size=(640, 640, 3),
                 normalization = (0,1),
                 padding_size=(0,0),
                 padding_color=(114,114,114),
                 verbose=False
                 ):
        super().__init__(image_size=image_size,
                         normalization=normalization,
                         padding_size=padding_size,
                         padding_color=padding_color,
                         verbose=verbose)
        # print(f'[+] {self.__class__.__name__} loaded')


    def __call__(self, image_path: Union[Path, str, np.ndarray]):

        image = super().__call__(image_path)

        # YOLO works with BGR color format, so need swap
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image, pad_add_extra = self.padding(image) #extra_padding is padding surrounding image

        padded_image_shape = image.shape[:2] # shape of image with extra padding




        #pad_add_to_size - padding added after resize to fill till size we need
        image, pad_ratio, pad_add_to_size = self.__letterbox(image,
                                                          new_shape=self.image_size,
                                                          color=self.padding_color,
                                                          auto=False,
                                                          scaleFill=False,
                                                          scaleup=True,
                                                          stride=32)



        if len(image.shape) == 3:
            image = np.expand_dims(image,0)

        return image, pad_ratio, pad_add_extra, pad_add_to_size, padded_image_shape


    def __letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        '''
        Function used for letterbox resize. It resize and uses padding to fill picture to new_shape size
        :param im: image
        :param new_shape: shape of new image
        :param color: color for padding fill
        :param auto: Make minimum rectangle as possible
        :param scaleFill: should we use stretch
        :param scaleup: IF needed should we use scaleup
        :param stride: size, of squares. Used for Yolo5, default 32
        :return:  new_image, ratio which been used to reduce size, padding size
        '''
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)


class OCRPreprocessing(BasePreprocessing):
    def __init__(self,
                 image_size=(31, 200, 1),
                 normalization=(0,1),
                 padding_size=(0,0),
                 padding_color=(0,0,0),
                 verbose=False):
        super().__init__(image_size=image_size,
                         normalization=normalization,
                         padding_size=padding_size,
                         padding_color=padding_color,
                         verbose=verbose)

    def __call__(self, image_path: np.ndarray):
        image = super().__call__(image_path)

        image = self.padding(image)

        return image

    @staticmethod
    def recalc_image(original_shape: Tuple[int, int]) -> Tuple[int, int]:
        target_h, target_w = [31, 200]
        orig_h, orig_w = original_shape
        new_h = target_h
        ratio = new_h / float(orig_h)
        new_w = int(orig_w * ratio)
        # для длинных лоскутов подгоняем высоту
        if new_w > target_w:
            new_w = target_w
            r = new_w / float(orig_w)
            new_h = int(orig_h * r)
        return new_h, new_w

    def padding(self, image: np.ndarray) -> np.ndarray:
        target_shape = [31, 200]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(image, self.recalc_image(image.shape)[::-1])
        orig_h, orig_w = resized.shape
        target_h, target_w = target_shape

        color_value = int(image[-1][-1])
        x_offset = 0
        y_offset = 0

        padded = cv2.copyMakeBorder(resized, y_offset, target_h - orig_h - y_offset, x_offset,
                                    target_w - orig_w - x_offset,
                                    borderType=0, value=[color_value, color_value])

        return padded
