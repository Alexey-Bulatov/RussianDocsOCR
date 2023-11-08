from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2

class Angle90(BaseModule):

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        self.model_name = 'Angle90'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:

        self.load_img(img)

        angle, conf = self.model.predict(img)
        meta = {
            self.model_name: {
                'angle': angle,
                'confidence': conf,
            }
        }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:

        img = self.load_img(img)
        angle, conf = self.model.predict(img)
        for _ in range(angle // 90):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        meta = {
            self.model_name: {
                'angle' : angle,
                'confidence': conf,
                'warped_img': img,
            }
        }

        return meta





