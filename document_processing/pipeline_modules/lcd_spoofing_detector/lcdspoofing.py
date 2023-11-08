from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2


class LCDSpoofing(BaseModule):
    """
    Detects spoofing from displays
    0 - a fake or electronic version
    1 - a real one
    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose=False):
        self.model_name = 'LCDSpoofing'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        self.load_img(img)
        result, conf = self.model.predict(img)
        meta = {
            self.model_name: (result, conf)
        }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:

        meta = {}

        return meta






