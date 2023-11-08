from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2


class PrintSpoofing(BaseModule):
    """
    Detects reprint spoofing and copies
    0 - a fake or copy
    1 - a real one
    I set the threshold to 0.9. The real document must have the confidence more than 0.9
    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose=False):
        self.model_name = 'PrintSpoofing'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        self.load_img(img)
        result, conf = self.model.predict(img)
        if conf < 0.9:
            meta = {
                self.model_name: ('FAKE', conf)
            }
        else:
            meta = {
                self.model_name: (result, conf)
            }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:

        meta = {}

        return meta






