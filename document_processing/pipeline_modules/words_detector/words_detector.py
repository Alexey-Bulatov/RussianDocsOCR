from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np

class WordsDetector(BaseModule):

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):

        self.model_name = 'WordsDetector'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:

        self.load_img(img)

        bbox = self.model.predict(img)
        meta = {
            self.model_name:
                {
                    'bbox': bbox,
                }
        }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:

        img = self.load_img(img)
        bbox = self.model.predict(img)

        img_patches = []


        #resort left -> right x coord
        bbox.sort(key=lambda x: x[0])

        for box in bbox:
            img_patches.append(img[box[1]:box[3], box[0]: box[2]])

        meta = {
            self.model_name:
                {
                    'bbox': bbox,
                    'warped_img': img_patches,
                }
        }

        return meta
