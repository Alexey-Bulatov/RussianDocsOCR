from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np

class DocType(BaseModule):

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):

        self.model_name = 'DocType'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:

        self.load_img(img)

        doc_type, dist, thresh = self.model.predict(img)
        conf = np.round(1 - dist/thresh,2)
        meta = {
            self.model_name:
                {
                    'doc_type': doc_type,
                    'confidence': conf
                }
        }
        return meta

