from ..base_module import BaseModule
from .image_transformation import fix_perspective
from typing import Union
from pathlib import Path
import numpy as np



class DocDetector(BaseModule):

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):

        self.model_name = 'DocDetector'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:

        self.load_img(img)

        bbox, mask, segm = self.model.predict(img)
        meta = {
            self.model_name:
                {
                    'bbox': bbox,
                    'mask': mask,
                    'segm': segm,
                }
        }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:

        img = self.load_img(img)
        bbox, mask, segm = self.model.predict(img)

        if segm:

            #If more then 2 segments, we get only 2 best
            if len(bbox) > 2:
                best_bboxes = np.sort(np.argsort(np.array(bbox[..., -2]))[-2:])
                bbox = np.array(bbox)[best_bboxes].tolist()
                mask = np.array(mask)[best_bboxes].tolist()
                segm = [seg for i, seg in enumerate(segm) if i in best_bboxes]


            try:
                result_img, borders_img = fix_perspective(img=img, segments=segm)
            except Exception as e:
                print('[!] Failed to fix perspective')
                result_img = borders_img = img
        else:
            result_img = borders_img = img
        meta = {
            self.model_name:
                {
                    'bbox': bbox,
                    'mask': mask,
                    'segm': segm,
                    'border_img': borders_img,
                    'warped_img': result_img,

                }
        }

        return meta
