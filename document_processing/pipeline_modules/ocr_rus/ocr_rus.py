from pathlib import Path
from typing import Union

import numpy as np

from ..base_module import BaseModule


class OCRRus(BaseModule):
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        self.model_name = 'OCRRus'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        self.load_img(img)

        ocr_output = self.model.predict(img)
        meta = {
            self.model_name: {
                'ocr_output': ocr_output
            }
        }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        pass

    def fix_errors(self, field_type: str, text: str) -> str:
        if field_type in ['Last_name_ru',
                          'First_name_ru',
                          'Birth_place_ru',
                          'Living_region_ru',
                          'Middle_name_ru']:
            return self.check_russian_names(text)
        elif field_type in ['Sex_ru']:
            return self.check_rus_sex(text)
        else:
            return text

    @staticmethod
    def check_russian_names(name: str) -> str:
        return name.lstrip('.')

    @staticmethod
    def check_rus_sex(sex: str) -> str:
        strip = sex.lstrip('.').upper()
        to_check = strip.replace('.', '')
        # if len(to_check) >= 3:
        #     result = 'МУЖ' if 'М' in to_check else 'ЖЕН'
        #     if '.' in strip:
        #         result = result + '.'
        # else:
        result = 'М' if 'М' in to_check else 'Ж'
            # if '.' in strip:
            #     result = result + '.'
        return result
