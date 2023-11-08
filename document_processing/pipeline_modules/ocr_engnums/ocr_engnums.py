from pathlib import Path
from typing import Union
from datetime import datetime

import numpy as np

from ..base_module import BaseModule


class OCREngNums(BaseModule):
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        self.model_name = 'OCREngNums'
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
        if field_type in ['Issue_date',
                          'Expiration_date',
                          'Birth_date']:
            try:
                return self.check_ddmmyyyy(text)
            except ValueError:
                return text
        elif field_type in ['Sex_en']:
            return self.check_en_sex(text)
        elif field_type in ['Driver_class']:
            return self.check_driver_class(text)
        else:
            return text

    @staticmethod
    def check_driver_class(driver_class: str) -> str:
        driver_class = driver_class.replace(' ', '')
        allowed_letters = ['A', 'B', 'C', 'D', 'E', 'M', '1']
        new_driver_class = ''
        for letter in driver_class:
            if letter in allowed_letters:
                new_driver_class += letter

        return new_driver_class


    @staticmethod
    def check_ddmmyyyy(date: str) -> str:
        date = date.replace('O', '0').replace('-', '.')
        pure_nums = ''.join(c for c in date if c.isnumeric())
        if len(pure_nums) == 8:
            ret = datetime.strptime(pure_nums, '%d%m%Y').strftime('%d.%m.%Y')
            return ret
        return date

    @staticmethod
    def check_en_sex(sex: str) -> str:
        strip = sex.lstrip('.').upper()
        to_check = strip.replace('.', '')
        # if len(to_check) >= 3:
        #     result = 'МУЖ' if 'М' in to_check else 'ЖЕН'
        #     if '.' in strip:
        #         result = result + '.'
        # else:
        result = 'M' if 'M' in to_check else 'F'
        # if '.' in strip:
        #     result = result + '.'
        return result