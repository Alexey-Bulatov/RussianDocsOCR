from typing import Union
from pathlib import Path
import numpy as np
from ..config import DEFAULT_CFG
from ..processing.models import ModelLoader
import cv2
import json

class BaseModule:

    def __init__(self, model_name: str, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        '''
        Loads model from json
        :param model_name: name of model to load
        :param model_format: which format of model to load
        :param device: load on cpu or gpu
        '''
        if model_name in DEFAULT_CFG.keys():
            self.__model_path = Path(DEFAULT_CFG.get(model_name)).joinpath(model_format, 'model.json')
        else:
            raise Exception("No path for this type of model detected in models_path.yaml")

        self.__model_info = json.loads(self.__model_path.read_bytes())
        print(f'[*] Loading model {model_name}!')
        self.model = ModelLoader(verbose=verbose)(self.__model_path, device=device)
        # print('[*] Model generated!\n')


    @property
    def model_info(self) -> dict:
        '''
        Returns info about model in dict format
        :return: dict
        '''
        return self.__model_info



    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        '''
        Just passes img to net and returns result from net without any transformations
        :param img: Can be Path type, str type or np.ndarray
        :return: meta in dict format
        '''
        pass


    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        '''
        Sends img to net and applies transformation function according to net result
        :param img: Can be Path type, str type or np.ndarray
        :return: modified img, meta in dict format
        '''
        pass

    @staticmethod
    def load_img(img_path: Union[str, Path, np.ndarray]):
        '''
        Method that loads image and converts it to RGB color mode
        '''
        if isinstance(img_path, Path):
            img = cv2.imread(img_path.as_posix())
            img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
        elif isinstance(img_path, str):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_path, np.ndarray):
            img = img_path
        else:
            raise Exception("Unsupported input type as img")
        return img