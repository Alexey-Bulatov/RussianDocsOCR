from pathlib import Path
from .preprocessing import BasePreprocessing, ClassificationPreprocessing, YoloPreprocessing, OCRPreprocessing
from .postprocessing import BasePostprocessing, MetricPostprocessing, OCRPostprocessing, \
    YoloDetectorPostprocessing, YoloSegmentorPostprocessing, MultiClassPostprocessing, BinaryClassPostprocessing
from .inference import ModelInference
import json
from typing import Union, List
import numpy as np

class ModelLoader:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, json_file: Path, device='gpu'):
        self.json_file = json.loads(json_file.read_text(encoding="utf8"))
        self.working_dir = json_file.parent
        self.device = device
        if self.json_file['Type'] == 'Metric':
            model = self.__load_metric_model()
        elif self.json_file['Type'] == 'YoloDetector':
            model = self.__load_yolo_detector()
        elif self.json_file['Type'] == 'YoloSegmentor':
            model = self.__load_yolo_segmentor()
        elif self.json_file['Type'] == 'BinaryClassification':
            model = self.__load_binary_classificator()
        elif self.json_file['Type'] == 'MultiLabelClassification':
            model = self.__load_multi_label_classificator()
        elif self.json_file['Type'] == 'OCR':
            model = self.__load_ocr(self.json_file['Lang'])
        else:
            raise Exception(f"[!] Not supported model type: {self.json_file['Type']}")


        return model


    def __load_metric_model(self):
        '''
        Loading metric processing
        :return:
        '''

        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                ClassificationPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose,
                )
            )

        model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
                                         device=self.device,
                                         verbose=self.verbose)

        postprocessings = []
        for outp_postprocess in self.json_file['Output']:
            postprocessings.append(
                MetricPostprocessing(
                    self.working_dir.joinpath(self.json_file['Centers']),
                    metric=outp_postprocess['Metric'],
                    verbose=self.verbose
                )
            )

        model = ClassificationModel(
            model_type= self.json_file['Type'],
            preprocessing=preprocessings[0], #TODO multi input
            model_inference=model_inference,
            postprocessing=postprocessings[0], #TODO Multi output
        )

        return model

    def __load_binary_classificator(self):
        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                ClassificationPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose
                )
            )

        model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
                                         device=self.device,
                                         verbose=self.verbose)

        postprocessings = []
        for _ in self.json_file['Output']:
            postprocessings.append(
                BinaryClassPostprocessing(
                    self.json_file['Labels'],
                    verbose=self.verbose
                )
            )

        model = ClassificationModel(
            model_type=self.json_file['Type'],
            preprocessing=preprocessings[0],  # TODO multi input
            model_inference=model_inference,
            postprocessing=postprocessings[0],  # TODO Multi output
        )

        return model

    def __load_ocr(self, lang):
        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                OCRPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose
                )
            )

        model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
                                         device=self.device,
                                         verbose=self.verbose)

        model = OCRModel(
            model_type=self.json_file['Type'],
            preprocessing=preprocessings[0],
            model_inference=model_inference,
            postprocessing=OCRPostprocessing(lang=lang, verbose=False),
        )

        return model

    def __load_multi_label_classificator(self):
        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                ClassificationPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose
                )
            )

        model_inference = ModelInference(
            self.working_dir.joinpath(self.json_file['File']),
            device=self.device,
            verbose=self.verbose
        )

        postprocessings = []
        for _ in self.json_file['Output']:
            postprocessings.append(
                MultiClassPostprocessing(
                    self.json_file['Labels'],
                    verbose=self.verbose
                )
            )

        model = ClassificationModel(
            model_type=self.json_file['Type'],
            preprocessing=preprocessings[0],  # TODO multi input
            model_inference=model_inference,
            postprocessing=postprocessings[0],  # TODO Multi output
        )

        return model

    def __load_yolo_detector(self):
        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                YoloPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose
                )
            )
        model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
                                         device=self.device,
                                         verbose=self.verbose)

        postprocessings = []
        for _ in self.json_file['Output']:
            postprocessings.append(
                YoloDetectorPostprocessing(
                    iou=self.json_file['IOU'],
                    cls=self.json_file['CLS'],
                    labels=self.json_file['Labels'],
                    verbose=self.verbose
                )
            )

        model = YoloDetectorModel(
            model_type=self.json_file['Type'],
            preprocessing=preprocessings[0],  # TODO multi input
            model_inference=model_inference,
            postprocessing=postprocessings[0],  # TODO Multi output
        )

        return model

    def __load_yolo_segmentor(self):
        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                YoloPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose,
                )
            )
        model_inference = ModelInference(
            self.working_dir.joinpath(self.json_file['File']),
            device=self.device,
            verbose=self.verbose
        )

        postprocessings = [
            YoloDetectorPostprocessing(
                iou=self.json_file['IOU'],
                cls=self.json_file['CLS'],
                labels=self.json_file['Labels'],
                verbose=self.verbose
            ),
            YoloSegmentorPostprocessing(self.json_file['MaskFilter'],verbose=self.verbose),
        ]

        model = YoloSegmentorModel(
            model_type=self.json_file['Type'],
            preprocessing=preprocessings[0],  # TODO multi input
            model_inference=model_inference,
            postprocessing=postprocessings,
        )

        return model

class Model:
    def __init__(self,
                 model_type: str,
                 preprocessing: BasePreprocessing,
                 model_inference: ModelInference,
                 postprocessing: Union[List[BasePostprocessing],BasePostprocessing]):

        self.__model_type = model_type
        self.preprocessing = preprocessing
        self.inference_model = model_inference
        self.postprocessing = postprocessing


    def predict(self, img: Union[Path, np.ndarray]):
        '''
        Method that utilizes preprocessing, inference, and postprocessing
        :param img: Accepts path to img as Path or np.ndarray
        :return: returns tuple as result
        '''
        pass

    def predict_fv(self, img: Union[Path, np.ndarray]):
        '''
        Method that utilizes preprocessing, inference
        :param img: Accepts path to img as Path or np.ndarray
        :return: returns result from model without postprocessing
        '''
        pass

    @property
    def model_type(self):
        '''
        :return: model type
        '''
        return self.__model_type

class ClassificationModel(Model):
    '''
    Class implementing the classification task.
    Has 2 methods, predict and predict_fv.
    predict returns result with postprocessing
    predict_fv return result without calling postprocessing
    '''
    def __init__(self, model_type:str,  preprocessing: BasePreprocessing, model_inference: ModelInference,
                 postprocessing: BasePostprocessing):
        super().__init__(model_type, preprocessing, model_inference, postprocessing)

    def predict(self, img: Union[Path, np.ndarray]):
        tensor = self.preprocessing(img)
        inf_result = self.inference_model.predict(tensor)[0]
        result = self.postprocessing(inf_result)
        return result

    def predict_fv(self, img: Union[Path, np.ndarray]):
        tensor = self.preprocessing(img)
        inf_result = self.inference_model.predict(tensor)[0]
        return inf_result


class OCRModel(Model):
    def __init__(self, model_type:str,  preprocessing: OCRPreprocessing, model_inference: ModelInference,
                 postprocessing: OCRPostprocessing):
        super().__init__(model_type, preprocessing, model_inference, postprocessing)

    def predict(self, img: Union[Path, np.ndarray]):
        tensor = self.preprocessing(img)
        inf_result = self.inference_model.predict(np.expand_dims(np.expand_dims(tensor, -1), 0))[0]
        result = self.postprocessing(inf_result)
        return result


class YoloDetectorModel(Model):
    '''
    Class implementing detection task using YOLO net.
    Has 2 methods, predict and predict_fv.
    predict returns result with postprocessing
    predict_fv return result without calling postprocessing
    '''

    def __init__(self, model_type:str, preprocessing: BasePreprocessing, model_inference: ModelInference,
                 postprocessing: BasePostprocessing):
        super().__init__(model_type, preprocessing, model_inference, postprocessing)

    def predict(self, img: Union[Path, np.ndarray]):
        tensor, pad_ratio, pad_extra, pad_to_size, _  = self.preprocessing(img)

        inf_result = self.inference_model.predict(tensor)

        padding_meta = {
            'pad_to_size': pad_to_size,
            'pad_extra': pad_extra,
            'ratio': pad_ratio,
        }

        bboxes = np.squeeze(inf_result)

        result = self.postprocessing(bboxes, padding_meta=padding_meta, resize=True)
        return result

    def predict_fv(self, img: Union[Path, np.ndarray]):
        tensor, pad_ratio, pad_add_extra, pad_add_to_size, _ = self.preprocessing(img)
        inf_result = self.inference_model.predict(tensor)
        bboxes = np.squeeze(inf_result)
        return bboxes


class YoloSegmentorModel(Model):
    '''
    Class implementing segmentation task for YOLO detection
    Has 2 methods, predict and predict_fv.
    predict returns result with postprocessing - bboxes, masks, segments
    predict_fv return result without calling postprocessing. bbox and masks without any postprocessing (nms and etc.)
    '''
    def __init__(self, model_type: str, preprocessing: BasePreprocessing, model_inference: ModelInference,
                 postprocessing: List[BasePostprocessing]):
        super().__init__(model_type, preprocessing, model_inference, postprocessing)

    def predict(self, img: Union[Path, np.ndarray]):


        tensor, pad_ratio, pad_extra, pad_to_size, img_shape  = self.preprocessing(img)
        # print(pad_extra)
        inf_result = self.inference_model.predict(tensor)


        padding_meta = {
            'pad_to_size': pad_to_size,
            'pad_extra': pad_extra,
            'ratio': pad_ratio,
        }

        bboxes, masks = np.squeeze(inf_result[0]), np.squeeze(inf_result[1])

        nms_prediction = self.postprocessing[0](bboxes[:], padding_meta=padding_meta, resize=True, numpy=True)
        if len(nms_prediction) == 0:
            return None, None, None

        masks, segments = self.postprocessing[1](masks, nms_prediction[:, 6:], nms_prediction[:, :4], pad_extra, img_shape, upsample=True)
        return nms_prediction[:, :6], masks, segments

    def predict_fv(self, img: Union[Path, np.ndarray]):
        tensor, pad_ratio, pad_add_extra, pad_add_to_size = self.preprocessing(img)
        inf_result = self.inference_model.predict(tensor)

        bboxes, masks = (np.squeeze(inf_result[0]), np.squeeze(inf_result[1])) \
            if isinstance(inf_result, list) else (np.squeeze(inf_result), None)

        # print(bboxes)

        return bboxes, masks