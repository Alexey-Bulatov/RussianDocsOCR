import json
from pathlib import Path
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import subprocess
import tf2onnx
import onnx
from onnxsim import simplify
import coremltools as ct
from typing import Union


class Converter:

    def __init__(self, model_path:Union[Path, tf.keras.models.Model], model_info:dict, path_to_save:Path,):
        if isinstance(model_path, tf.keras.models.Model):
            self.model = model_path
        elif model_path.suffix == '.h5':
            self.model: tf.keras.models.Model = tf.keras.models.load_model(model_path)
        elif model_path.is_dir():
            try:
                self.model: tf.keras.models.Model = tf.keras.models.load_model(model_path)
            except:
                print('[!] Not a saved model dir')
        else:
            print("[!] Unsupported type")

        self.path_to_save = path_to_save
        self.model_info = model_info

    def convert_model(self,
                      convert_to:list,
                      quantize:dict = None,
                      extra_params:dict = None,
                      ):

        self.path_to_save.mkdir(exist_ok=True, parents=True)

        for convert_format in convert_to:
            if convert_format == 'H5':
                path_to_save_model = self.path_to_save.joinpath(convert_format)
                path_to_save_model.mkdir(parents=True, exist_ok=True)
                self.model.save(path_to_save_model.joinpath('model.h5'))

                self.model_info['Format'] = convert_format
                self.model_info['File'] = 'model.h5'
                self.generate_json(**self.model_info)
                path_to_save_model.joinpath('model.json').write_text(json.dumps(self.settings, indent=4))

                print('[+] Saved model done !')

            if convert_format == 'PB':
                path_to_save_model = self.path_to_save.joinpath(convert_format)
                path_to_save_model.mkdir(parents=True, exist_ok=True)

                full_model = tf.function(lambda x: self.model(x))
                model_function = full_model.get_concrete_function(
                    tf.TensorSpec(shape=self.model.input.shape, dtype=tf.float32))
                frozen_function = convert_variables_to_constants_v2(model_function)
                frozen_function.graph.as_graph_def()

                # frozen_inp_outp = {'InputLayer': frozen_function.graph.get_operations()[0].name,
                #                    'OutputLayer': frozen_function.graph.get_operations()[-1].name,
                #                    }

                tf.io.write_graph(graph_or_graph_def=frozen_function.graph, logdir=path_to_save_model.as_posix(),
                                  name=f"model.pb",
                                  as_text=False)

                self.model_info['Format'] = convert_format
                self.model_info['File'] = 'model.pb'
                self.generate_json(**self.model_info)
                path_to_save_model.joinpath('model.json').write_text(json.dumps(self.settings, indent=4))

                print('[+] Saved frozen model !')

            if convert_format == 'TFlite':
                path_to_save_model = self.path_to_save.joinpath(convert_format)
                path_to_save_model.mkdir(parents=True, exist_ok=True)

                converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
                    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
                ]
                if quantize:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                path_to_save_model.joinpath('model.tflite').write_bytes(tflite_model)

                self.model_info['Format'] = convert_format
                self.model_info['File'] = 'model.tflite'
                self.generate_json(**self.model_info)
                path_to_save_model.joinpath('model.json').write_text(json.dumps(self.settings, indent=4))

                print('[+] Saved TFLite model !')

            if  convert_format == 'ONNX':
                path_to_save_model = self.path_to_save.joinpath(convert_format)
                path_to_save_model.mkdir(parents=True, exist_ok=True)

                tf2onnx.convert.from_keras(model=self.model,
                                           opset=extra_params['opset'] if extra_params.get('opset') else 12,
                                           output_path=path_to_save_model.joinpath('model.onnx'),
                                           )

                self.model_info['Format'] = convert_format
                self.model_info['File'] = 'model.onnx'
                self.generate_json(**self.model_info)
                path_to_save_model.joinpath('model.json').write_text(json.dumps(self.settings, indent=4))

                print('[+] Base ONNX model saved! Trying to simplify it')

                onnx_model = onnx.load(path_to_save_model.joinpath('model.onnx').as_posix())
                model_simp, check = simplify(onnx_model)
                assert check, "Simplified ONNX model could not be validated"
                onnx.save(model_simp, path_to_save_model.joinpath('model.onnx').as_posix())

                print('[+] Saved Simplified ONNX model !')

            if convert_format == 'CoreML':
                path_to_save_model = self.path_to_save.joinpath(convert_format)
                path_to_save_model.mkdir(parents=True, exist_ok=True)
                model = ct.convert(self.model,
                                   inputs=[ct.ImageType(bias=[0,0,0], scale=1, channel_first=False)],)

                self.model_info['Format'] = convert_format
                self.model_info['File'] = 'model.mlmodel'
                self.generate_json(**self.model_info)

                model.save(path_to_save_model.joinpath(self.model_info['File']).as_posix())
                path_to_save_model.joinpath('model.json').write_text(json.dumps(self.settings, indent=4))

                print('[+] Saved CoreML model !')



    def generate_json(self, **kwargs):

        self.settings = {}

        self.settings['Name'] = self.model.name

        assert kwargs['Type'] in ['BinaryClassification', 'MultiLabelClassification', 'Metric',
                                  'YoloDetector', 'YoloSegmentor', 'OCR'], "Model type is not allowed"
        self.settings['Type'] = kwargs['Type']

        self.settings['Format'] = kwargs['Format']
        self.settings['File'] = kwargs['File']


        model_inputs = []
        for i, inp in enumerate(self.model.inputs):
            model_inputs.append({'Name': inp.name,
                                 'Shape': inp.shape.as_list()[1:],
                                 'Normalization':  kwargs['Normalization'][i] if kwargs.get('Normalization') else (0, 1),
                                 'PaddingSize': kwargs['PaddingSize'][i] if kwargs.get('PaddingSize') else (0, 0),
                                 'PaddingColor': kwargs['PaddingColor'][i] if kwargs.get('PaddingColor') else (0,0,0),
                                 })

        self.settings['Input'] = model_inputs


        model_outputs = []
        for i, outp in enumerate(self.model.outputs):
            outp_dict = {
                'Name': outp.name,
                'Shape': outp.shape.as_list()[1:],
            }
            if kwargs.get('Metric') is not None:
                outp_dict['Metric'] = kwargs['Metric'][i]

            model_outputs.append(outp_dict)
        self.settings['Output'] = model_outputs


        if self.settings['Type'] == 'Metric':
            assert kwargs['Centers'], "Centers file not found"

            self.path_to_save.joinpath(self.settings['Format'], 'resources').mkdir(parents=True, exist_ok=True)
            self.path_to_save.joinpath(self.settings['Format'], 'resources', 'centers' + kwargs['Centers'].suffix).\
                write_bytes(kwargs['Centers'].read_bytes())

            self.settings['Centers'] = r'resources\centers' + kwargs['Centers'].suffix

        elif self.settings['Type'] in ['YoloDetector', 'YoloSegmentor']:
            self.settings['IOU'] = kwargs['IOU']
            self.settings['CLS'] = kwargs['CLS']
            self.settings['Labels'] = kwargs['Labels']
            if kwargs.get('MaskFilter') is not None:
                self.settings['MaskFilter'] = kwargs['MaskFilter']
        elif self.settings['Type'] in ['MultiLabelClassification']:
            self.settings['Labels'] = kwargs['Labels']
        elif self.settings['Type'] == 'BinaryClassification':
            self.settings['Labels'] = kwargs['Labels']
            self.settings['Threshold'] = kwargs['Threshold']

        return self.settings








