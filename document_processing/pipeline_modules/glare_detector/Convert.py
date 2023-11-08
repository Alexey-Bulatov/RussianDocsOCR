import tensorflow as tf
from datetime import datetime
from ModelConverter import Converter
from pathlib import Path


if __name__ == '__main__':

    model = tf.keras.models.load_model('model.h5')

    date_now = datetime.now().strftime('%Y-%m-%d')

    converter = Converter(
        model_path=model,
        model_info={
            'Type': 'BinaryClassification',
            'Labels': [
                    "NO",
                    "GLARE"
                ],
            'Threshold': 0.5
        },
        path_to_save=Path('Models'),
    )

    converter.convert_model(
        convert_to=['H5', 'TFlite', 'ONNX'],
        quantize={'float16': True},
        extra_params={
            'opset': 14,
        },
    )
