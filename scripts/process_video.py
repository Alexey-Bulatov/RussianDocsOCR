import os
import sys
import time

import cv2

sys.path.append('..')
from document_processing import Pipeline
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    """Runs document analysis pipeline on live webcam stream.

    Initializes camera capture and document analysis pipeline.
    Displays live video overlayed with pipeline results including:

    - Detected document type
    - OCR extracted text
    - Frame rate

    The pipeline inference runs continously on a separate thread
    to maximize frame rate. Output is printed and displayed.

    Performs inference using OpenVINO models on CPU by default.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1440)
    cap.set(4, 720)

    pipeline = Pipeline(model_format='OpenVINO', device='cpu', )

    frames = 0
    fps = 0
    frame_time = time.time()
    frame_time_in_sec = 0
    while True:

        frame_time = time.time() - frame_time
        frame_time_in_sec = frame_time_in_sec + frame_time
        frames += 1
        if frame_time_in_sec > 1:
            frame_time_in_sec = 0
            fps = frames
            frames = 0

        if fps != 0:
            print(f'fps = {fps} Frame performing time = {str(int(frame_time * 1000))}ms')

        frame_time = time.time()

        ret, img = cap.read()

        ###
        original_image = img.copy()
        cv2.putText(img, 'FPS = ' + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (260, 80, 80), 1)
        ###
        result = pipeline(original_image, check_quality=False, low_quality=False, docconf=0.2, img_size=1500)
        ocr_result = result.ocr
        print(ocr_result)
        ###
        cv2.imshow("Camera", img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        current_time = time.time()

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
