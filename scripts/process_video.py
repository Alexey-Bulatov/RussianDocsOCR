import cv2
import os
import time
import sys
from document_processing import Pipeline
from process_img import process_img
from pathlib import Path
import pprint
import argparse
import warnings

sys.path.append('..')
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

    # webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(3, 1920)
    # cap.set(4, 1080)
    cap.set(3, 1440)
    cap.set(4, 720)

    # mobile
    # cap = cv2.VideoCapture('http://192.168.0.1:8080/video', cv2.CAP_ANY)
    # cap.set(3, 1920)
    # cap.set(4, 1080)

    pipeline = Pipeline(model_format='ONNX', device='gpu', )

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
        if img is not None:
            original_image = img.copy()
        else:
            print('Camera is not connected. Check camera connection.')
            cap.release()
            cv2.destroyAllWindows()
            break

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