import pathlib
import cv2
import time
import numpy as np


class QualityChecker(object):

    def __init__(self, init_model, init_canvas_size):
        self.model = init_model
        # canvas size in blocks
        self.canvas_size = init_canvas_size
        # window size must be 128
        self.window_size = 128
        # colors for drawing
        self.colors = {'c_blue': (260, 80, 80), 'c_white': (255, 255, 255), 'c_gray': (160, 160, 160), 'c_gray_f': (96,
                                                                                                                    96,
                                                                                                                    96),
                       'c_red': (255, 0, 0), 'c_redish': (255, 153, 153), 'c_yellow': (255, 255, 0),
                       'c_green': (173, 255, 47)}
        # list of (class, coordinates, confidence)
        self.quality_result_list = []
        # drawn image
        self.tested_image = np.ndarray([])

    def perform_image(self, image):
        self.quality_result_list = []
        canvas_in_pixels = tuple(map(lambda x: x * self.window_size, self.canvas_size))
        self.tested_image = cv2.cvtColor(cv2.resize(image, canvas_in_pixels), cv2.COLOR_BGR2RGB)
        # self.tested_image = cv2.resize(image, canvas_in_pixels)

        for x_step in range(self.canvas_size[0]):
            for y_step in range(self.canvas_size[1]):
                x = self.window_size * x_step
                y = self.window_size * y_step
                frame_image = self.tested_image[y:y + self.window_size, x:x + self.window_size]
                # print(frame_image)
                result = self.model.predict(frame_image)
                result_class = result[0]
                confidence = result[1]
                # print(confidence, x_step, y_step)
                # class, coordinates, confidence
                self.quality_result_list.append((result_class, ((x, y), (x + self.window_size, y + self.window_size)),
                                                 confidence))

        self.tested_image = cv2.cvtColor(self.tested_image, cv2.COLOR_RGB2BGR)
        return True

    def annotate_image(self, image):
        self.perform_image(image)

        for block in self.quality_result_list:

            cv2.rectangle(self.tested_image, block[1][0], block[1][1], color=self.colors[
                "c_white"], thickness=1)
            result = block[0]
            confidence = block[2]
            x = block[1][0][0]
            y = block[1][0][1]
            # print(result)

            if result == 'NO':
                cv2.putText(self.tested_image, 'NO', (x + 14, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            self.colors['c_green']
                            , 2)
                cv2.putText(self.tested_image, str(round(confidence, 2)), (x + 14, y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            self.colors['c_green']
                            , 2)


            if result == 'GLARE':
                cv2.putText(self.tested_image, 'GLARE', (x + 14, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[
                    "c_red"], 2)
                cv2.putText(self.tested_image, str(round(confidence, 2)), (x + 14, y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[
                    "c_red"], 2)

        return self.tested_image

    def check_image_quality(self, image):
        self.perform_image(image)
        result_list = []
        for block in self.quality_result_list:
            result = block[0]
            confidence = block[2]
            # print(result, confidence)

            if result == 'GLARE' and confidence > 0.85:
                result_list.append(0)
            else:
                result_list.append(1)

        max_level_for_normalization = len(result_list)
        quality_level = 0
        for block in result_list:
            quality_level += block
        quality_level = 1 - quality_level / max_level_for_normalization
        return quality_level

    def video_processing(self):
        frame_for_print = '0'
        start = time.time()
        frame = 0

        canvas_in_pixels = tuple(map(lambda x: x * self.window_size, self.canvas_size))

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, canvas_in_pixels[0])
        cap.set(4, canvas_in_pixels[1])

        while True:
            time_diff = time.time() - start
            frame += 1

            if time_diff > 1:
                frame_for_print = str(frame)
                # print('FPS = ' + frame_for_print)
                frame = 0
                start = time.time()

            ret, img = cap.read()
            ##############################
            cv2.putText(img, 'FPS = ' + frame_for_print, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["c_blue"]
                        , 2)
            img = self.annotate_image(img)
            ##############################
            cv2.imshow("camera", img)
            ##############################

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        return True