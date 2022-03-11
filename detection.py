import torch
import numpy as np
import cv2
from time import time, sleep

from speed import VehicleCounter


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a input video using Opencv2.
    """
    MIN_WIDTH=30 #Minimum width of rectangle
    MIN_HEIGHTn=40 #Minimum height of rectangle
    OFFSET=15 #Offset for centroid
    LINE_POS=400 #Position of line in frame
    DELAY= 60 #FPS do vídeo

    def __init__(self, url, out_file="Labeled_Video.avi"):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detec = []
        self.cars = 0
        self.car_counter = None

    def get_video_from_url(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        #return cv2.VideoCapture(self._URL)
        return cv2.VideoCapture(0)


    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def centroid_calc(self, x, y, w, h):
        """
        Calculates the centroid of a bounding rectangle
        :param x: lower left x-coordinate of the bounding rectangle
        :param y: lower left y-coordinate of the bounding rectange
        :param w: width of the bounding rectangle
        :param h: height of the bounding rectange
        """
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx,cy

    def plot_boxes(self, results, frame, frame_num):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        matches = []

        cv2.line(frame, (300, self.LINE_POS), (1200, self.LINE_POS), (255,127,0), 3)
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                centroid = self.centroid_calc(x1, y1, x2-x1, y2-y1)
                matches.append(((x1, y1, x2-x1, y2-y1), centroid))
                if centroid[0] > 300 and centroid[0] < 1200 and centroid[1]<(self.LINE_POS+self.OFFSET) and centroid[1]>(self.LINE_POS-self.OFFSET):
                    self.cars+=1
                    cv2.line(frame, (300, self.LINE_POS), (1200, self.LINE_POS), (0,127,255), 3)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.circle(frame, centroid, 3, (0, 0,255), -1)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        
        self.car_counter.update_count(matches, frame_num, frame)
        cv2.putText(frame, "VEHICLE COUNT : "+str(self.cars), (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),2)
        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        player = self.get_video_from_url()
        assert player.isOpened()
        frame_num = -1
        
        while True:
            start_time = time()
            ret, frame = player.read()
            frame_num += 1
            assert ret
            tempo = float(1/self.DELAY)
            sleep(0.2) 

            if self.car_counter is None:
                self.car_counter = VehicleCounter(frame.shape[:2], self.LINE_POS, player.get(cv2.CAP_PROP_FPS))

            #TODO add a resize to fix vid size at 640:480
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame, frame_num)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")
            cv2.imshow("Video" , frame)

            if cv2.waitKey(1) == 27:
                break

# Create a new object and execute.
a = ObjectDetection("AVDS 8.mp4")
a()