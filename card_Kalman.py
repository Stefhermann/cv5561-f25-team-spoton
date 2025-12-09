import numpy as np
import cv2

class KalmanCardBox:
    def __init__(self):
        # 8 state variables (x1, y1, x2, y2, vx1, vy1, vx2, vy2)
        # 4 measurement values (x1, y1, x2, y2)
        self.kf = cv2.KalmanFilter(8, 4)

        self.kf.measurementMatrix = np.zeros((4, 8), np.float32)
        self.kf.measurementMatrix[:4, :4] = np.eye(4, dtype=np.float32)

        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i+4] = 1  # add velocity

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.3

        self.initialized = False

    def update(self, bbox):
        # bbox is [x1, y1, x2, y2]
        bbox = np.array(bbox, dtype=np.float32).reshape(4, 1)

        if not self.initialized:
            # Initialize state
            self.kf.statePre[:4] = bbox
            self.kf.statePre[4:] = 0
            self.initialized = True

        self.kf.predict()

        corrected = self.kf.correct(bbox)
        return corrected[:4].reshape(4).tolist()
