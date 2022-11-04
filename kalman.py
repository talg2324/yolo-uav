import numpy as np
from scipy.spatial.distance import cdist
from track_utils import flow_at_point, is_in_frame, random_color

class KalmanObjectTracker:
    def __init__(self, xc, yc):
        """
        Tracking Object using the Kalman Filter algorithm.
        Assign a color and serial number to the object
        Initialize the matrices needed for Kalman Equations

        A - state-transition model
        B - control-input model
        H - observation model
        P - covariance matrix
        Q - covariance of process noise
        R - covariance of observation noise
        """

        self.x = np.array([xc, yc, 0, 0], dtype=np.float32)
        
        self.A = np.array([[1, 0, 1, 0], 
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.At = self.A.T

        self.B = np.array([[1, 0], 
                           [0, 1],
                           [0, 0],
                           [0, 0]], dtype=np.float32)

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.Ht = self.H.T

        self.P = np.eye(4, dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32)
        self.I = np.eye(4, dtype=np.float32)

        self.color = random_color()
        self.serial_no = np.random.randint(10000, 100000)

    def iter(self, detections, flow):
        """
        Input: 
                detections - list of centroids found by yolo as (xn, yn)
                flow - dense optical flow

        Perform a Kalman Filter Iteration:
            1. Predict the prior: xhat = A @ x + B @ u
            2. Update the posterior: x = xhat + K @ (z - H @ xhat)
        """

        in_frame = False
        min_idx = -1

        if detections.shape[0] > 0:
        
            u = flow_at_point(self.x, flow)
            xhat = self.A @ self.x + self.B @ u

            self.P = self.A @ self.P @ self.At + self.Q
            K = self.P @ self.Ht @ np.linalg.pinv(self.H @ self.P @ self.Ht + self.R)

            distances = cdist(np.expand_dims(xhat[:2], axis=0), detections)
            min_idx = np.argmin(distances)

            if distances[0, min_idx] < 5:

                z = detections[min_idx, :]

                self.x = xhat + K @ (z - self.H @ xhat)
                self.P = (self.I - K @ self.H) @ self.P

                in_frame = is_in_frame(self.x)

        return in_frame, min_idx

    def get_pos(self):
        return int(self.x[0]), int(self.x[1])