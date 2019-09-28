import numpy as np
import cv2


class KalmanFilter(object):
    def __init__(self, point):
        self.dt = 0.2
        self.kalman = cv2.KalmanFilter(4, 2, 0)

        self.kalman.transitionMatrix = np.array(
            [[1., 0., self.dt, 0.], [0., 1., 0., self.dt], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        self.kalman.processNoiseCov = 0.5 * np.array([[self.dt ** 4.0 / 4.0, 0., self.dt ** 3.0 / 2.0, 0.],
                                                      [0., self.dt ** 4.0 / 4.0, 0., self.dt ** 3.0 / 2.0],
                                                      [self.dt ** 3.0 / 2.0, 0., self.dt ** 2.0, 0.],
                                                      [0., self.dt ** 3.0 / 2.0, 0., self.dt ** 2.0]], dtype=np.float32)
        self.kalman.processNoiseCov[2:, 2:] *= 1
        self.kalman.processNoiseCov[3:, 3:] *= 1

        self.kalman.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
        self.kalman.measurementNoiseCov[2:, 2:] *= 1
        self.kalman.measurementNoiseCov[3:, 3:] *= 1

        self.kalman.errorCovPost = 1e-1 * np.eye(4, dtype=np.float32)
        self.kalman.errorCovPost[2:, 2:] *= 1
        self.kalman.errorCovPost[3:, 3:] *= 1

        self.kalman.statePost = np.array([[1. * point[0][0]], [1. * point[1][0]], [0], [0]], dtype=np.float32)
        self.lastResult = np.array([[1. * point[0][0]], [1. * point[1][0]]], dtype=np.float32)

    def predict(self):
        # print('state before predict: ', self.kalman.statePost)
        prediction = self.kalman.predict()
        # print('state after predict: ', prediction)
        ix = prediction[0][0]
        iy = prediction[1][0]
        self.lastResult = np.array([[ix], [iy]], dtype=np.float32)
        # print('last result after predict: ', self.lastResult)
        return self.lastResult

    def correct(self, b, flag):
        if not flag:  # update using prediction
            measurement = self.lastResult
        else:  # update using detection
            measurement = np.array([[b[0][0]], [b[1][0]]], dtype=np.float32)
        # print('Flag: ', flag)
        # print('Measurement: ', measurement)

        y = measurement - np.dot(self.kalman.measurementMatrix, self.kalman.statePre)
        # print(y)
        C = np.dot(np.dot(self.kalman.measurementMatrix, self.kalman.errorCovPre),
                   self.kalman.measurementMatrix.T) + self.kalman.measurementNoiseCov
        # print(C.shape)
        K = np.dot(np.dot(self.kalman.errorCovPre, self.kalman.measurementMatrix.T), np.linalg.inv(C))
        # print(K.shape)

        self.kalman.statePost = self.kalman.statePre + np.dot(K, y)
        self.kalman.errorCovPost = self.kalman.errorCovPre - np.dot(K, np.dot(C, K.T))

        estimate = self.kalman.statePost
        # print('Estimate: ', estimate)
        self.lastResult = np.array([[estimate[0][0]], [estimate[1][0]]], dtype=np.float32)
        # print('Last result: ', self.lastResult)
        return self.lastResult
