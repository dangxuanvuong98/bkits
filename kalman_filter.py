import numpy as np
import cv2


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.], dtype=np.float32).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score], dtype=np.float32).reshape(
            (1, 5))


class KalmanFilter(object):
    def __init__(self, bbox):
        self.kalman = cv2.KalmanFilter(7, 4, 0)

        self.dt = 0.2

        self.kalman.transitionMatrix = np.array(
            [[1, 0, 0, 0, self.dt, 0, 0], [0, 1, 0, 0, 0, self.dt, 0], [0, 0, 1, 0, 0, 0, self.dt],
             [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]],
            dtype=np.float32)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
            dtype=np.float32)

        self.kalman.processNoiseCov = 1e-1 * np.eye(7, dtype=np.float32)
        self.kalman.processNoiseCov[-1, -1] *= 0.01
        self.kalman.processNoiseCov[4:, 4:] *= 0.01

        self.kalman.measurementNoiseCov = 1e-1 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov[2:, 2:] *= 10

        self.kalman.errorCovPost = 1e-1 * np.eye(7, dtype=np.float32)
        self.kalman.errorCovPost[4:, 4:] *= 100  # give high uncertainty to the unobservable initial velocities
        #self.kalman.errorCovPost *= 10

        state = convert_bbox_to_z(bbox)
        self.kalman.statePost = np.array([[state[0][0]], [state[1][0]], [state[2][0]], [state[3][0]], [0], [0], [0]],
                                         dtype=np.float32)

        # lastResult saves the original box [x, y, s, r]
        self.lastResult = bbox

    def predict(self):
        if (self.kalman.statePost[6] + self.kalman.statePost[2]) <= 0:
            self.kalman.statePost[6] *= 0.0
        # print('state before predict: ', self.kalman.statePost)
        prediction = self.kalman.predict()
        # print('state after predict: ', prediction)

        # lastResult saves the original box [x, y, s, r]
        self.lastResult = convert_x_to_bbox(prediction)
        # print('last result after predict: ', self.lastResult)
        return self.lastResult

    def correct(self, bbox, flag):
        if not flag:  # update using prediction
            measurement = self.lastResult
        else:  # update using detection
            measurement = bbox

        # print('Flag: ', flag)
        # if not flag:
        #    print('Measurement before: ', measurement)
        # print('bbox: ', bbox)
        measurement = convert_bbox_to_z(measurement.reshape((4, 1)))
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

        # lastResult saves the original box [x, y, s, r]
        self.lastResult = convert_x_to_bbox(estimate)

        # print('Last result: \n', self.lastResult)
        return self.lastResult
