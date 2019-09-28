import numpy as np
from kalman_filter_center import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter(prediction)  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.counted = False
        self.trace = []  # trace path
        self.ground_truth_box = self.prediction
        self.cnt_obj = {'person': 0, 'motorbike': 0, 'car': 0, 'truck': 0, 'bicycle': 0, 'bus': 0}
        self.label = ''
        self.box = []
        self.conf_score = []
        self.has_truebox = False

    def update_obj(self, predicted_obj):
        self.cnt_obj[predicted_obj] += 1

    def get_obj(self):
        maxValue = -1
        obj = None
        for key, value in self.cnt_obj.items():
            if value > maxValue:
                maxValue = value
                obj = key
        return obj


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            iou_thresh: iou threshold. When smaller than the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, detections, obj_type, frame_idx, conf_score):
        # Create tracks if no tracks vector found
        if len(self.tracks) == 0:
            for i in range(len(detections)):
                center = [[(detections[i][0] + detections[i][2]) / 2.0], [(detections[i][1] + detections[i][3]) / 2.0]]
                track = Track(center, self.trackIdCount)
                track.update_obj(obj_type[i])
                self.trackIdCount += 1
                self.tracks.append(track)

        # print('Prediction: ')
        # for i in range(len(self.tracks)):
        #     print(self.tracks[i].prediction)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))  # Cost matrix
        cost2 = np.zeros(shape=(N, M))  # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    center = [[(detections[j][0] + detections[j][2]) / 2.0],
                              [(detections[j][1] + detections[j][3]) / 2.0]]
                    diff = self.tracks[i].prediction - center
                    distance = np.sqrt(diff[0][0] * diff[0][0] +
                                       diff[1][0] * diff[1][0])
                    cost2[i][j] = distance
                    if distance > self.dist_thresh:
                        distance = 1e9
                    cost[i][j] = distance
                except:
                    pass

        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        # print(cost2)
        # print(cost)
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]
        # print('Before:', assignment)

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1
        # print('After: ', assignment)
        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)
        # print('Number of undetected: ', len(un_assigned_detects), un_assigned_detects)
        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                j = un_assigned_detects[i]
                center = [[(detections[j][0] + detections[j][2]) / 2.0], [(detections[j][1] + detections[j][3]) / 2.0]]
                track = Track(center, self.trackIdCount)
                track.update_obj(obj_type[j])
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                j = assignment[i]
                center = [[(detections[j][0] + detections[j][2]) / 2.0], [(detections[j][1] + detections[j][3]) / 2.0]]
                self.tracks[i].prediction = self.tracks[i].KF.correct(center, 1)
                self.tracks[i].ground_truth_box = detections[j]
                self.tracks[i].update_obj(obj_type[j])
                self.tracks[i].conf_score.append(conf_score[assignment[i]])
                self.tracks[i].box.append((detections[j], frame_idx))
                self.tracks[i].has_truebox = True
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                    [[0.], [0.]], 0)
                self.tracks[i].ground_truth_box = None
                self.tracks[i].has_truebox = False

            if len(self.tracks[i].trace) > self.max_trace_length:
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]
            # print('pred', self.tracks[i].prediction)
            cx, cy = self.tracks[i].prediction[0][0], self.tracks[i].prediction[1][0]

            self.tracks[i].trace.append([[cx], [cy]])
            self.tracks[i].KF.lastResult = self.tracks[i].prediction
