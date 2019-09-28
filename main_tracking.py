from detectors import YOLO
import cv2
from PIL import Image
from tracker_center_demo import Tracker
import numpy as np
from keras import backend as K
import math
import matplotlib.pyplot as plt
import colorsys
import queue
import threading
import time

ix, iy, ex, ey = -1, -1, -1, -1
frame_sz = (1280, 720)
cap_from_stream = False
# path = 'rtsp://192.168.10.246:554'
path = 'cam246.mp4'
hsv_tuples = [(x / 80, 1., 1.)
              for x in range(80)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

mapped_label = {'car': 70, 'truck': 10, 'motorbike': 42, 'bicycle': 18, 'person': 7, 'bus': 52}

q = queue.Queue()
running = True


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = np.array(boxA).reshape((4,))
    boxB = np.array(boxB).reshape((4,))
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.5
    K.set_session(K.tf.Session(config=cfg))


def draw_rec(event, x, y, flags, param):
    global ix, iy, ex, ey, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        ex, ey = x, y
        cv2.rectangle(param, (ix, iy), (x, y), (0, 255, 0), 0)


def get_crop_size(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if cap_from_stream:
            frame = cv2.resize(frame, frame_sz)
        cv2.namedWindow('draw_rectangle')
        cv2.setMouseCallback('draw_rectangle', draw_rec, frame)
        print("Choose your area of interest!")
        while 1:
            cv2.imshow('draw_rectangle', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('a'):
                cv2.destroyAllWindows()
                break
        break


def grab(cam, queue, delay):
    global running

    capture = cv2.VideoCapture(cam)
    now = time.time()

    if type(cam) == int:
        while running:
            fr = {}
            capture.grab()
            retval, img = capture.read()
            if not retval:
                continue
            fr["img"] = img
    else:
        while running:
            fr = {}
            retval, img = capture.read()
            if not retval:
                continue
            fr["img"] = img
            cur = time.time()
            if cur - now >= delay:
                queue.put(fr)
                now = cur


def main():
    global running
    # limit_mem()
    # Choose area of interest
    get_crop_size(path)
    print('Your area of interest: ', ix, ' ', iy, ' ', ex, ' ', ey)
    area = (ix, iy, ex, ey)
    S_ROI = (ex - ix + 1) * (ey - iy + 1)

    # Create opencv video capture object
    cap = cv2.VideoCapture(path)
    w = int(cap.get(3))
    h = int(cap.get(4))
    if cap_from_stream:
        w = frame_sz[0]
        h = frame_sz[1]

    video_fps = 25  # cap.get(cv2.CAP_PROP_FPS)
    meter_per_pixel = 0.05
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('res.avi', fourcc, video_fps, (w, h))

    # Create Object Detector
    detector = YOLO()

    # Create Object Tracker
    tracker = Tracker(dist_thresh=30.0, max_frames_to_skip=5, max_trace_length=10, trackIdCount=0)

    # Variables initialization
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]

    count_vehicle = {'person': 0, 'motorbike': 0, 'car': 0, 'truck': 0, 'bicycle': 0, 'bus': 0}

    # Variables to plot
    xar = []
    yar = {'person': [], 'motorbike': [], 'car': [], 'truck': [], 'bicycle': [], 'bus': []}
    count_local = {'person': 0, 'motorbike': 0, 'car': 0, 'truck': 0, 'bicycle': 0, 'bus': 0}
    time_counter = 0
    frame_counter = 0
    total_frame = 0
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    capture_thread = threading.Thread(target=grab, args=(path, q, 0))
    capture_thread.start()

    while True:  # cap.isOpened():
        # Capture frame-by-frame
        try:

            fr = q.get()
            print(q.qsize())
            frame = fr["img"]

            # if not ret:
            #    continue
            if cap_from_stream:
                frame = cv2.resize(frame, frame_sz)
            frame = Image.fromarray(frame)

            # Detect and return centeroids of the objects in the frame
            result, centers, box_detected, obj_type, conf_score = detector.detect_image(frame, area)
            result = np.asarray(result)

            for i in range(len(box_detected)):
                left, top, right, bottom = box_detected[i]
                predicted_class = obj_type[i]
                score = conf_score[i]

                # name_str = '{}'.format(predicted_class)
                # conf_str = '{:.2f}p'.format(score)
                #
                # label = ''
                # #label += name_str
                # label += conf_str
                # font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                # ret, baseline = cv2.getTextSize(label, fontFace=font, fontScale=0.5, thickness=1)
                # cv2.rectangle(result, (left, top), (left + ret[0], top + ret[1] + baseline),
                #               color=colors[mapped_label[predicted_class]], thickness=-1)
                # cv2.putText(result, label, (left, top + ret[1] + baseline), font, 0.5, (0, 0, 0), 1,
                #             cv2.LINE_AA)

                output = result.copy()
                alpha = 0.3
                cv2.rectangle(result, (left, top), (right, bottom), colors[mapped_label[predicted_class]], -1)
                cv2.addWeighted(result, alpha, output, 1 - alpha, 0, output)

                result = output.copy()

                cv2.rectangle(result, (left, top), (right, bottom), colors[mapped_label[predicted_class]], 1)

            frame_counter += 1
            total_frame += 1
            print('Number of detections: ', len(centers))

            # Calculate density of vehicle
            S_total = 0
            for i in range(len(box_detected)):
                x, y, u, v = box_detected[i]
                S_total += (u - x + 1) * (v - y + 1)
            for i in range(len(box_detected) - 1):
                for j in range(i + 1, len(box_detected)):
                    S_total -= bb_intersection_over_union(box_detected[i], box_detected[j])
            density = S_total * 100.0 / S_ROI
            print('Density: ', density)

            velocity = {'person': [], 'motorbike': [], 'car': [], 'truck': [], 'bicycle': [], 'bus': []}
            # If centroids are detected then track them
            if len(box_detected) > 0:

                # Track object using Kalman Filter
                tracker.Update(box_detected, obj_type, total_frame, conf_score)

                # For identified object tracks draw tracking line
                # Use various colors to indicate different track_id
                for i in range(len(tracker.tracks)):
                    if len(tracker.tracks[i].trace) >= 2:
                        # for j in range(len(tracker.tracks[i].trace) - 1):
                        #     # Draw trace line
                        #     x1 = tracker.tracks[i].trace[j][0][0]
                        #     y1 = tracker.tracks[i].trace[j][1][0]
                        #     x2 = tracker.tracks[i].trace[j + 1][0][0]
                        #     y2 = tracker.tracks[i].trace[j + 1][1][0]
                        #     clr = tracker.tracks[i].track_id % 9
                        #     cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)),
                        #              track_colors[clr], 2)
                        classes = tracker.tracks[i].get_obj()

                        # Counting vehicle
                        if (len(tracker.tracks[i].trace) >= 8) and (not tracker.tracks[i].counted):
                            tracker.tracks[i].label = classes
                            tracker.tracks[i].counted = True
                            count_vehicle[classes] += 1
                            count_local[classes] += 1
                            if tracker.tracks[i].ground_truth_box is not None:
                                bbox = tracker.tracks[i].ground_truth_box.reshape((4, 1))
                                cv2.rectangle(result, (bbox[0][0], bbox[1][0]), (bbox[2][0], bbox[3][0]),
                                              color=(255, 0, 255),
                                              thickness=3)

                        # Calculate velocity
                        if (len(tracker.tracks[i].trace) >= 8) and tracker.tracks[i].has_truebox:
                            j = len(tracker.tracks[i].trace) - 2
                            x1 = tracker.tracks[i].trace[j][0][0]
                            y1 = tracker.tracks[i].trace[j][1][0]
                            x2 = tracker.tracks[i].trace[j + 1][0][0]
                            y2 = tracker.tracks[i].trace[j + 1][1][0]
                            d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            print(d, video_fps)
                            v1 = d * (video_fps - 1) * meter_per_pixel  # m/s
                            v2 = v1 * 3.6  # km/h
                            velocity[tracker.tracks[i].label].append(v2)

                            bbox = tracker.tracks[i].ground_truth_box.reshape((4, 1))
                            left, top = bbox[0][0], bbox[1][0]
                            conf_str = '{:.2f} km/h'.format(v2)

                            label = ''
                            label += conf_str
                            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                            ret, baseline = cv2.getTextSize(label, fontFace=font, fontScale=0.5, thickness=1)
                            cv2.rectangle(result, (left, top), (left + ret[0], top + ret[1] + baseline),
                                          color=colors[mapped_label[classes]], thickness=-1)
                            cv2.putText(result, label, (left, top + ret[1] + baseline), font, 0.5, (0, 0, 0), 1,
                                        cv2.LINE_AA)

            # Plot graph
            if frame_counter == int(video_fps * 3):
                frame_counter = 0
                time_counter += 3
                xar.append(time_counter)
                for key, value in count_local.items():
                    yar[key].append(value)
                    count_local[key] = 0
            ax.clear()
            ax.set_ylim(bottom=0, top=50)
            ax.set_xlim(left=0, right=300)
            ax.set_xlabel('Timer (s)')
            ax.set_ylabel('Number of vehicle')
            for key, value in yar.items():
                ax.plot(xar, value, label=key)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (640, 640))
            cv2.imshow("plot", img)

            # ------------------------------------ DISPLAY RESULT ------------------------------------------#
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            # Display density and road status
            if density <= 25.:
                status = 'Normal'
                color_status = (255, 255, 255)
            elif density <= 50.:
                status = 'High'
                color_status = (255, 0, 0)
            else:
                status = 'Very high'
                color_status = (0, 0, 255)
            text = 'Density: %.2f %%. Status: %s' % (density, status)
            left = 30
            top = frame_sz[1] - 50
            ret, baseline = cv2.getTextSize(text, fontFace=font, fontScale=0.7, thickness=1)
            cv2.rectangle(result, (left, top), (left + ret[0], top + ret[1] + baseline),
                          color=color_status, thickness=-1)
            cv2.putText(result, text, (left, top + ret[1] + baseline), font, 0.7, (0, 0, 0), 1,
                        cv2.LINE_AA)

            text = 'TRAFFIC MONITORING SYSTEM'
            left = 0
            top = 0
            ret, baseline = cv2.getTextSize(text, fontFace=font, fontScale=1.3, thickness=1)
            cv2.rectangle(result, (left, top), (left + ret[0], top + ret[1] + baseline),
                          color=(255, 255, 255), thickness=-1)
            cv2.putText(result, text, (left, top + ret[1] + baseline), font, 1.3, (0, 0, 0), 1,
                        cv2.LINE_AA)

            # Display counter and velocity
            x = 30
            y = 30
            i = 1

            ret, baseline = cv2.getTextSize('100000000000', fontFace=font, fontScale=0.7, thickness=1)
            left = x
            top = y
            cv2.rectangle(result, (left, top), (left + ret[0], top + ret[1] + baseline),
                          color=(255, 255, 255), thickness=-1)
            cv2.putText(result, 'Classes', (left, top + ret[1] + baseline), font, 0.7, (0, 0, 0), 1,
                        cv2.LINE_AA)

            top = y + ret[1] + baseline + 2
            cv2.rectangle(result, (left, top), (left + ret[0], top + ret[1] + baseline),
                          color=(255, 255, 255), thickness=-1)
            cv2.putText(result, 'Counter', (left, top + ret[1] + baseline), font, 0.7, (0, 0, 0), 1,
                        cv2.LINE_AA)

            top = y + 2 * (ret[1] + baseline) + 4
            cv2.rectangle(result, (left, top), (left + ret[0], top + ret[1] + baseline),
                          color=(255, 255, 255), thickness=-1)
            cv2.putText(result, 'Velocity', (left, top + ret[1] + baseline), font, 0.7, (0, 0, 0), 1,
                        cv2.LINE_AA)

            for key in count_vehicle.keys():
                value = count_vehicle[key]
                text = str(value)
                ret, baseline = cv2.getTextSize('100000000000', fontFace=font, fontScale=0.7, thickness=1)
                left = x + i * ret[0]
                top = y
                cv2.rectangle(result, (left, top), (left + ret[0], top + ret[1] + baseline),
                              color=colors[mapped_label[key]], thickness=-1)
                cv2.putText(result, key, (left, top + ret[1] + baseline), font, 0.7, (0, 0, 0), 1,
                            cv2.LINE_AA)

                top = y + ret[1] + baseline + 2
                cv2.rectangle(result, (left, top), (left + ret[0], top + ret[1] + baseline),
                              color=colors[mapped_label[key]], thickness=-1)
                cv2.putText(result, text, (left, top + ret[1] + baseline), font, 0.7, (0, 0, 0), 1,
                            cv2.LINE_AA)

                top = y + 2 * (ret[1] + baseline) + 4
                value = np.mean(velocity[key])
                if np.isnan(value):
                    text = 'N/A'
                else:
                    text = '%.2f km/h' % value
                cv2.rectangle(result, (left, top), (left + ret[0], top + ret[1] + baseline),
                              color=colors[mapped_label[key]], thickness=-1)
                cv2.putText(result, text, (left, top + ret[1] + baseline), font, 0.7, (0, 0, 0), 1,
                            cv2.LINE_AA)
                i += 1

            cv2.rectangle(result, (ix, iy), (ex, ey), (0, 255, 0), 0)
            cv2.imshow('Tracking', result)
            out.write(result)

            # Check for key strokes
            k = cv2.waitKey(1) & 0xff
            if k == ord('n'):
                continue
            elif k == 27:  # 'esc' key has been pressed, exit program.
                running = False
                break
        except Exception as inst:
            print(inst)
            pass

    # When everything done, release the capture
    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    main()
