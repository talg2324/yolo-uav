import cv2
import numpy as np
import torch
from track_utils import *
from kalman import KalmanObjectTracker

from yolov5.utils.general import non_max_suppression, scale_boxes

def main():  
    """
    Pick a video file that's been split into .jpeg images
    Load a YOLOv5 model and dataloader
    Track individual objects using the YOLO predictions as Kalman filter input
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_yolo_model(device)
    uav_dataset = load_dataset()

    conf_thres=0.45  # confidence threshold
    iou_thres=0.05   # NMS IOU threshold
    max_det=100      # maximum detections per image

    objects = []
    for path, im, im0, vid_cap, s in uav_dataset:
        im = torch.from_numpy(im).to(model.device).float()
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)[0]

        pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], im0.shape).round()

        flow = dense_flow(cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY))

        detections = []
        for x0_, y0_, xf_, yf_, confidence, label in pred:

            x0 = int(x0_)
            y0 = int(y0_)
            xf = int(xf_)
            yf = int(yf_)

            xc = (x0 + xf) // 2
            yc = (y0 + yf) // 2

            w = (xf - x0) // 2
            h = (yf - y0) // 2

            detections.append((xc, yc, w, h))

        # Match the objects we've tracked
        next_objects = []
        detection_array = np.array(detections)[:, :2]

        for object in objects:
            in_frame, obj_idx = object.iter(detection_array, flow)

            if in_frame:
                next_objects.append(object)

                p = object.get_pos()
                w, h = detections[obj_idx][2:]

                top_left     = (p[0] - w, p[1] - h)
                bottom_right = (p[0] + w, p[1] + h)

                cv2.rectangle(im0, top_left, bottom_right, object.color, 1)
                cv2.putText(im0, 
                            '%d' %object.serial_no,
                            top_left, 
                            cv2.FONT_HERSHEY_PLAIN, 
                            fontScale=1, 
                            color=(255,255,255))

            if obj_idx > 0:
                detections.pop(obj_idx)
                detection_array = np.delete(detection_array, obj_idx, axis=0)

        # New objects that entered this frame
        if len(detections):
            for xc, yc, w, h in detections:
                next_objects.append(KalmanObjectTracker(xc, yc))

        objects = next_objects.copy()

        cv2.imshow('Multi Object Tracking', im0)
        cv2.waitKey(10)

def vid_prior():  
    """
    Utility function to view the effect of different parameters on the apriori functions:

    - YOLOv5
    - NMS
    - Optical Flow

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_yolo_model(device)
    uav_dataset = load_dataset(vid_num=13)

    names = ['car', 'bus', 'truck']

    colors = [
                (255, 0, 0), # car
                (0, 255, 0), # bus
                (0, 0, 255) # truck
        ]
    
    conf_thres=0.45  # confidence threshold
    iou_thres=0.05   # NMS IOU threshold
    max_det=100      # maximum detections per image
        
    hsv = np.zeros((IMG_SIZE_Y, IMG_SIZE_X, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    for path, im, im0, vid_cap, s in uav_dataset:
        im = torch.from_numpy(im).to(model.device).float()
        im /= 255  # 0 - 255 to 0.0 - 1.0


        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)[0]

        pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], im0.shape).round()

        flow = dense_flow(cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY))
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for x0, y0, xf, yf, confidence, label in pred:
            color = colors[int(label)]
            p0 = (int(x0), int(y0))
            pf = (int(xf), int(yf))

            cv2.rectangle(im0, p0, pf, color, 1)
            cv2.putText(im0, 
                        '%s %.2f' %(names[int(label)], confidence),
                        p0, 
                        cv2.FONT_HERSHEY_PLAIN, 
                        fontScale=1, 
                        color=(255,255,255))

        cv2.imshow('', np.hstack((im0, bgr)))
        cv2.waitKey(10)

if __name__ == "__main__":
    main()