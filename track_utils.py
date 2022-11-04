import os
import numpy as np
import cv2

from yolov5.utils.dataloaders import LoadImages
from yolov5.models.common import DetectMultiBackend

IMG_SIZE_X = 1024
IMG_SIZE_Y = 540

LAST_IM = np.zeros((IMG_SIZE_Y, IMG_SIZE_X), dtype=np.uint8)

def load_dataset(lib='./data/UAV-benchmark-M/', vid_num=None):
    """
    Get a yolo dataloader to read .jpg files from the data folder
    """

    vids = os.listdir(lib)

    if vid_num is None:
        vid_num = np.random.randint(len(vids))

    vid = vids[vid_num]
    
    return LoadImages(lib+vid, img_size=IMG_SIZE_X)

def load_yolo_model(device, weights='./yolov5/runs/train/exp11/weights/best.pt'):
    return DetectMultiBackend(weights, device=device).to(device)

def dense_flow(new_im):
    global LAST_IM
    flow = cv2.calcOpticalFlowFarneback(LAST_IM, new_im, None, 0.5, levels=2, winsize=25, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    LAST_IM = new_im.copy()
    return flow

def flow_at_point(x, flow):
    return flow[int(x[1]), int(x[0]), :]

def is_in_frame(x):

    retval = True

    xc = x[0]
    yc = x[1]

    if xc < 0 or xc > IMG_SIZE_X:
        retval &= False
    if yc < 0 or yc > IMG_SIZE_Y:
        retval &= False

    return retval

def random_color():
    return np.random.randint(0, 255, 3, dtype=np.uint8).tolist()