import os
import numpy as np
import cv2
import shutil

IMDIM_X = 1024
IMDIM_Y = 540

def format_data():

    np.random.seed(7)

    lib0 = './UAV-benchmark-M/'
    lib1 = './yoloformat/'
    vids = os.listdir(lib0)

    idx = 0

    for vid in vids:
        labels_file = './labels/UAV-benchmark-MOTD_v1.0/GT/%s_gt_whole.txt' %vid
        labels = np.loadtxt(labels_file, dtype=np.int32, delimiter=',')
        frame_idx = labels[:, 0]

        train, val, test = split_data_indeces(frame_idx.max())

        for i, imfile in enumerate(os.listdir(lib0 + vid)):

            obj_idx = np.where(frame_idx==i+1)[0]
            obj_rows = labels[obj_idx, :]

            xcenter = (obj_rows[:, 2] + obj_rows[:, 4]//2) / IMDIM_X
            ycenter = (obj_rows[:, 3] + obj_rows[:, 5]//2) / IMDIM_Y

            yolo_label = np.stack((obj_rows[:, -1]-1, xcenter, ycenter, obj_rows[:, 4] / IMDIM_X, obj_rows[:, 5] / IMDIM_Y)).T

            if i in val:
                np.savetxt(lib1 + '/labels/val/%06d.txt'%idx, yolo_label, fmt=['%d', '%.6f', '%.6f', '%.6f', '%.6f'])
                shutil.copy(src=lib0 + vid + '/' + imfile, dst=lib1 + '/images/val/%06d.jpg'%idx)

            elif i in test:
                np.savetxt(lib1 + '/labels/test/%06d.txt'%idx, yolo_label, fmt=['%d', '%.6f', '%.6f', '%.6f', '%.6f'])
                shutil.copy(src=lib0 + vid + '/' + imfile, dst=lib1 + '/images/test/%06d.jpg'%idx)

            else:
                np.savetxt(lib1 + '/labels/train/%06d.txt'%idx, yolo_label, fmt=['%d', '%.6f', '%.6f', '%.6f', '%.6f'])
                shutil.copy(src=lib0 + vid + '/' + imfile, dst=lib1 + '/images/train/%06d.jpg'%idx)

            idx += 1

def split_data_indeces(maxval, train=0.6, val=0.2, test=0.2):

        jumbled = np.random.permutation(maxval)
        cutoffs = np.cumsum(np.round(np.array([train, val, test]) * jumbled.size).astype(np.int32))

        return jumbled[:cutoffs[0]], jumbled[cutoffs[0]:cutoffs[1]], jumbled[cutoffs[1]:]

def test_yolo_data():
    lib = './yoloformat/'

    imdir = 'images/val/'
    labeldir = 'labels/val/'
    
    ims = os.listdir(lib + imdir)
    labels = os.listdir(lib + labeldir)

    colors = np.random.randint(0, 255, (3, 3)).tolist()

    for i in range(len(ims)):

        im = cv2.imread(lib + imdir + ims[i])

        imlabel = np.loadtxt(lib + labeldir + labels[i]).reshape(-1, 5)

        for label in imlabel:
            x0 = int((label[1] - label[3]/2) * IMDIM_X)
            y0 = int((label[2] - label[4]/2) * IMDIM_Y)

            xf = int((label[1] + label[3]/2) * IMDIM_X)
            yf = int((label[2] + label[4]/2) * IMDIM_Y)

            cv2.rectangle(im, (x0, y0), (xf, yf), colors[int(label[0])])

        cv2.imshow('', im)
        cv2.waitKey(10)

def vidshow():
    lib = './UAV-benchmark-M/'
    vids = os.listdir(lib)
    vid = np.random.choice(vids)
    colors = np.random.randint(0, 255, (3, 3)).tolist()

    labels_file = './labels/UAV-benchmark-MOTD_v1.0/GT/%s_gt_whole.txt' %vid
    labels = np.loadtxt(labels_file, dtype=np.int32, delimiter=',')
    frame_idx = labels[:, 0]


    for i, imfile in enumerate(os.listdir(lib + vid)):
        im = cv2.imread(lib + vid + '/' + imfile)

        obj_idx = np.where(frame_idx==i+1)[0]
        obj_rows = labels[obj_idx, :]

        for row in obj_rows:
            cv2.rectangle(im, (row[2], row[3]), (row[2] + row[4], row[3] + row[5]), colors[row[-1]])


        cv2.imshow('im', im)
        cv2.waitKey(10)


if __name__ == "__main__":
    test_yolo_data()
    # vidshow()
    # format_data()