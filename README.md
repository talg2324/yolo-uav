# yolo-uav
Simple Kalman Filter implemented above the YOLOv5 algorithm to track vehicles in UAV footage

<p align="center" width="100%">
    <img width="75%" src="./sample/sample_vid.gif"> 
</p>

The YOLOv5 algorithm taken from https://github.com/ultralytics/yolov5 is transferred onto a UAV footage MOT dataset:
https://paperswithcode.com/sota/multi-object-tracking-on-mot16

A kalman filter is added to use the YOLOv5 bounding boxes detected each frame to register and track vehicles across the screen.
Dense optical flow is implemented to control the kalman filter prediction phase.

YOLOv5 was trained for only 1 epoch on to the MOT16 dataset using 60% of the data for training.
