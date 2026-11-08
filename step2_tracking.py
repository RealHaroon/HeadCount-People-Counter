import cv2
from ultralytics import YOLO

model =YOLO('yolo11n.pt')
cap=cv2.VideoCapture('people.mp4')

While True: