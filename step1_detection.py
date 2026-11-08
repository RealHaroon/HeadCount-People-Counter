import cv2
from ultralytics import YOLO

model=YOLO('yolo11n.pt')
cap=cv2.VideoCapture('people.mp4')

while True:
    ok, frame=cap.read()
    if not ok:
        break

    results=model(frame,classes=[0],conf=0.4,verbose=False)

    annotated=results[0].plot()
    cv2.imshow('Step 1 - Detection',annotated)

    if cv2.waitkey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindow()
