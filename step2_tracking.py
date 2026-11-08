import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("people.mp4")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # track() instead of model() -> adds IDs. persist=True remembers people between frames
    results = model.track(frame, persist=True, classes=[0], conf=0.4,
                          tracker="bytetrack.yaml", verbose=False)

    boxes = results[0].boxes
    if boxes.id is not None:                     # sometimes no one is tracked
        ids = boxes.id.int().tolist()
        coords = boxes.xyxy.int().tolist()       # [x1, y1, x2, y2] for each box

        for person_id, (x1, y1, x2, y2) in zip(ids, coords):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {person_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Step 2 - Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()