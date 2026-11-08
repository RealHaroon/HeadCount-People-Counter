import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("people.mp4")

# --- Counting line -------------------------------------------------------
# A horizontal line across the frame. LINE_POS is a fraction of the height
# (0.5 = middle). If people in your video walk left/right instead of
# up/down, set VERTICAL = True and LINE_POS becomes a fraction of the width.
LINE_POS = 0.5
VERTICAL = False
BAND = 40                     # dead-zone in pixels on each side of the line. People standing
                              # ON the line jitter back and forth; they only change side once
                              # they are more than BAND px past it

DISPLAY_WIDTH = 1280          # video is 1920x1080, shrink the window to fit
MIN_FRAMES = 10               # an ID must be tracked this many frames to count as "seen"
                              # (filters out 1-frame flickers / false detections)
IMG_SIZE = 1280               # YOLO input size. Default 640 misses small/far people;
                              # 1280 finds ~2.5x more here but is slower
OUTPUT = "output.mp4"         # annotated video is saved here (set to None to skip)

last_side = {}                # person_id -> which side of the line they were on last frame
frames_seen = {}              # person_id -> how many frames we have tracked them
counted = set()               # IDs already counted, so one person can't count twice
count_in = 0                  # crossed top->bottom (or left->right)
count_out = 0                 # crossed bottom->top (or right->left)
writer = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    line = int((w if VERTICAL else h) * LINE_POS)

    results = model.track(frame, persist=True, classes=[0], conf=0.4, imgsz=IMG_SIZE,
                          tracker="bytetrack.yaml", verbose=False)

    boxes = results[0].boxes
    if boxes.id is not None:
        ids = boxes.id.int().tolist()
        coords = boxes.xyxy.int().tolist()

        for person_id, (x1, y1, x2, y2) in zip(ids, coords):
            # Use the centre of the box as the person's position
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pos = cx if VERTICAL else cy
            prev = last_side.get(person_id)

            # Which side are they on? False = before the line, True = after it.
            # Inside the dead-zone we keep their previous side (no decision yet).
            if pos > line + BAND:
                side = True
            elif pos < line - BAND:
                side = False
            else:
                side = prev

            # Crossing = they were clearly on one side before and clearly on the other now
            if prev is not None and prev != side and person_id not in counted:
                if side:
                    count_in += 1
                else:
                    count_out += 1
                counted.add(person_id)
            if side is not None:
                last_side[person_id] = side
            frames_seen[person_id] = frames_seen.get(person_id, 0) + 1

            color = (0, 0, 255) if person_id in counted else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, f"ID {person_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Draw the counting line
    if VERTICAL:
        cv2.line(frame, (line, 0), (line, h), (0, 255, 255), 3)
    else:
        cv2.line(frame, (0, line), (w, line), (0, 255, 255), 3)
    # thin lines showing the dead-zone
    for off in (-BAND, BAND):
        if VERTICAL:
            cv2.line(frame, (line + off, 0), (line + off, h), (0, 200, 200), 1)
        else:
            cv2.line(frame, (0, line + off), (w, line + off), (0, 200, 200), 1)

    # Counter panel
    cv2.rectangle(frame, (0, 0), (360, 150), (0, 0, 0), -1)
    cv2.putText(frame, f"IN:  {count_in}", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"OUT: {count_out}", (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    seen = sum(1 for n in frames_seen.values() if n >= MIN_FRAMES)
    cv2.putText(frame, f"Seen: {seen}", (15, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if OUTPUT:
        if writer is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)

    scale = DISPLAY_WIDTH / w
    cv2.imshow("Step 3 - Counter", cv2.resize(frame, None, fx=scale, fy=scale))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

print(f"Final -> IN: {count_in}  OUT: {count_out}  Unique people seen: {seen}")
