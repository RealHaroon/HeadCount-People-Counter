"""HeadCount - people counter with YOLO + ByteTrack + OpenCV.

Everything from step1-3 in one script, plus:
  * live occupancy (IN - OUT)
  * optional polygon zone that counts how many people are inside it right now
  * webcam / RTSP / video file input
  * CSV log of every line crossing

Examples:
  python main.py                                  # people.mp4, horizontal line in the middle
  python main.py --source 0                       # webcam
  python main.py --source rtsp://user:pass@ip/... # IP camera
  python main.py --line 0.6 --band 30             # move the line lower, smaller dead-zone
  python main.py --vertical                       # people walk left/right
  python main.py --zone "0.1,0.2 0.5,0.2 0.5,0.8 0.1,0.8"   # zone as fractions of the frame
  python main.py --no-show --output out.mp4 --log events.csv  # headless, save results
"""
import argparse
import csv
import time

import cv2
import numpy as np
from ultralytics import YOLO

GREEN, RED, YELLOW, WHITE, BLACK, BLUE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0), (255, 128, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    p = argparse.ArgumentParser(description="Count people crossing a line / inside a zone.")
    p.add_argument("--source", default="people.mp4", help="video file, webcam index (0) or RTSP/HTTP URL")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights (yolo11s.pt is more accurate, slower)")
    p.add_argument("--imgsz", type=int, default=1280, help="YOLO input size; 640 is faster but misses small people")
    p.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold")
    p.add_argument("--line", type=float, default=0.5, help="line position as a fraction of height (or width with --vertical)")
    p.add_argument("--vertical", action="store_true", help="use a vertical line (people walk left/right)")
    p.add_argument("--band", type=int, default=40, help="dead-zone in px around the line to ignore jitter")
    p.add_argument("--no-line", action="store_true", help="disable line counting (e.g. zone only)")
    p.add_argument("--zone", default=None, help='polygon as fractions "x,y x,y x,y ..." (at least 3 points)')
    p.add_argument("--min-frames", type=int, default=10, help="frames an ID must be tracked to count as 'seen'")
    p.add_argument("--output", default=None, help="save annotated video to this path")
    p.add_argument("--log", default=None, help="save crossing events to this CSV path")
    p.add_argument("--no-show", action="store_true", help="don't open a window (servers / headless)")
    p.add_argument("--display-width", type=int, default=1280, help="window width")
    return p.parse_args()


def open_source(src):
    # "0", "1" ... -> webcam index, anything else -> file path / stream URL
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")
    ok, frame = cap.read()
    if not ok:
        # Usually a codec problem, e.g. AV1 videos that pip's OpenCV can't decode.
        raise SystemExit(f"Opened {src} but could not read a frame. If it's a file, try re-encoding: "
                         f"ffmpeg -i {src} -c:v libx264 fixed.mp4")
    return cap, frame


def parse_zone(text, w, h):
    pts = [tuple(map(float, pair.split(","))) for pair in text.split()]
    if len(pts) < 3:
        raise SystemExit("--zone needs at least 3 points")
    return np.array([(int(x * w), int(y * h)) for x, y in pts], dtype=np.int32)


class LineCounter:
    """Counts each track ID once when it moves from one side of the line to the other."""

    def __init__(self, pos, vertical, band):
        self.pos, self.vertical, self.band = pos, vertical, band
        self.last_side = {}      # id -> False (before line) / True (after line)
        self.counted = set()
        self.count_in = 0        # top->bottom (or left->right)
        self.count_out = 0       # bottom->top (or right->left)

    def update(self, pid, cx, cy):
        """Returns 'IN', 'OUT' or None."""
        p = cx if self.vertical else cy
        prev = self.last_side.get(pid)
        if p > self.pos + self.band:
            side = True
        elif p < self.pos - self.band:
            side = False
        else:
            side = prev                      # inside the dead-zone: keep previous decision
        if side is None:
            return None
        self.last_side[pid] = side
        if prev is not None and prev != side and pid not in self.counted:
            self.counted.add(pid)
            if side:
                self.count_in += 1
                return "IN"
            self.count_out += 1
            return "OUT"
        return None

    def draw(self, frame):
        h, w = frame.shape[:2]
        for off, color, thick in ((0, YELLOW, 3), (-self.band, (0, 200, 200), 1), (self.band, (0, 200, 200), 1)):
            v = self.pos + off
            a, b = ((v, 0), (v, h)) if self.vertical else ((0, v), (w, v))
            cv2.line(frame, a, b, color, thick)


def draw_panel(frame, rows):
    """Black box top-left with one line of text per (label, color) row."""
    height = 20 + 45 * len(rows)
    cv2.rectangle(frame, (0, 0), (400, height), BLACK, -1)
    for i, (text, color) in enumerate(rows):
        cv2.putText(frame, text, (15, 45 + 45 * i), FONT, 1.1, color, 3)


def main():
    args = parse_args()
    model = YOLO(args.model)
    cap, frame = open_source(args.source)
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    line = None
    if not args.no_line:
        line = LineCounter(int((w if args.vertical else h) * args.line), args.vertical, args.band)
    zone = parse_zone(args.zone, w, h) if args.zone else None

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if args.output else None
    log_file = open(args.log, "w", newline="") if args.log else None
    log = csv.writer(log_file) if log_file else None
    if log:
        log.writerow(["frame", "video_time_s", "wall_time", "person_id", "direction"])

    frames_seen = {}
    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            results = model.track(frame, persist=True, classes=[0], conf=args.conf, imgsz=args.imgsz,
                                  tracker="bytetrack.yaml", verbose=False)
            boxes = results[0].boxes
            in_zone = 0

            if boxes.id is not None:
                for pid, (x1, y1, x2, y2) in zip(boxes.id.int().tolist(), boxes.xyxy.int().tolist()):
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    frames_seen[pid] = frames_seen.get(pid, 0) + 1

                    if line:
                        event = line.update(pid, cx, cy)
                        if event and log:
                            log.writerow([frame_idx, round(frame_idx / fps, 2),
                                          time.strftime("%Y-%m-%d %H:%M:%S"), pid, event])

                    inside = zone is not None and cv2.pointPolygonTest(zone, (cx, cy), False) >= 0
                    in_zone += inside

                    color = BLUE if inside else (RED if line and pid in line.counted else GREEN)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
                    cv2.putText(frame, f"ID {pid}", (x1, y1 - 8), FONT, 0.8, color, 2)

            # --- overlay ---------------------------------------------------
            rows = []
            if line:
                line.draw(frame)
                rows += [(f"IN:  {line.count_in}", GREEN), (f"OUT: {line.count_out}", RED),
                         (f"Occupancy: {line.count_in - line.count_out}", YELLOW)]
            if zone is not None:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [zone], BLUE)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.polylines(frame, [zone], True, BLUE, 3)
                rows.append((f"In zone: {in_zone}", BLUE))
            seen = sum(1 for n in frames_seen.values() if n >= args.min_frames)
            rows.append((f"Seen: {seen}", WHITE))
            proc_fps = (frame_idx + 1) / max(time.time() - t_start, 1e-6)
            rows.append((f"FPS: {proc_fps:.1f}", WHITE))
            draw_panel(frame, rows)

            if writer:
                writer.write(frame)
            if not args.no_show:
                scale = args.display_width / w
                cv2.imshow("HeadCount", cv2.resize(frame, None, fx=scale, fy=scale))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                break
    finally:
        # runs even on Ctrl+C, so the output video / CSV aren't left corrupted
        cap.release()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        cv2.destroyAllWindows()

    summary = f"Frames: {frame_idx}  Seen: {seen}"
    if line:
        summary = (f"IN: {line.count_in}  OUT: {line.count_out}  "
                   f"Occupancy: {line.count_in - line.count_out}  " + summary)
    print(summary)


if __name__ == "__main__":
    main()
