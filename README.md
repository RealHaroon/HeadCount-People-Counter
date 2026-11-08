# HeadCount – People Counter

Counts people in video using **YOLO11** (detection), **ByteTrack** (tracking IDs) and **OpenCV** (drawing / video I/O).

- **Line counting** – IN / OUT when people cross a line, plus live occupancy (IN − OUT)
- **Zone counting** – how many people are inside a polygon right now
- Video file, webcam or RTSP stream input
- Saves annotated video and a CSV log of every crossing

## Setup

```bash
pip install -r requirements.txt
# CPU-only machines: install the small CPU build of torch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The YOLO weights (`yolo11n.pt`) download automatically on first run.

## Usage

```bash
python main.py                                   # people.mp4, horizontal line in the middle
python main.py --source 0                        # webcam
python main.py --vertical --line 0.4             # vertical line at 40% of the width
python main.py --zone "0.1,0.35 0.35,0.35 0.35,0.65 0.1,0.65"   # polygon, fractions of the frame
python main.py --no-show --output out.mp4 --log events.csv      # headless, save results
python main.py -h                                # all options
```

Press **q** to quit.

| Box color | Meaning |
|---|---|
| Green | tracked, not counted yet |
| Red | already counted crossing the line |
| Blue | inside the zone |

## How it works

1. **Detect** – YOLO finds people in each frame (`step1_detection.py`).
2. **Track** – ByteTrack gives each person a stable ID across frames (`step2_tracking.py`).
3. **Count** – for each ID we remember which side of the line its center is on. When it switches side, that's one crossing (`step3_counter.py`). Each ID counts once.

Tuning notes:
- `--imgsz 1280` (default) finds far / small people that the YOLO default of 640 misses, at the cost of speed.
- `--band` is a dead-zone around the line: people standing *on* the line jitter back and forth, so a side change only counts once they're clearly past it.
- "Seen" only counts IDs tracked for `--min-frames` frames, to ignore flicker detections. It still over-counts a bit when the tracker loses someone and gives them a new ID – the line counts are more reliable.

## Troubleshooting

**"could not read a frame"** – usually an AV1-encoded video, which pip's OpenCV can't decode. Re-encode it:
```bash
ffmpeg -i input.mp4 -c:v libx264 fixed.mp4
```
