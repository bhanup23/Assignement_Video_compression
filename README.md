# Smart Behavioral Video Compression
**Sentio Mind · POC Assignment · Project 2**

GitHub: https://github.com/Sentiodirector/Assignement_Video_compression.git
Branch: FirstName_LastName_RollNumber

---

## Why This Exists

Four cameras running all day in a school building produce 40 to 80 GB of raw footage. Uploading that to the Sentio Mind server over a typical school internet connection takes 6 to 12 hours. That is not practical.

Blindly compressing with ffmpeg throws away frames that contain people, which breaks the analysis. Your job is to build a smarter compressor — one that keeps every frame containing a human and aggressively discards empty hallway footage and near-duplicate frames.

---

## What You Receive

```
p2_video_compression/
├── video_sample_1.mov          ← 2-3 min raw CCTV clip, download from dataset link
├── video_compression.py        ← your template — copy to solution.py
├── video_compression.json      ← schema for segments_kept.json
└── README.md
```

---

## What You Must Build

Run `python solution.py` → it must produce:

1. `compressed_output.mp4` — H.264, 12 fps, at least 70% smaller than input
2. `compression_report.html` — size comparison, duration comparison, thumbnail storyboard
3. `segments_kept.json` — follows `video_compression.json` schema exactly

### Decision Algorithm (implement in this exact order)

```
For each frame:

Step 1 — pHash similarity
  Compute perceptual hash of this frame.
  If similarity to last kept frame > 0.95 → discard (near-duplicate).

Step 2 — Motion score
  Compute dense optical flow vs previous frame.
  If motion_score < 0.05 → mark as discard candidate (static empty scene).

Step 3 — Face override
  Run Haar face detection.
  If any face found → keep this frame regardless of steps 1 and 2.

Step 4 — Motion override
  If no face found but motion_score > 0.15 → keep anyway.

Step 5 — Context frame rule
  Every 3 seconds of original video → force-keep one frame no matter what.
```

Then re-encode all kept frames to H.264 MP4 at 12 fps using ffmpeg.

### Performance Targets

- File size reduction: 70% or more
- Processing speed: 2-minute video must finish in 10 seconds or less on a laptop

---

## Hard Rules

- Do not rename functions in `video_compression.py`
- Do not change key names in `video_compression.json`
- Output video must play in VLC without codec issues
- `compression_report.html` must work offline
- Python 3.9+, no Jupyter notebooks
- ffmpeg must be installed: `sudo apt install ffmpeg`

## Libraries

```
opencv-python==4.9.0   numpy==1.26.4   imagehash==4.3.1   Pillow==10.3.0
```

---

## Submit

| # | File | What |
|---|------|------|
| 1 | `solution.py` | Working script |
| 2 | `compressed_output.mp4` | Compressed video |
| 3 | `compression_report.html` | Report with storyboard |
| 4 | `segments_kept.json` | Segment log matching schema |
| 5 | `demo.mp4` | Screen recording under 2 min |

Push to your branch only. Do not touch main.

---

## Bonus

Auto-calibrate the motion threshold from the first 30 seconds of the video. Different cameras at different lighting levels need different thresholds — hardcoding 0.05 for every camera is fragile.

*Sentio Mind · 2026*
