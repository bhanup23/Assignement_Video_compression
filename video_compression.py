"""
video_compression.py
Sentio Mind · Project 2 · Smart Behavioral Video Compression

Copy this file to solution.py and fill in every TODO block.
Do not rename any function.
Run: python solution.py
Requires ffmpeg installed on your system: sudo apt install ffmpeg
"""

import cv2
import json
import base64
import subprocess
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_IN               = Path("video_sample_1.mov")
VIDEO_OUT              = Path("compressed_output.mp4")
REPORT_HTML_OUT        = Path("compression_report.html")
SEGMENTS_JSON_OUT      = Path("segments_kept.json")

PHASH_THRESHOLD        = 0.95   # similarity above this = near-duplicate, discard
MOTION_KEEP_THRESH     = 0.15   # keep frame if motion exceeds this (no face needed)
MOTION_DISCARD_THRESH  = 0.05   # definitely discard below this
CONTEXT_EVERY_SEC      = 3      # force-keep one frame every this many seconds
OUTPUT_FPS             = 12     # frame rate of the output video
OUTPUT_CRF             = 28     # ffmpeg quality: lower = better quality + larger file


# ---------------------------------------------------------------------------
# PERCEPTUAL HASH
# ---------------------------------------------------------------------------

def compute_phash(frame: np.ndarray) -> str:
    """
    Compute a perceptual hash of the frame.
    Steps: resize to 32×32 grayscale → DCT → threshold at mean → flatten to bit string.
    Return a string of '0' and '1' characters, length 64.

    You can use the imagehash library (imagehash.phash) or implement manually.
    TODO: implement
    """
    # TODO
    return "0" * 64


def phash_similarity(h1: str, h2: str) -> float:
    """
    Compare two hash strings. Return 1.0 if identical, 0.0 if completely different.
    Formula: 1.0 - (hamming_distance / length)
    TODO: implement
    """
    if not h1 or not h2 or len(h1) != len(h2):
        return 0.0
    # TODO
    return 0.0


# ---------------------------------------------------------------------------
# MOTION SCORE
# ---------------------------------------------------------------------------

def compute_motion_score(prev_gray, curr_gray: np.ndarray) -> float:
    """
    Dense optical flow between two grayscale frames. Return mean magnitude, ~0.0-1.0.
    If prev_gray is None, return 0.0.
    TODO: cv2.calcOpticalFlowFarneback
    Params: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2
    """
    if prev_gray is None:
        return 0.0
    # TODO
    return 0.0


# ---------------------------------------------------------------------------
# FACE PRESENCE CHECK
# ---------------------------------------------------------------------------

def has_face(frame: np.ndarray, cascade) -> bool:
    """
    True if at least one face detected. Use the Haar cascade passed in.
    Equalise histogram on grayscale first for better CCTV detection.
    TODO: cascade.detectMultiScale — scaleFactor=1.1, minNeighbors=3, minSize=(20,20)
    """
    # TODO
    return False


# ---------------------------------------------------------------------------
# FRAME KEEP DECISION
# ---------------------------------------------------------------------------

def should_keep_frame(frame: np.ndarray,
                      prev_frame,
                      prev_kept_hash: str,
                      last_kept_time_sec: float,
                      current_time_sec: float,
                      cascade) -> tuple:
    """
    Apply the 5-step decision algorithm from README in order.
    Return: (keep: bool, reason: str, motion_score: float, face_found: bool)

    Reason strings (use exactly these):
      'face_detected', 'motion_above_threshold', 'context_frame',
      'face_and_motion', 'discarded_duplicate', 'discarded_static'

    TODO: implement
    """
    # TODO
    return False, "discarded_static", 0.0, False


# ---------------------------------------------------------------------------
# THUMBNAIL HELPER
# ---------------------------------------------------------------------------

def frame_to_b64_thumb(frame: np.ndarray, width: int = 200) -> str:
    """Resize frame keeping aspect ratio, encode as base64 JPEG."""
    h, w = frame.shape[:2]
    nh = int(h * width / w)
    thumb = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# VIDEO WRITING
# ---------------------------------------------------------------------------

def write_frames_to_video(kept_frames: list, output_path: Path,
                          fps: float, frame_size: tuple):
    """
    Write kept_frames to a temporary file, then re-encode with ffmpeg to H.264 MP4.

    Steps:
      1. Write to temp_raw.avi using cv2.VideoWriter (mp4v codec)
      2. Call ffmpeg: ffmpeg -y -i temp_raw.avi -vcodec libx264 -crf CRF -preset fast out.mp4
      3. Delete temp_raw.avi

    TODO: implement
    """
    # TODO
    pass


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_compression_report(segments: list, stats: dict, output_path: Path):
    """
    Write a self-contained HTML file showing:
      - Original vs compressed size (MB and % reduction)
      - Original vs compressed duration (seconds)
      - Processing time
      - Storyboard grid: one thumbnail per segment
      - Frames kept vs discarded count

    No CDN. Inline CSS only. Must work offline.
    TODO: implement
    """
    # TODO
    pass


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap          = cv2.VideoCapture(str(VIDEO_IN))
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total / fps_in
    orig_mb      = VIDEO_IN.stat().st_size / 1_000_000

    print(f"Input: {VIDEO_IN}  |  {total} frames  |  {duration:.1f}s  |  {orig_mb:.1f} MB")

    kept_frames = []
    segments    = []
    prev_frame  = None
    prev_gray   = None
    prev_hash   = ""
    last_kept_t = -999.0
    cur_seg     = None
    disc_dup    = 0
    disc_stat   = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps_in

        keep, reason, motion, face = should_keep_frame(
            frame, prev_frame, prev_hash, last_kept_t, ts, cascade
        )

        if keep:
            kept_frames.append(frame.copy())
            prev_hash   = compute_phash(frame)
            last_kept_t = ts

            if cur_seg is None or (ts - cur_seg["end_sec"]) > 2.5:
                if cur_seg:
                    segments.append(cur_seg)
                cur_seg = {
                    "segment_id":            len(segments) + 1,
                    "start_sec":             round(ts, 2),
                    "end_sec":               round(ts, 2),
                    "frames_in_segment":     1,
                    "reason_kept":           reason,
                    "face_count_in_segment": 1 if face else 0,
                    "motion_score_avg":      round(motion, 3),
                    "thumbnail_b64":         frame_to_b64_thumb(frame),
                }
            else:
                cur_seg["end_sec"]               = round(ts, 2)
                cur_seg["frames_in_segment"]    += 1
                cur_seg["face_count_in_segment"] += 1 if face else 0
        else:
            if "duplicate" in reason:
                disc_dup  += 1
            else:
                disc_stat += 1

        prev_frame = frame
        prev_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_idx += 1

    if cur_seg:
        segments.append(cur_seg)
    cap.release()

    print(f"Kept {len(kept_frames)} / {total} frames across {len(segments)} segments")
    print("Writing compressed video ...")
    write_frames_to_video(kept_frames, VIDEO_OUT, OUTPUT_FPS, (fw, fh))

    comp_mb = VIDEO_OUT.stat().st_size / 1_000_000 if VIDEO_OUT.exists() else 0.0
    t_end   = time.time()

    stats = {
        "source_video":             str(VIDEO_IN),
        "compressed_video":         str(VIDEO_OUT),
        "original_size_mb":         round(orig_mb, 2),
        "compressed_size_mb":       round(comp_mb, 2),
        "reduction_pct":            round((1 - comp_mb / (orig_mb + 1e-9)) * 100, 1),
        "original_duration_sec":    round(duration, 2),
        "compressed_duration_sec":  round(len(kept_frames) / OUTPUT_FPS, 2),
        "original_fps":             round(fps_in, 2),
        "output_fps":               OUTPUT_FPS,
        "frames_original":          total,
        "frames_kept":              len(kept_frames),
        "processing_time_sec":      round(t_end - t_start, 2),
        "segments":                 segments,
        "frames_discarded_reasons": {
            "near_duplicate_phash": disc_dup,
            "low_motion_no_face":   disc_stat,
            "total_discarded":      total - len(kept_frames),
        },
    }

    with open(SEGMENTS_JSON_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    generate_compression_report(segments, stats, REPORT_HTML_OUT)

    print()
    print("=" * 55)
    print(f"  Done in {stats['processing_time_sec']}s")
    print(f"  Size:     {orig_mb:.1f} MB  →  {comp_mb:.1f} MB  ({stats['reduction_pct']}% smaller)")
    print(f"  Duration: {duration:.1f}s  →  {stats['compressed_duration_sec']:.1f}s")
    print(f"  Report  → {REPORT_HTML_OUT}")
    print(f"  JSON    → {SEGMENTS_JSON_OUT}")
    print("=" * 55)
