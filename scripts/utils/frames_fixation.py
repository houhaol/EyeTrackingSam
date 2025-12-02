import pandas as pd
import numpy as np
import os
import cv2
import re
import argparse

def extract_timestamp_ns(filename):
    match = re.search(r"frames_(\d+)\.png", filename)
    return int(match.group(1)) if match else None

def run_sam_by_gaze_or_fixation(frame_dir, output_overlay_dir, gaze_path=None, fixation_path=None):
    os.makedirs(output_overlay_dir, exist_ok=True)

    # Load gaze or fixation data
    gaze_df = None
    fixation_df = None

    if gaze_path:
        gaze_df = pd.read_csv(gaze_path)
        gaze_df = gaze_df[gaze_df["gaze x [px]"].notna() & gaze_df["gaze y [px]"].notna()]
        gaze_df["timestamp_ns"] = gaze_df["timestamp [ns]"].astype(np.int64)

    if fixation_path:
        fixation_df = pd.read_csv(fixation_path)
        fixation_df["start timestamp [ns]"] = fixation_df["start timestamp [ns]"].astype(np.int64)
        fixation_df["end timestamp [ns]"] = fixation_df["end timestamp [ns]"].astype(np.int64)

    # List and filter frames
    all_frame_filenames = sorted([
        fname for fname in os.listdir(frame_dir)
        if re.match(r"frames_\d+\.png", fname)
    ])

    if fixation_df is not None:
        # Collect frames within any fixation interval
        filtered_frames = []
        for fname in all_frame_filenames:
            ts_ns = extract_timestamp_ns(fname)
            in_any_fixation = ((fixation_df["start timestamp [ns]"] <= ts_ns) & (fixation_df["end timestamp [ns]"] >= ts_ns)).any()
            if in_any_fixation:
                filtered_frames.append(fname)
        frame_filenames = filtered_frames
    else:
        frame_filenames = all_frame_filenames

    for fname in frame_filenames:
        timestamp_ns = extract_timestamp_ns(fname)
        if timestamp_ns is None:
            continue

        prompt_source = None
        if fixation_df is not None:
            fixation = fixation_df[
                (fixation_df["start timestamp [ns]"] <= timestamp_ns) &
                (fixation_df["end timestamp [ns]"] >= timestamp_ns)
            ]
            if not fixation.empty:
                fixation = fixation.iloc[0]
                x = fixation["fixation x [px]"]
                y = fixation["fixation y [px]"]
                prompt_source = f"fixation ID {fixation['id']}"
        if gaze_df is not None and prompt_source is None:
            gaze_df["abs_diff"] = np.abs(gaze_df["timestamp_ns"] - timestamp_ns)
            gaze = gaze_df.loc[gaze_df["abs_diff"].idxmin()]
            x = gaze["gaze x [px]"]
            y = gaze["gaze y [px]"]
            prompt_source = "gaze"

        if prompt_source is None:
            print(f"⚠️ Skipping {fname}: no valid gaze or fixation")
            continue

        frame_path = os.path.join(frame_dir, fname)
        image = cv2.imread(frame_path)
        if image is None:
            print(f"⚠️ Could not read {frame_path}")
            continue

        # Create overlay
        overlay = image.copy()
        cv2.circle(overlay, (int(x), int(y)), radius=10, color=(0, 0, 255), thickness=10)

        overlay_filename = os.path.join(output_overlay_dir, f"overlay_{timestamp_ns}.png")
        cv2.imwrite(overlay_filename, overlay)

        print(f"✅ Processed {fname} using {prompt_source}")

    print("🎉 SAM segmentation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM segmentation using gaze or fixation data.")
    parser.add_argument("--frames", required=True, help="Directory of frame images (frames_<timestamp>.png)")
    parser.add_argument("--overlays", default="overlays", help="Directory to save overlay images")
    parser.add_argument("--gaze", help="Path to gaze.csv (optional)")
    parser.add_argument("--fixation", help="Path to fixation.csv (optional)")

    args = parser.parse_args()

    run_sam_by_gaze_or_fixation(
        frame_dir=args.frames,
        output_overlay_dir=args.overlays,
        gaze_path=args.gaze,
        fixation_path=args.fixation,
    )
