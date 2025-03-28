import pandas as pd
import numpy as np
import os
import cv2
import re
import argparse
from segment_anything import sam_model_registry, SamPredictor

def extract_timestamp_ns(filename):
    match = re.search(r"frames_(\d+)\.png", filename)
    return int(match.group(1)) if match else None

def find_closest_gaze_point(timestamp_ns, gaze_df):
    gaze_df["abs_diff"] = np.abs(gaze_df["timestamp_ns"] - timestamp_ns)
    return gaze_df.loc[gaze_df["abs_diff"].idxmin()]

def find_fixation_for_frame(timestamp_ns, fixation_df):
    match = fixation_df[
        (fixation_df["start timestamp [ns]"] <= timestamp_ns) &
        (fixation_df["end timestamp [ns]"] >= timestamp_ns)
    ]
    return match.iloc[0] if not match.empty else None

def run_sam_by_gaze_or_fixation(frame_dir, checkpoint_path, output_mask_dir, output_overlay_dir, gaze_path=None, fixation_path=None):
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_overlay_dir, exist_ok=True)

    # Load SAM model
    model_type = "_".join(os.path.basename(checkpoint_path).split('_')[1:3])
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)

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

    # List frames
    frame_filenames = sorted([
        fname for fname in os.listdir(frame_dir)
        if re.match(r"frames_\d+\.png", fname)
    ])

    for fname in frame_filenames:
        timestamp_ns = extract_timestamp_ns(fname)
        if timestamp_ns is None:
            continue

        # Determine prompt point (x, y)
        prompt_source = None
        if fixation_df is not None:
            fixation = find_fixation_for_frame(timestamp_ns, fixation_df)
            if fixation is not None:
                x = fixation["fixation x [px]"]
                y = fixation["fixation y [px]"]
                prompt_source = f"fixation ID {fixation['id']}"
        if gaze_df is not None and prompt_source is None:
            gaze = find_closest_gaze_point(timestamp_ns, gaze_df)
            x = gaze["gaze x [px]"]
            y = gaze["gaze y [px]"]
            prompt_source = "gaze"

        if prompt_source is None:
            print(f"⚠️ Skipping {fname}: no valid gaze or fixation")
            continue

        input_point = np.array([[x, y]])
        input_label = np.array([1])  # foreground

        frame_path = os.path.join(frame_dir, fname)
        image = cv2.imread(frame_path)
        if image is None:
            print(f"⚠️ Could not read {frame_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        best_mask = masks[np.argmax(scores)]

        # Save mask image
        mask_result = image.copy()
        mask_result[best_mask] = [0, 255, 0]
        mask_filename = os.path.join(output_mask_dir, f"mask_{timestamp_ns}.png")
        cv2.imwrite(mask_filename, mask_result)

        # Create overlay
        overlay = image.copy()
        overlay_mask = np.zeros_like(overlay, dtype=np.uint8)
        overlay_mask[best_mask] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 1.0, overlay_mask, 0.5, 0)
        cv2.circle(overlay, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

        overlay_filename = os.path.join(output_overlay_dir, f"overlay_{timestamp_ns}.png")
        cv2.imwrite(overlay_filename, overlay)

        print(f"✅ Processed {fname} using {prompt_source}")

    print("🎉 SAM segmentation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM segmentation using gaze or fixation data.")
    parser.add_argument("--frames", required=True, help="Directory of frame images (frames_<timestamp>.png)")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint (e.g., sam_vit_b.pth)")
    parser.add_argument("--masks", default="masks", help="Directory to save mask outputs")
    parser.add_argument("--overlays", default="overlays", help="Directory to save overlay images")
    parser.add_argument("--gaze", help="Path to gaze.csv (optional)")
    parser.add_argument("--fixation", help="Path to fixation.csv (optional)")

    args = parser.parse_args()

    run_sam_by_gaze_or_fixation(
        frame_dir=args.frames,
        checkpoint_path=args.checkpoint,
        output_mask_dir=args.masks,
        output_overlay_dir=args.overlays,
        gaze_path=args.gaze,
        fixation_path=args.fixation,
    )
