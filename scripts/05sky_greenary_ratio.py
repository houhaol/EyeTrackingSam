


import pandas as pd
import numpy as np
import os
import cv2
import re
import argparse
import torch
import h5py
from PIL import Image
import tqdm

def load_h5(h5_path, key='frames'):
    h5_file = h5py.File(h5_path, 'r')
    group = h5_file[key]
    all_keys = list(group.keys())
    single_key = key[:-1] if key.endswith('s') else key
    key_map = {f"{single_key}_{k.split('_')[1]}.png": k for k in all_keys if re.match(rf"{single_key}_\d+", k)}
    return h5_file, group, key_map

def extract_timestamp_ns(filename):
    match = re.search(r"frames_(\d+)\.png", filename)
    return int(match.group(1)) if match else None

def is_h5_file(path):
    return os.path.isfile(path) and path.lower().endswith('.h5')

def greenery_sky_with_semantics(
    frame_h5_path,
    semantic_h5_path,
    id2label_path=None,
    output_csv_path=None,
    demo_mode = False,
    output_dir=None
):
    if id2label_path:
        with open(id2label_path, 'r') as f:
            id2label = {int(line.split(':')[0]): line.split(':')[1].strip() for line in f.readlines()}
    else:
        id2label = None

    # Load semantic maps
    h5_file, frames_group, frames_key_map = load_h5(frame_h5_path)
    seg_files, seg_group, _ = load_h5(semantic_h5_path, key='semantic_maps')
    all_frame_keys = list(seg_group.keys())

    # Map frame keys to timestamps
    frame_key_to_ts = {k: int(re.search(r"frame_(\d+)", k).group(1)) for k in all_frame_keys}

    results = []

    # Define label ids for sky and greenery
    sky_label_id = None
    greenery_label_ids = []
    if id2label:
        for k, v in id2label.items():
            if v.lower() == "sky":
                sky_label_id = k
            if v.lower() in ["vegetation", "terrain"]:
                greenery_label_ids.append(k)
    else:
        # fallback to default ids
        sky_label_id = 27
        greenery_label_ids = [30]

    import random
    random.seed(42)
    palette = {}
    for idx in id2label:
        palette[idx] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

    # Write idx-to-color file if not exists
    idx2color_path = os.path.join(os.path.dirname(id2label_path), 'idx2color.txt')
    if not os.path.exists(idx2color_path):
        with open(idx2color_path, 'w') as f:
            for idx, color in palette.items():
                f.write(f"{idx}: {color[0]},{color[1]},{color[2]}\n")

    if demo_mode:
        n = len(seg_group)
        module = n // 20
        for i in tqdm.tqdm(range(0, n, module)):
            frame_key = list(frames_key_map.keys())[i]
            frame = frames_group[frames_key_map[frame_key]][()]
            
            demo_frame_key = all_frame_keys[i]
            seg = seg_group[demo_frame_key][()]
            total_pixels = seg.size
            sky_pixels = np.sum(seg == sky_label_id)
            greenery_pixels = np.sum(np.isin(seg, greenery_label_ids))
            sky_ratio = sky_pixels / total_pixels
            greenery_ratio = greenery_pixels / total_pixels
            print(f"Frame {i}: Sky ratio = {sky_ratio:.4f}, Greenery ratio = {greenery_ratio:.4f}")
            # Save segmented image
            # Convert BGR to RGB before saving
            alpha = 0.5  # Transparency factor (0: fully transparent, 1: fully opaque)
            overlay = frame.copy()
            mask_sky = seg == sky_label_id
            mask_greenery = np.isin(seg, greenery_label_ids)

            # Create color overlays
            color_sky = np.array([255, 0, 0], dtype=np.uint8)  # Red for sky (RGB)
            color_greenery = np.array([0, 255, 0], dtype=np.uint8)  # Green for greenery (RGB)

            # Blend colors with original frame
            overlay[mask_sky] = (alpha * color_sky + (1 - alpha) * overlay[mask_sky]).astype(np.uint8)
            overlay[mask_greenery] = (alpha * color_greenery + (1 - alpha) * overlay[mask_greenery]).astype(np.uint8)

            overlay_rgb = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB)
            seg_img = Image.fromarray(overlay_rgb)
            seg_img.save(os.path.join(output_dir, f"segmented_frame_{i}.png"))   
            print("Segmented images saved.")   # Save to CSV
        results.append({
            "frame": 10,
            "sky_ratio": sky_ratio,
            "greenery_ratio": greenery_ratio
        })
    else:
        # Process all frames
        for k in tqdm(all_frame_keys, desc="Processing frames"):
            ts = frame_key_to_ts[k]
            seg = seg_group[k][()]
            total_pixels = seg.size
            sky_pixels = np.sum(seg == sky_label_id)
            greenery_pixels = np.sum(np.isin(seg, greenery_label_ids))
            sky_ratio = sky_pixels / total_pixels
            greenery_ratio = greenery_pixels / total_pixels
            results.append({
                "frame": ts,
                "sky_ratio": sky_ratio,
                "greenery_ratio": greenery_ratio
            })

        # Output results
        results_df = pd.DataFrame(results)
        if output_csv_path:
            results_df.to_csv(output_csv_path, index=False)
            print(f"Ratios saved to {output_csv_path}")
        else:
            print(results_df)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute greenery and sky ratio with semantic labels using Mask2Former outputs.")
    parser.add_argument("--frame_h5", required=True, help="Path to frames HDF5 file")
    parser.add_argument("--semantic_h5", required=True, help="Path to semantic maps HDF5 file (from Mask2Former)")
    parser.add_argument("--id2label", help="Path to id2label.txt file")
    parser.add_argument("--output_dir", help="Directory to save segmented images in demo mode")
    parser.add_argument("--output_csv", help="Path to output CSV file for greenery and sky ratio")

    args = parser.parse_args()
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    greenery_sky_with_semantics(
        frame_h5_path=args.frame_h5,
        semantic_h5_path=args.semantic_h5,
        output_dir=args.output_dir,
        output_csv_path=args.output_csv,
        id2label_path=args.id2label,
        demo_mode=True
    )
