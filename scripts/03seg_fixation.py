import pandas as pd
import numpy as np
import os
import cv2
import re
import argparse
import torch
import h5py
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm  # <-- add tqdm import

def extract_timestamp_ns(filename):
    match = re.search(r"frames_(\d+)\.png", filename)
    return int(match.group(1)) if match else None

def is_h5_file(path):
    return os.path.isfile(path) and path.lower().endswith('.h5')

def annotate_fixations_with_semantics(
    semantic_h5_path,
    fixation_path,
    id2label_path=None,
    output_csv_path=None
):
    

    if id2label_path:
        with open(id2label_path, 'r') as f:
            id2label = {int(line.split(':')[0]): line.split(':')[1].strip() for line in f.readlines()}
    else:
        id2label = None
    fixation_df = pd.read_csv(fixation_path)
    fixation_df["start timestamp [ns]"] = fixation_df["start timestamp [ns]"].astype(np.int64)
    fixation_df["end timestamp [ns]"] = fixation_df["end timestamp [ns]"].astype(np.int64)

    # Load semantic maps
    semantic_h5 = h5py.File(semantic_h5_path, "r")
    seg_group = semantic_h5["semantic_maps"]
    all_frame_keys = list(seg_group.keys())

    # Map frame keys to timestamps
    frame_key_to_ts = {k: int(re.search(r"frame_(\d+)", k).group(1)) for k in all_frame_keys}

    results = []
    for _, fixation in tqdm(fixation_df.iterrows(), total=len(fixation_df), desc="Annotating fixations"):
        start_ts = fixation["start timestamp [ns]"]
        end_ts = fixation["end timestamp [ns]"]
        x = int(fixation["fixation x [px]"])
        y = int(fixation["fixation y [px]"])
        fixation_id = fixation["fixation id"] if "fixation id" in fixation else None

        # Find frames in this fixation
        relevant_keys = [k for k, ts in frame_key_to_ts.items() if start_ts <= ts <= end_ts]
        if not relevant_keys:
            continue
        labels = []
        for k in relevant_keys:
            seg = seg_group[k][()]
            # Check bounds
            if 0 <= y < seg.shape[0] and 0 <= x < seg.shape[1]:
                label = seg[y, x]
                labels.append(label)
        if labels:
            # Vote for most common label
            voted_label = int(pd.Series(labels).mode()[0])
        else:
            voted_label = None

        if id2label is not None:
            voted_label_name = id2label.get(voted_label, "Unknown")
            results.append({
                "fixation_id": fixation_id,
                "fixation_x": x,
                "fixation_y": y,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "voted_label": voted_label,
                "voted_label_name": voted_label_name
            })
        else:
            # If id2label is not provided, just store the label ID
            results.append({
                "fixation_id": fixation_id,
                "fixation_x": x,
                "fixation_y": y,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "voted_label": voted_label
            })

    semantic_h5.close()

    # Output results
    results_df = pd.DataFrame(results)
    if output_csv_path:
        results_df.to_csv(output_csv_path, index=False)
        print(f"Fixation annotations saved to {output_csv_path}")
    else:
        print(results_df)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate fixations with semantic labels using Mask2Former outputs.")
    parser.add_argument("--semantic_h5", required=True, help="Path to semantic maps HDF5 file (from Mask2Former)")
    parser.add_argument("--fixation", required=True, help="Path to fixation.csv")
    parser.add_argument("--id2label", help="Path to id2label.txt file")
    parser.add_argument("--output_csv", help="Path to output CSV for fixation annotations")

    args = parser.parse_args()

    annotate_fixations_with_semantics(
        semantic_h5_path=args.semantic_h5,
        fixation_path=args.fixation,
        output_csv_path=args.output_csv,
        id2label_path=args.id2label
    )
