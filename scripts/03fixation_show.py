import h5py
import pandas as pd
import numpy as np
import random
import cv2
import os

# Paths (update as needed)
h5_path = "/home/houhao/workspace/EyeTrackingSam/data/pilot2/frames/frames.h5"
csv_path = "/home/houhao/workspace/EyeTrackingSam/data/pilot2/fixations_objects.csv"

# Load frames
frames = h5py.File(h5_path, 'r')['frames']
frame_keys = list(frames.keys())

# Load fixation annotations
df = pd.read_csv(csv_path)

# Randomly sample 10 frames
sampled = random.sample(frame_keys, 10)

# Output directory
output_dir = "/home/houhao/workspace/EyeTrackingSam/data/pilot2/output_fixation_debug"
os.makedirs(output_dir, exist_ok=True)

for frame_key in sampled:
    frame = frames[frame_key][()]
    timestamp = int(frame_key.split('_')[1])  # Assuming frame_key format is like 'frame_123456'

    # Find which fixation corresponds to this frame
    try:
        row = df[(df['start_ts'] <= timestamp) & (df['end_ts'] >= timestamp)].iloc[0]
    except:
        print(f"No fixation found for frame {frame_key} at timestamp {timestamp}")
        continue
    # Annotate frame
    x, y = int(row['fixation_x']), int(row['fixation_y'])
    label = row['voted_label_name']
    frame_disp = frame.copy()
    # Draw a larger red circle at fixation
    cv2.circle(frame_disp, (x, y), 25, (0, 0, 255), -1)

    # Put text at the bottom middle, increased font size, cyan color
    text = f"Label: {label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (frame_disp.shape[1] - text_size[0]) // 2
    text_y = frame_disp.shape[0] -50
    cv2.putText(frame_disp, text, (text_x, text_y), font, font_scale, (255, 255, 0), thickness)

    # Save annotated frame
    out_path = os.path.join(output_dir, f"{frame_key}_fixation_{x}_{y}_label_{label}.png")
    cv2.imwrite(out_path, frame_disp)
    print(f"Saved: {out_path} | Object: {label}, Fixation: ({x}, {y})")