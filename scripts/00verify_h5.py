import random
import h5py
import numpy as np
import json
from pycocotools import mask as maskUtils
import cv2
import os
import tqdm
import re
import pandas as pd

def load_h5(h5_path, key='frames'):
    h5_file = h5py.File(h5_path, 'r')
    group = h5_file[key]
    all_keys = list(group.keys())
    single_key = key[:-1] if key.endswith('s') else key
    key_map = {f"{single_key}_{k.split('_')[1]}.png": k for k in all_keys if re.match(rf"{single_key}_\d+", k)}
    return h5_file, group, key_map

def inspect_raw_frames(frames_h5_path, debug_dir):
    h5_file, frames_group, frames_key_map = load_h5(frames_h5_path)
    # sample 10 images from the frames and save for debug purpose
    n = len(frames_group)
    module = n // 10
    for i in tqdm.tqdm(range(0, n, module)):
        frame_key = list(frames_key_map.keys())[i]
        frame = frames_group[frames_key_map[frame_key]][()]
        
        # Save the frame as an image file
        cv2.imwrite(f"{debug_dir}/debug_frame_{i}.png",frame)
    print("Finished saving debug frames.")
    return


def inspect_frames_with_masks(frames_h5_path, masks_h5_path,
                              id2label_path, debug_dir):
    h5_file, frames_group, frames_key_map = load_h5(frames_h5_path)
    masks_file, masks_group, _ = load_h5(masks_h5_path, key='semantic_maps')

    # Parse id2label.txt
    with open(id2label_path, 'r') as f:
        lines = f.readlines()
    idx2label = {}
    for line in lines:
        line = line.strip()
        if line:
            idx, label = line.split(':', 1)
            idx2label[int(idx.strip())] = label.strip()

    # Create a palette (using random colors for each class)
    import random
    random.seed(42)
    palette = {}
    for idx in idx2label:
        palette[idx] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

    # Write idx-to-color file if not exists
    idx2color_path = os.path.join(os.path.dirname(id2label_path), 'idx2color.txt')
    if not os.path.exists(idx2color_path):
        with open(idx2color_path, 'w') as f:
            for idx, color in palette.items():
                f.write(f"{idx}: {color[0]},{color[1]},{color[2]}\n")

    # sample 10 images from the frames and save for debug purpose
    n = len(frames_group)
    module = n // 10
    for i in tqdm.tqdm(range(0, n, module)):
        frame_key = list(frames_key_map.keys())[i]
        frame = frames_group[frames_key_map[frame_key]][()]
        mask_key = frame_key.split('.png')[0]
        mask = masks_group[mask_key][()]

        # Overlay mask colors on frame
        overlay = frame.copy()
        legend_height = 25 * 8  # up to 8 classes per image, adjust as needed
        legend_width = frame.shape[1]  # match frame width
        if mask.ndim == 2:
            mask_vis = np.zeros_like(frame)
            for idx, color in palette.items():
                mask_vis[mask == idx] = color
            # Blend overlay
            overlay = cv2.addWeighted(frame, 0.5, mask_vis, 0.5, 0)

            # Add legend for present classes
            present_classes = np.unique(mask)
            legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            y0 = 20
            for j, idx in enumerate(present_classes):
                if idx in palette:
                    color = tuple(int(c) for c in palette[idx])
                    label = idx2label.get(idx, str(idx))
                    y = y0 + j * 25
                    cv2.rectangle(legend, (10, y-10), (40, y+10), color, -1)
                    cv2.putText(legend, f"{idx}: {label}", (50, y+5), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
            # Concatenate legend below overlay
            overlay = np.concatenate([overlay, legend], axis=0)
        else:
            # If mask is not 2D, skip overlay
            overlay = frame

        # Save the frame as an image file
        cv2.imwrite(f"{debug_dir}/debug_frame_{i}.png", overlay)
    print("Finished saving debug frames.")


def inspect_frames_with_fixation_objects(frames_h5_path, obj_csv_path, debug_dir):
    # Load frames
    frames = h5py.File(frames_h5_path, 'r')['frames']
    frame_keys = list(frames.keys())

    # Load fixation annotations
    df = pd.read_csv(obj_csv_path)

    # randomly select 10
    random.seed(42)
    selected_frame_keys = random.sample(frame_keys, 10)

    n = len(selected_frame_keys)
    for i in tqdm.tqdm(range(0, n)):
        frame_key = selected_frame_keys[i]
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
        out_path = os.path.join(debug_dir, f"{frame_key}_fixation_{x}_{y}_label_{label}.png")
        cv2.imwrite(out_path, frame_disp)
        print(f"Saved: {out_path} | Object: {label}, Fixation: ({x}, {y})")


if __name__ == "__main__":
    # Path to the HDF5 file
    suffix = '_02'
    frames_h5_path = f"/home/houhao/workspace/EyeTrackingSam/data/BF004/frames{suffix}.h5"
    masks_h5_path = f"/home/houhao/workspace/EyeTrackingSam/data/BF004/mask2former_semantic_maps{suffix}.h5"
    
    id2label_path = "/home/houhao/workspace/EyeTrackingSam/scripts/id2label.txt"

    obj_csv_path = f"/home/houhao/workspace/EyeTrackingSam/data/BF004/seg_fixation{suffix}.csv"

    debug_dir = "/home/houhao/workspace/EyeTrackingSam/data/BF004/frames_debug"
    # if debug_dir not exist
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # inspect sampled raw video frames from goggles
    # inspect_raw_frames(frames_h5_path, debug_dir)

    # inspect frames with mask2former 
    # inspect_frames_with_masks(frames_h5_path, masks_h5_path, id2label_path, debug_dir)

    inspect_frames_with_fixation_objects(frames_h5_path, obj_csv_path, debug_dir)

