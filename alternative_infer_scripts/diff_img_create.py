import os
import cv2
import numpy as np
from glob import glob
import argparse

def get_mask_bbox(mask_img):
    """Get bounding box of green mask region."""
    green_mask = (mask_img[:, :, 1] > 200) & (mask_img[:, :, 0] < 50) & (mask_img[:, :, 2] < 50)
    if green_mask.sum() == 0:
        return None
    y_idx, x_idx = np.where(green_mask)
    return x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()

def load_and_sort_files(folder, prefix):
    """Load and sort image paths based on numeric timestamp."""
    files = glob(os.path.join(folder, f"{prefix}_*.png"))
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    return files

def create_cropped_patches(frame_dir, mask_dir, output_rgb_dir, output_diff_dir):
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_diff_dir, exist_ok=True)

    frame_files = load_and_sort_files(frame_dir, "frames")
    mask_files = load_and_sort_files(mask_dir, "mask")

    for i in range(1, len(frame_files)):
        current_frame = cv2.imread(frame_files[i])
        prev_frame = cv2.imread(frame_files[i - 1])
        mask = cv2.imread(mask_files[i])

        if current_frame is None or prev_frame is None or mask is None:
            continue

        bbox = get_mask_bbox(mask)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        rgb_patch = current_frame[y1:y2, x1:x2]
        prev_patch = prev_frame[y1:y2, x1:x2]
        diff_patch = cv2.absdiff(rgb_patch, prev_patch)

        frame_id = os.path.basename(frame_files[i]).split('.')[0]
        rgb_path = os.path.join(output_rgb_dir, f"{frame_id}_rgb.png")
        diff_path = os.path.join(output_diff_dir, f"{frame_id}_diff.png")

        cv2.imwrite(rgb_path, rgb_patch)
        cv2.imwrite(diff_path, diff_patch)
        print(f"Saved RGB patch to {rgb_path} and differential patch to {diff_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate RGB and differential image patches from masked video frames.")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory containing frame images.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing mask images.")
    parser.add_argument("--output_rgb_dir", type=str, required=True, help="Directory to save cropped RGB patches.")
    parser.add_argument("--output_diff_dir", type=str, required=True, help="Directory to save cropped differential patches.")
    args = parser.parse_args()

    create_cropped_patches(args.frame_dir, args.mask_dir, args.output_rgb_dir, args.output_diff_dir)

if __name__ == "__main__":
    main()
