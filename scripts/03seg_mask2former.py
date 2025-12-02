
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import random
from tqdm import tqdm

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def run_mask2former(images, processor, model, target_sizes=None):
    # images: list of PIL Images or a single PIL Image
    if not isinstance(images, list):
        images = [images]
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    if target_sizes is None:
        target_sizes = [img.size[::-1] for img in images]
    segs = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
    # Convert each segmentation map to numpy array
    segs = [seg.cpu().numpy() for seg in segs]
    return segs if len(segs) > 1 else segs[0]

def get_label_id(label_name, id2label):
    for id, name in id2label.items():
        if name.lower() == label_name.lower():
            return id
    return None

def get_greenery_label_ids(id2label):
    greenery_keywords = ["tree", "grass", "vegetation", "bush", "plant", "shrub", "foliage"]
    greenery_ids = []
    for id, name in id2label.items():
        for keyword in greenery_keywords:
            if keyword in name.lower():
                greenery_ids.append(id)
                break
    return greenery_ids

def sky_greenery_ratio(segmentation_map, id2label):
    # compute sky and greenery ratio given segmentation map and id2label mapping
    total_pixels = segmentation_map.size
    sky_pixels = np.sum(segmentation_map == get_label_id("sky", id2label))
    greenery_pixels = np.sum(np.isin(segmentation_map, get_greenery_label_ids(id2label)))
    sky_ratio = sky_pixels / total_pixels
    greenery_ratio = greenery_pixels / total_pixels
    return sky_ratio, greenery_ratio

def sample_and_visualize(frames, frame_keys, processor, model, output_dir, num_samples=10, save_seg_h5=None):
    import random
    random.seed(42)
    sampled_indices = random.sample(range(len(frame_keys)), num_samples)
    color_palette = [list(np.random.choice(range(256), size=3)) for _ in range(len(model.config.id2label))]
    palette = np.array(color_palette)
    id2label = model.config.id2label

    # Create h5 file for saving segmentation maps if specified
    seg_file = None
    if save_seg_h5:
        seg_file = h5py.File(save_seg_h5, 'w')
        seg_group = seg_file.create_group('semantic_maps')
    visualize_all = False
    for idx in sampled_indices:
        frame = frames[frame_keys[idx]]
        frame_array = np.array(frame)
        image = Image.fromarray(frame_array.astype(np.uint8)).convert("RGB")
        seg = run_mask2former(image, processor, model)

        if not visualize_all:
            # Compute sky and greenery ratio
            sky_ratio, greenery_ratio = sky_greenery_ratio(seg, id2label)
            print(f"Frame {frame_keys[idx]}: Sky ratio = {sky_ratio:.4f}, Greenery ratio = {greenery_ratio:.4f}")
            # only visualize sky and greenery
            sky_label_id = get_label_id("sky", id2label)
            greenery_label_ids = get_greenery_label_ids(id2label)
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            if sky_label_id is not None:
                color_seg[seg == sky_label_id, :] = [255, 0, 0]  # Red for sky
            if greenery_label_ids:
                color_seg[np.isin(seg, greenery_label_ids), :] = [0, 255, 0]  # Green for greenery
            orig_img = np.array(image)
            overlay = (orig_img * 0.7 + color_seg * 0.3).astype(np.uint8)
            overlay_bgr = overlay[:, :, ::-1]  # Convert RGB to BGR
            seg_img = Image.fromarray(overlay_bgr)
            seg_img.save(os.path.join(output_dir, f"{frame_keys[idx]}.png"))
        
        # Save segmentation to h5 if specified
        if seg_file is not None:
            seg_group.create_dataset(frame_keys[idx], data=seg, compression="gzip")
        
        # visualize all labels
        if visualize_all:
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color

            present_labels = np.unique(seg)
            legend_handles = []
            for label_id in present_labels:
                label_name = id2label[label_id]
                color = palette[label_id] / 255.0
                patch = mpatches.Patch(color=color, label=label_name)
                legend_handles.append(patch)

            orig_img = np.array(image)
            overlay = (orig_img * 0.5 + color_seg * 0.5).astype(np.uint8)

            fig, ax = plt.subplots(figsize=(15, 10))
            ax.imshow(overlay)
            ax.axis('off')
            ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/predicted_semantic_map_{idx}.png", bbox_inches='tight')
            plt.close(fig)
    
    # Close h5 file if it was created
    if seg_file is not None:
        seg_file.close()

def process_all_frames(frames, frame_keys, processor, model, output_h5_path, batch_size=64):
    # batch_size can be adjusted as needed
    with h5py.File(output_h5_path, 'w') as out_f:
        seg_group = out_f.create_group('semantic_maps')
        num_frames = len(frame_keys)
        for start in tqdm(range(0, num_frames, batch_size), desc="Processing frames", unit="batch"):
            end = min(start + batch_size, num_frames)
            batch_keys = frame_keys[start:end]
            batch_images = []
            batch_sizes = []
            for key in batch_keys:
                frame = frames[key]
                frame_array = np.array(frame)
                image = Image.fromarray(frame_array.astype(np.uint8)).convert("RGB")
                batch_images.append(image)
                batch_sizes.append(image.size[::-1])
            
            segs = run_mask2former(batch_images, processor, model, batch_sizes)
            for key, seg in zip(batch_keys, segs):
                seg_group.create_dataset(key, data=seg, compression="gzip")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Mask2Former semantic segmentation on frames.")
    parser.add_argument('--h5_path', type=str, default="/home/houhao/workspace/EyeTrackingSam/data/pilot2/frames/frames.h5", help='Path to input frames h5 file')
    parser.add_argument('--output_dir', type=str, default="/home/houhao/workspace/EyeTrackingSam/data/pilot2/mask2former_output", help='Directory to save visualization outputs')
    parser.add_argument('--output_h5_path', type=str, default="/home/houhao/workspace/EyeTrackingSam/data/pilot2/mask2former_semantic_maps.h5", help='Path to save semantic maps h5 file')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--id2label_path', type=str, default="id2label.txt", help='Path to save id2label mapping')
    parser.add_argument('--model_name', type=str, default="facebook/mask2former-swin-large-mapillary-vistas-semantic", help='Model name or path')
    args = parser.parse_args()

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_name)
    model.to(DEVICE)
    id2label = model.config.id2label
    with open(args.id2label_path, "w") as f:
        for id, label in id2label.items():
            f.write(f"{id}: {label}\n")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with h5py.File(args.h5_path, 'r') as f:
        frames = f['frames']
        frame_keys = list(frames.keys())
        # sample_and_visualize(frames, frame_keys, processor, model, args.output_dir, num_samples=args.num_samples, save_seg_h5=args.output_dir+"/demo_semantic_maps.h5")
        process_all_frames(frames, frame_keys, processor, model, args.output_h5_path)
