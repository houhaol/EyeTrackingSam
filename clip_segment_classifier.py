import os
import re
import cv2
import argparse
import torch
import clip
import numpy as np
import json
from PIL import Image

def extract_timestamp_ns(filename):
    match = re.search(r"frames_(\d+)\.png", filename)
    return match.group(1) if match else None

def get_mask_bbox(mask_img):
    green_mask = (mask_img[:, :, 1] > 200) & (mask_img[:, :, 0] < 50) & (mask_img[:, :, 2] < 50)
    if green_mask.sum() == 0:
        return None, None
    y_idx, x_idx = np.where(green_mask)
    return green_mask, (x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max())

def load_prompts_from_file(prompt_file):
    with open(prompt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def run_clip_labeling(frame_dir, mask_dir, overlay_dir, output_dir, prompt_list, json_output):
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Tokenize prompt labels
    text_tokens = clip.tokenize(prompt_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    results = []

    for fname in os.listdir(mask_dir):
        if not fname.startswith("mask_") or not fname.endswith(".png"):
            continue

        timestamp = fname.replace("mask_", "").replace(".png", "")
        frame_path = os.path.join(frame_dir, f"frames_{timestamp}.png")
        mask_path = os.path.join(mask_dir, fname)
        overlay_path = os.path.join(overlay_dir, f"overlay_{timestamp}.png")

        if not (os.path.exists(frame_path) and os.path.exists(mask_path) and os.path.exists(overlay_path)):
            print(f"Skipping {timestamp} — missing required files")
            continue

        frame = cv2.imread(frame_path)
        mask_img = cv2.imread(mask_path)
        overlay_img = cv2.imread(overlay_path)

        green_mask, bbox = get_mask_bbox(mask_img)
        if bbox is None:
            print(f"No valid mask in {fname}")
            continue

        x_min, y_min, x_max, y_max = bbox
        cropped = frame[y_min:y_max+1, x_min:x_max+1]
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        image_input = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred_id = similarity.argmax().item()
            pred_label = prompt_list[pred_id]

        # Annotate overlay
        annotated = overlay_img.copy()
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
        cv2.putText(
            annotated, pred_label, (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA
        )

        output_path = os.path.join(output_dir, f"clip_labeled_{timestamp}.png")
        cv2.imwrite(output_path, annotated)
        print(f"✅ Labeled {timestamp}: {pred_label}")

        # Save result to list for JSON
        results.append({
            "timestamp_ns": timestamp,
            "label": pred_label,
            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
        })

    print("🎉 CLIP-based label classification complete.")

    if json_output:
        with open(json_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to: {json_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label SAM-segmented objects using CLIP and candidate prompts.")
    parser.add_argument("--frames", required=True, help="Directory of original frames (frames_<timestamp>.png)")
    parser.add_argument("--masks", required=True, help="Directory of SAM mask images (mask_<timestamp>.png)")
    parser.add_argument("--overlays", required=True, help="Directory of SAM overlay images (overlay_<timestamp>.png)")
    parser.add_argument("--output", default="clip_labeled_overlays", help="Directory to save labeled overlay images")
    parser.add_argument("--prompt_file", required=True, help="Path to text file containing one prompt per line")
    parser.add_argument("--json_output", default="clip_results.json", help="Path to save results as JSON")

    args = parser.parse_args()
    prompts = load_prompts_from_file(args.prompt_file)

    run_clip_labeling(
        frame_dir=args.frames,
        mask_dir=args.masks,
        overlay_dir=args.overlays,
        output_dir=args.output,
        prompt_list=prompts,
        json_output=args.json_output
    )
