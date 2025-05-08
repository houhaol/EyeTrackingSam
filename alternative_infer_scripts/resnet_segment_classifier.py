import os
import cv2
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import requests

def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return response.text.strip().splitlines()

def get_mask_bbox(mask_image):
    # Detect green area (mask drawn as [0,255,0])
    green_mask = (mask_image[:, :, 1] > 200) & (mask_image[:, :, 0] < 50) & (mask_image[:, :, 2] < 50)
    if green_mask.sum() == 0:
        return None, None, None
    y_indices, x_indices = np.where(green_mask)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return green_mask, (x_min, y_min, x_max, y_max)

def run_visualization(frame_dir, mask_dir, overlay_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load model and labels
    model = models.resnet50(pretrained=True)
    model.eval()
    labels = load_imagenet_labels()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    for fname in os.listdir(mask_dir):
        if not fname.startswith("mask_") or not fname.endswith(".png"):
            continue

        timestamp = fname.replace("mask_", "").replace(".png", "")
        frame_path = os.path.join(frame_dir, f"frames_{timestamp}.png")
        mask_path = os.path.join(mask_dir, fname)
        overlay_path = os.path.join(overlay_dir, f"overlay_{timestamp}.png")

        if not os.path.exists(frame_path) or not os.path.exists(overlay_path):
            print(f"⚠️ Missing frame or overlay for {timestamp}")
            continue

        frame = cv2.imread(frame_path)
        mask_img = cv2.imread(mask_path)
        overlay_img = cv2.imread(overlay_path)

        green_mask, bbox = get_mask_bbox(mask_img)
        if bbox is None:
            print(f"⚠️ No mask found in {fname}")
            continue

        x_min, y_min, x_max, y_max = bbox
        cropped = frame[y_min:y_max+1, x_min:x_max+1]
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(pil_img).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)
            pred_id = logits.argmax().item()
            label = labels[pred_id]

        # Annotate overlay
        annotated = overlay_img.copy()
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        cv2.putText(
            annotated, label, (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
        )

        # Save labeled overlay
        output_path = os.path.join(output_dir, f"labeled_{timestamp}.png")
        cv2.imwrite(output_path, annotated)
        print(f"✅ Saved labeled overlay: {output_path} — [{label}]")

    print("🎉 All labels visualized and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predicted labels on SAM mask overlays.")
    parser.add_argument("--frames", required=True, help="Path to directory of original frames (frames_<timestamp>.png)")
    parser.add_argument("--masks", required=True, help="Path to directory of SAM mask images (mask_<timestamp>.png)")
    parser.add_argument("--overlays", required=True, help="Path to directory of overlay images (overlay_<timestamp>.png)")
    parser.add_argument("--output", default="labeled_overlays", help="Directory to save labeled overlay images")
    args = parser.parse_args()

    run_visualization(
        frame_dir=args.frames,
        mask_dir=args.masks,
        overlay_dir=args.overlays,
        output_dir=args.output
    )
