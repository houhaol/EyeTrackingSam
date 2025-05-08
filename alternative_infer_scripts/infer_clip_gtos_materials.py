import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import open_clip


def load_model(model_name, pretrained, checkpoint_path, num_classes, device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device
    )
    model.to(device).eval()

    classifier = torch.nn.Linear(model.visual.output_dim, num_classes).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    classifier.load_state_dict(checkpoint['classifier'])

    return model, classifier, preprocess


def get_mask_bbox(mask_img):
    green_mask = (mask_img[:, :, 1] > 200) & (mask_img[:, :, 0] < 50) & (mask_img[:, :, 2] < 50)
    if green_mask.sum() == 0:
        return None
    y_idx, x_idx = np.where(green_mask)
    return x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()


def predict_cropped_region(image, bbox, model, classifier, preprocess, classnames, device):
    x_min, y_min, x_max, y_max = bbox
    cropped = image[y_min:y_max+1, x_min:x_max+1]
    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    image_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        logits = classifier(features)
        pred_id = logits.argmax(dim=1).item()

    return classnames[pred_id]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load classnames
    with open(args.classnames_txt, 'r') as f:
        classnames = [line.strip() for line in f if line.strip()]

    model, classifier, preprocess = load_model(
        model_name=args.model,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        num_classes=len(classnames),
        device=device
    )

    # Iterate over all masks
    for fname in sorted(os.listdir(args.masks)):
        if not fname.startswith("mask_") or not fname.endswith(".png"):
            continue

        timestamp = fname.replace("mask_", "").replace(".png", "")
        frame_path = os.path.join(args.frames, f"frames_{timestamp}.png")
        mask_path = os.path.join(args.masks, fname)

        if not os.path.exists(frame_path):
            print(f"⚠️ Frame not found for {fname}")
            continue

        frame = cv2.imread(frame_path)
        mask_img = cv2.imread(mask_path)
        bbox = get_mask_bbox(mask_img)
        if bbox is None:
            print(f"⚠️ No valid green mask found in {fname}")
            continue

        label = predict_cropped_region(frame, bbox, model, classifier, preprocess, classnames, device)

        # Annotate
        annotated = frame.copy()
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
        cv2.putText(annotated, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        out_path = os.path.join(args.output, f"labeled_{timestamp}.png")
        cv2.imwrite(out_path, annotated)
        print(f"✅ Labeled {fname} as {label}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Infer using finetuned OpenCLIP model on masked regions")
    parser.add_argument("--frames", type=str, required=True, help="Directory of original frames")
    parser.add_argument("--masks", type=str, required=True, help="Directory of mask images")
    parser.add_argument("--output", type=str, required=True, help="Directory to save labeled overlay images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="OpenCLIP pretrained weights")
    parser.add_argument("--classnames_txt", type=str, required=True, help="Path to classInd.txt or class name file")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
