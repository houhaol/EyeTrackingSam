import os
import re
import cv2
import argparse
import torch
import clip
import numpy as np
import json
import pandas as pd
from PIL import Image
from collections import defaultdict, Counter

def extract_timestamp_ns(filename):
    match = re.search(r"frames_(\d+)\.png", filename)
    return match.group(1) if match else None

def get_mask_bbox(mask_img):
    green_mask = (mask_img[:, :, 1] > 200) & (mask_img[:, :, 0] < 50) & (mask_img[:, :, 2] < 50)
    if green_mask.sum() == 0:
        return None, None
    y_idx, x_idx = np.where(green_mask)
    return green_mask, (x_idx.min(), y_idx.min(), x_max := x_idx.max(), y_max := y_idx.max())

def load_prompts_from_file(prompt_file):
    with open(prompt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_fixation_mapping(fixation_path):
    df = pd.read_csv(fixation_path)
    df["start timestamp [ns]"] = df["start timestamp [ns]"].astype(np.int64)
    df["end timestamp [ns]"] = df["end timestamp [ns]"].astype(np.int64)
    return df

def find_fixation_id_for_timestamp(ts, fixation_df):
    match = fixation_df[
        (fixation_df["start timestamp [ns]"] <= int(ts)) &
        (fixation_df["end timestamp [ns]"] >= int(ts))
    ]
    return int(match.iloc[0]["id"]) if not match.empty else None

def run_clip_labeling(frame_dir, mask_dir, overlay_dir, output_dir, prompt_list, json_output, fixation_path=None):
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    text_tokens = clip.tokenize(prompt_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    fixation_df = load_fixation_mapping(fixation_path) if fixation_path else None
    fixation_predictions = defaultdict(list)
    frame_level_results = []

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

        # Compute confidence score for top label
        top_confidence = float(similarity[0, pred_id].item())

        # Save frame-level result
        result = {
            "timestamp_ns": timestamp,
            "label": pred_label,
            "confidence": round(top_confidence, 4),
            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
        }
        frame_level_results.append(result)

        # Group by fixation if available
        if fixation_df is not None:
            fixation_id = find_fixation_id_for_timestamp(timestamp, fixation_df)
            if fixation_id is not None:
                fixation_predictions[fixation_id].append(pred_label)

        # Save annotated overlay
        annotated = overlay_img.copy()
        label_text = f"{pred_label} ({top_confidence:.2f})"
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
        cv2.putText(
            annotated, label_text, (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA
        )
        output_path = os.path.join(output_dir, f"clip_labeled_{timestamp}.png")
        cv2.imwrite(output_path, annotated)

    print('Fixation-level voting results with top-2 fallback and visual overlay')
    if fixation_df is not None:
        voted_results = []
        for _, row in fixation_df.iterrows():
            fixation_id = int(row["id"])
            start_ts = int(row["start timestamp [ns]"])
            end_ts = int(row["end timestamp [ns]"])
            labels = fixation_predictions.get(fixation_id, [])

            if not labels:
                continue

            vote_counter = Counter(labels)
            most_common = vote_counter.most_common()
            top_label = most_common[0][0]

            # Tie fallback to second-most
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                top_label = most_common[1][0]

            voted_results.append({
                "fixation_id": fixation_id,
                "start_timestamp_ns": start_ts,
                "end_timestamp_ns": end_ts,
                "label": top_label,
                "votes": dict(vote_counter)
            })

            # Annotate all frames in fixation
            for r in frame_level_results:
                ts = int(r["timestamp_ns"])
                if start_ts <= ts <= end_ts:
                    overlay_path = os.path.join(overlay_dir, f"overlay_{r['timestamp_ns']}.png")
                    if os.path.exists(overlay_path):
                        overlay_img = cv2.imread(overlay_path)
                        x_min, y_min, x_max, y_max = r["bbox"]
                        annotated = overlay_img.copy()
                        label_text = f"{top_label} (fixation {fixation_id}, {r.get('confidence', 0.0):.2f})"
                        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                        cv2.putText(
                            annotated, label_text, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
                        )
                        out_path = os.path.join(output_dir, f"fixation_labeled_{fixation_id}_{r['timestamp_ns']}.png")
                        cv2.imwrite(out_path, annotated)

        fixation_json = json_output.replace(".json", "_fixation_voted.json")
        with open(fixation_json, 'w') as f:
            json.dump(voted_results, f, indent=2)
        print(f"📄 Fixation-voted results saved to: {fixation_json}")

    if json_output:
        with open(json_output, 'w') as f:
            json.dump(frame_level_results, f, indent=2)
        print(f"📄 Frame-level results saved to: {json_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label SAM-segmented objects using CLIP and candidate prompts.")
    parser.add_argument("--frames", required=True, help="Directory of original frames (frames_<timestamp>.png)")
    parser.add_argument("--masks", required=True, help="Directory of SAM mask images (mask_<timestamp>.png)")
    parser.add_argument("--overlays", required=True, help="Directory of SAM overlay images (overlay_<timestamp>.png)")
    parser.add_argument("--output", default="clip_labeled_overlays", help="Directory to save labeled overlay images")
    parser.add_argument("--prompt_file", required=True, help="Path to text file containing one prompt per line")
    parser.add_argument("--json_output", default="clip_results.json", help="Path to save results as JSON")
    parser.add_argument("--fixation", help="Optional: path to fixation.csv for voting")

    args = parser.parse_args()
    prompts = load_prompts_from_file(args.prompt_file)

    run_clip_labeling(
        frame_dir=args.frames,
        mask_dir=args.masks,
        overlay_dir=args.overlays,
        output_dir=args.output,
        prompt_list=prompts,
        json_output=args.json_output,
        fixation_path=args.fixation
    )
