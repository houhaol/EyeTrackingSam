import os
import re
import cv2
import json
import torch
import open_clip
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict, Counter
import tqdm

def extract_timestamp_ns(filename):
    match = re.search(r"frames_(\d+)\.png", filename)
    return match.group(1) if match else None

def get_mask_bbox(mask_img):
    green_mask = (mask_img[:, :, 1] > 200) & (mask_img[:, :, 0] < 50) & (mask_img[:, :, 2] < 50)
    if green_mask.sum() == 0:
        return None, None
    y_idx, x_idx = np.where(green_mask)
    return green_mask, (x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max())

def expand_bbox(bbox, img_shape, margin=20, size_threshold=300):
    x_min, y_min, x_max, y_max = bbox
    h, w = img_shape[:2]
    if (x_max - x_min > size_threshold) or (y_max - y_min > size_threshold):
        return x_min, y_min, x_max, y_max
    return (
        max(0, x_min - margin), max(0, y_min - margin),
        min(w - 1, x_max + margin), min(h - 1, y_max + margin)
    )

def load_prompts_from_file(prompt_file):
    with open(prompt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_fixation_mapping(fixation_path):
    df = pd.read_csv(fixation_path)
    df["start timestamp [ns]"] = df["start timestamp [ns]"].astype(np.int64)
    df["end timestamp [ns]"] = df["end timestamp [ns]"].astype(np.int64)
    return df

def find_fixation_id_for_timestamp(ts, fixation_df):
    match = fixation_df[(fixation_df["start timestamp [ns]"] <= int(ts)) &
                        (fixation_df["end timestamp [ns]"] >= int(ts))]
    return int(match.iloc[0]["id"]) if not match.empty else None

class CLIPLabeler:
    def __init__(self, frame_dir, mask_dir, overlay_dir, output_dir,
                 prompt_list, json_output, fixation_path=None,
                 confidence_thr=0.6, visualize=True, vote_fixation=True,
                 merge_video=False):
        self.frame_dir = frame_dir
        self.mask_dir = mask_dir
        self.overlay_dir = overlay_dir
        self.output_dir = output_dir
        self.prompt_list = prompt_list
        self.json_output = json_output
        self.fixation_path = fixation_path
        self.confidence_thr = confidence_thr
        self.visualize = visualize
        self.vote_fixation = vote_fixation
        self.merge_video = merge_video
        os.makedirs(output_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14-quickgelu", pretrained="dfn5b", device=self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")
        self.text_features = self._encode_text(prompt_list)

        self.fixation_df = load_fixation_mapping(fixation_path) if fixation_path else None
        self.fixation_predictions = defaultdict(list)
        self.frame_level_results = []
        self.video_frames = []

    def _encode_text(self, prompt_list):
        tokens = self.tokenizer(prompt_list).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            return features / features.norm(dim=-1, keepdim=True)

    def run(self):
        for fname in tqdm.tqdm(sorted(os.listdir(self.mask_dir))):
            if not fname.startswith("mask_") or not fname.endswith(".png"):
                continue

            timestamp = fname.replace("mask_", "").replace(".png", "")
            frame_path = os.path.join(self.frame_dir, f"frames_{timestamp}.png")
            mask_path = os.path.join(self.mask_dir, fname)
            overlay_path = os.path.join(self.overlay_dir, f"overlay_{timestamp}.png")

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

            x_min, y_min, x_max, y_max = expand_bbox(bbox, frame.shape)
            masked = frame.copy()
            masked[~green_mask] = 0
            cropped = masked[y_min:y_max+1, x_min:x_max+1]
            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                pred_id = similarity.argmax().item()
                pred_label = self.prompt_list[pred_id]
                confidence = float(similarity[0, pred_id].item())

            self.frame_level_results.append({
                "timestamp_ns": timestamp,
                "label": pred_label,
                "confidence": round(confidence, 4),
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
            })

            if self.fixation_df is not None:
                fixation_id = find_fixation_id_for_timestamp(timestamp, self.fixation_df)
                if fixation_id is not None:
                    self.fixation_predictions[fixation_id].append(pred_label)

            if self.visualize:
                annotated = self._annotate_image(confidence, pred_label, overlay_img, frame, x_min, y_min, x_max, y_max)
                if self.merge_video:
                    self.video_frames.append(annotated)
                else:
                    out_path = os.path.join(self.output_dir, f"clip_labeled_{timestamp}.png")
                    cv2.imwrite(out_path, annotated)

        if self.json_output:
            with open(self.json_output, 'w') as f:
                json.dump(self.frame_level_results, f, indent=2)
            print(f"📄 Frame-level results saved to: {self.json_output}")

        if self.fixation_df is not None and self.vote_fixation:
            self._vote_fixation()

        if self.visualize and self.merge_video and self.video_frames:
            out_video_path = os.path.join(self.output_dir, "clip_labeled_video.mp4")
            height, width, _ = self.video_frames[0].shape
            out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
            for frame in self.video_frames:
                out.write(frame)
            out.release()
            print(f"📽️ Merged video saved to: {out_video_path}")

    def _annotate_image(self, confidence, pred_label, overlay_img, raw_img, x_min, y_min, x_max, y_max):
        annotated = overlay_img.copy()
        if confidence >= self.confidence_thr:
            label_text = f"{pred_label} ({confidence:.2f})"
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
            cv2.putText(annotated, label_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            annotated = raw_img.copy()
        return annotated

    def _vote_fixation(self):
        voted_results = []
        for i in range(len(self.frame_level_results)):
            window = self.frame_level_results[max(0, i - 2):i + 3]  # sliding window of size 5
            labels = [r["label"] for r in window]
            vote_counter = Counter(labels)
            most_common = vote_counter.most_common()
            top_label = most_common[0][0] if most_common else None
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                top_label = most_common[1][0]

            result = self.frame_level_results[i]
            result_ts = result["timestamp_ns"]
            result_bbox = result["bbox"]

            voted_results.append({
                "timestamp_ns": result_ts,
                "label": top_label,
                "votes": dict(vote_counter)
            })

            if self.visualize:
                overlay_path = os.path.join(self.overlay_dir, f"overlay_{result_ts}.png")
                if os.path.exists(overlay_path):
                    overlay_img = cv2.imread(overlay_path)
                    x_min, y_min, x_max, y_max = result_bbox
                    annotated = overlay_img.copy()
                    label_text = f"{top_label} (windowed)"
                    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
                    cv2.putText(annotated, label_text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
                    if self.merge_video:
                        self.video_frames.append(annotated)
                    else:
                        out_path = os.path.join(self.output_dir, f"voted_labeled_{result_ts}.png")
                        cv2.imwrite(out_path, annotated)

        fixation_json = self.json_output.replace(".json", "_fixation_voted.json")
        with open(fixation_json, 'w') as f:
            json.dump(voted_results, f, indent=2)
        print(f"📄 Fixation-voted results saved to: {fixation_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--overlays", required=True)
    parser.add_argument("--output", default="clip_labeled_overlays")
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--json_output", default="clip_results.json")
    parser.add_argument("--fixation", help="Optional: path to fixation.csv")
    parser.add_argument("--confidence_thr", type=float, default=0.6)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vote_fixation", action="store_true")
    parser.add_argument("--merge_video", action="store_true", help="Merge output frames into a single video")
    args = parser.parse_args()

    prompts = load_prompts_from_file(args.prompt_file)
    labeler = CLIPLabeler(
        frame_dir=args.frames,
        mask_dir=args.masks,
        overlay_dir=args.overlays,
        output_dir=args.output,
        prompt_list=prompts,
        json_output=args.json_output,
        fixation_path=args.fixation,
        confidence_thr=args.confidence_thr,
        visualize=args.visualize,
        vote_fixation=args.vote_fixation,
        merge_video=args.merge_video
    )
    labeler.run()
