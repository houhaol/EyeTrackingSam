import cv2
import os
import argparse
from glob import glob

def frames_to_video(frame_dir, output_path, fps=24, resize_ratio=1.0):
    # Collect all .png frames and sort them
    frame_paths = sorted(glob(os.path.join(frame_dir, "*.png")))

    if not frame_paths:
        raise ValueError("No PNG images found in the directory.")

    # Read the first image to get frame size
    sample_frame = cv2.imread(frame_paths[0])
    if resize_ratio != 1.0:
        sample_frame = cv2.resize(sample_frame, (0, 0), fx=resize_ratio, fy=resize_ratio)

    height, width, _ = sample_frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"⚠️ Skipping unreadable frame: {frame_path}")
            continue
        if resize_ratio != 1.0:
            frame = cv2.resize(frame, (width, height))
        out.write(frame)

    out.release()
    print(f"🎞️ Video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge image frames into a video.")
    parser.add_argument("--frames", required=True, help="Directory containing .png frames")
    parser.add_argument("--output", required=True, help="Output video file path (e.g., output.mp4)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    parser.add_argument("--resize_ratio", type=float, default=1.0,
                        help="Resize ratio (e.g. 0.5 for half-size output)")
    args = parser.parse_args()

    frames_to_video(args.frames, args.output, args.fps, args.resize_ratio)
