import cv2
import os
import argparse
from glob import glob

def frames_to_video(frame_dir, output_path, fps=24):
    # Collect all .png frames and sort them
    frame_paths = sorted(glob(os.path.join(frame_dir, "*.png")))

    if not frame_paths:
        raise ValueError("No PNG images found in the directory.")

    # Read the first image to get frame size
    sample_frame = cv2.imread(frame_paths[0])
    height, width, _ = sample_frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"⚠️ Skipping unreadable frame: {frame_path}")
            continue
        out.write(frame)

    out.release()
    print(f"🎞️ Video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge image frames into a video.")
    parser.add_argument("--frames", required=True, help="Directory containing .png frames")
    parser.add_argument("--output", required=True, help="Output video file path (e.g., output.mp4)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    args = parser.parse_args()

    frames_to_video(args.frames, args.output, args.fps)
