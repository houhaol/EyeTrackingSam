import cv2
import numpy as np
import os
import argparse
from glob import glob

def sample_frames(video_path, timestamps_path, output_dir, test_mode=False, start_time=None, end_time=None):
    os.makedirs(output_dir, exist_ok=True)

    timestamps_ns = np.load(timestamps_path)
    if test_mode:
        timestamps_ns = timestamps_ns[:10]

    # Use the first timestamp as the video start time
    start_timestamp_ns = timestamps_ns[0]
    relative_timestamps_s = (timestamps_ns - start_timestamp_ns) / 1e9

    # Filter by start and end time if provided
    if start_time is not None and end_time is not None:
        mask = (relative_timestamps_s >= start_time) & (relative_timestamps_s <= end_time)
        timestamps_ns = timestamps_ns[mask]
        relative_timestamps_s = relative_timestamps_s[mask]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞 Total frames: {total_frames}, FPS: {fps:.2f}")

    for timestamp_ns, rel_time_s in zip(timestamps_ns, relative_timestamps_s):
        frame_index = min(int(rel_time_s * fps), total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if not success:
            print(f"⚠️ Failed to read frame at index={frame_index} (timestamp={timestamp_ns})")
            continue

        filename = os.path.join(output_dir, f"frames_{timestamp_ns}.png")
        cv2.imwrite(filename, frame)
        print(f"✅ Saved frame: {filename}")

    cap.release()
    print("🎉 Frame sampling completed.")

def merge_frames_to_video(frame_dir, output_video_path, fps=30):
    frame_files = sorted(glob(os.path.join(frame_dir, "frames_*.png")))
    if not frame_files:
        print("❌ No frames found to merge.")
        return

    # Read the first frame to get size
    first_frame = cv2.imread(frame_files[0])
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change to 'XVID' or 'avc1' if needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        out.write(frame)

    out.release()
    print(f"🎬 Merged video saved to: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample video frames using real-world Unix timestamps")
    parser.add_argument("--video", required=True, help="Path to the input video (e.g. video.mp4)")
    parser.add_argument("--timestamps", required=True, help="Path to world_timestamps_unix.npy (nanoseconds)")
    parser.add_argument("--output", default="sampled_frames", help="Directory to save sampled frames")
    parser.add_argument("--test", action="store_true", help="Only sample the first 10 timestamps for testing")
    parser.add_argument("--start", type=float, help="Start time in seconds from beginning of video")
    parser.add_argument("--end", type=float, help="End time in seconds from beginning of video")
    parser.add_argument("--merge", action="store_true", help="Merge sampled frames into a video")
    parser.add_argument("--merge_fps", type=float, default=30, help="FPS for the output merged video")

    args = parser.parse_args()

    sample_frames(args.video, args.timestamps, args.output, args.test, args.start, args.end)

    if args.merge:
        output_video_path = os.path.join(args.output, "merged_output.mp4")
        merge_frames_to_video(args.output, output_video_path, fps=args.merge_fps)
