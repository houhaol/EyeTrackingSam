import cv2
import numpy as np
import os
import argparse
from glob import glob
import h5py


def sample_frames(video_path, timestamps_path, output_dir, test_mode=False, start_time=None, end_time=None, save_images=True, h5_path=None, frame_interval=1):
    os.makedirs(output_dir, exist_ok=True)

    # If h5_path is provided, ensure it is inside output_dir
    if h5_path is not None:
        # If h5_path is just a filename, join with output_dir
        if not os.path.isabs(h5_path):
            h5_path = os.path.join(output_dir, h5_path)
        else:
            # If absolute, check if it's in output_dir, else warn and use output_dir
            if not os.path.commonpath([os.path.abspath(h5_path), os.path.abspath(output_dir)]) == os.path.abspath(output_dir):
                print(f"⚠️ h5_path is not in output_dir, saving to output_dir instead.")
                h5_path = os.path.join(output_dir, os.path.basename(h5_path))

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

    h5_file = None
    if h5_path is not None:
        h5_file = h5py.File(h5_path, 'w')
        frames_group = h5_file.create_group('frames')


    for idx, (timestamp_ns, rel_time_s) in enumerate(zip(timestamps_ns, relative_timestamps_s)):
        if idx % frame_interval != 0:
            continue
        print(f"Processing frame {idx+1}/{len(timestamps_ns)} (timestamp={timestamp_ns})")
        frame_index = min(int(rel_time_s * fps), total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if not success:
            print(f"⚠️ Failed to read frame at index={frame_index} (timestamp={timestamp_ns})")
            continue

        if save_images:
            filename = os.path.join(output_dir, f"frames_{timestamp_ns}.png")
            cv2.imwrite(filename, frame)
            print(f"✅ Saved frame: {filename}")
        if h5_file is not None:
            frames_group.create_dataset(f'frame_{timestamp_ns}', data=frame, dtype='uint8')

    if h5_file is not None:
        h5_file.close()
        print(f"💾 HDF5 file saved to: {h5_path}")

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
    parser.add_argument("--h5", type=str, default=None, help="Path to output HDF5 file (optional)")
    parser.add_argument("--no_image", action="store_true", help="Do not save PNG images, only save to HDF5 if specified")
    parser.add_argument("--frame_interval", type=int, default=2, help="Interval for saving frames to HDF5 (e.g., 2 means save every 2nd frame)")

    args = parser.parse_args()

    sample_frames(
        args.video,
        args.timestamps,
        args.output,
        args.test,
        args.start,
        args.end,
        save_images=not args.no_image,
        h5_path=args.h5,
        frame_interval=args.frame_interval
    )

    if args.merge:
        output_video_path = os.path.join(args.output, "merged_output.mp4")
        merge_frames_to_video(args.output, output_video_path, fps=args.merge_fps)
