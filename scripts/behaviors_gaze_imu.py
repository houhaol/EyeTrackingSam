import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import argparse


class GazeIMUProcessor:
    def __init__(self, args):
        self.args = args
        self.load_data()
        self.select_segment()
        self.resample_data()
        self.compute_vectors()

    def load_data(self):
        self.gaze = pd.read_csv(self.args.gaze)
        self.imu = pd.read_csv(self.args.imu)
        self.frame_timestamps = np.load(self.args.timestamps)
        with open(self.args.info, "r") as f:
            self.info = json.load(f)
        self.start_time = self.info["start_time"]

        self.gaze["relative ts [s]"] = (self.gaze["timestamp [ns]"] - self.start_time) * 1e-9
        self.imu["relative ts [s]"] = (self.imu["timestamp [ns]"] - self.start_time) * 1e-9

    def select_segment(self):
        video_start = self.args.video_start_time_sec or 0
        video_end = self.args.video_end_time_sec or (self.frame_timestamps[-1] - self.start_time) * 1e-9

        abs_start_ts = self.start_time + video_start * 1e9
        abs_end_ts = self.start_time + video_end * 1e9

        self.start_frame_idx = np.searchsorted(self.frame_timestamps, abs_start_ts, side="left")
        self.end_frame_idx = np.searchsorted(self.frame_timestamps, abs_end_ts, side="right")

        self.frame_timestamps = self.frame_timestamps[self.start_frame_idx:self.end_frame_idx]
        self.relative_demo_video_ts = (self.frame_timestamps - self.start_time) * 1e-9

    def resample_data(self):
        ts = self.relative_demo_video_ts
        imu = self.imu
        gaze = self.gaze

        self.quaternions_resampled = np.array([
            np.interp(ts, imu["relative ts [s]"], imu[f"quaternion {k}"]) for k in "wxyz"
        ]).T

        self.gaze_elevation = np.interp(ts, gaze["relative ts [s]"], gaze["elevation [deg]"])
        self.gaze_azimuth = np.interp(ts, gaze["relative ts [s]"], gaze["azimuth [deg]"])
        self.gaze_px_x = np.interp(ts, gaze["relative ts [s]"], gaze["gaze x [px]"])
        self.gaze_px_y = np.interp(ts, gaze["relative ts [s]"], gaze["gaze y [px]"])

    def compute_vectors(self):
        self.headings_in_world = self.imu_heading_in_world(self.quaternions_resampled)
        self.cart_gazes_in_world = self.gaze_3d_to_world(self.gaze_elevation, self.gaze_azimuth, self.quaternions_resampled)

    @staticmethod
    def imu_heading_in_world(quats):
        heading_vec = np.array([0.0, 1.0, 0.0])
        return transform_imu_to_world(heading_vec, quats)

    @staticmethod
    def gaze_3d_to_world(elev, azim, quats):
        cart = spherical_to_cartesian_scene(elev, azim)
        return transform_scene_to_world(cart, quats, translation_in_imu=np.zeros(3))

    def generate_plots(self):
        plot_vectors_over_time(self.relative_demo_video_ts, self.headings_in_world, "IMU Heading", "imu_heading_components.png")
        plot_vectors_over_time(self.relative_demo_video_ts, self.cart_gazes_in_world, "Gaze", "gaze_direction_components.png")

    def overlay_video(self):
        cap = cv2.VideoCapture(self.args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame_idx)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(self.args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for idx in range(len(self.relative_demo_video_ts)):
            ret, frame = cap.read()
            if not ret:
                break
            gaze_px = np.array([self.gaze_px_x[idx], self.gaze_px_y[idx]])
            gaze_vec = self.cart_gazes_in_world[idx]
            head_vec = self.headings_in_world[idx]

            cv2.circle(frame, gaze_px.astype(int), 20, (0, 255, 255), -1)
            label = f"Gaze: {vector_to_direction(gaze_vec)} | Head: {vector_to_direction(head_vec)}"
            cv2.putText(frame, label, (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            out.write(frame)

        cap.release()
        out.release()
        print(f"✅ Output video saved to {self.args.output}")

    def export_csv(self, filename="gaze_head_behavior.csv"):
        rows = []
        for idx in range(len(self.relative_demo_video_ts)):
            unix_ts_ns = self.frame_timestamps[idx]
            gaze_dir = vector_to_direction(self.cart_gazes_in_world[idx])
            head_dir = vector_to_direction(self.headings_in_world[idx])
            rows.append({
                "timestamp_ns": unix_ts_ns,
                "gaze_behavior": gaze_dir,
                "head_behavior": head_dir
            })
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"✅ CSV exported to {filename}")


def transform_imu_to_world(imu_coords, quats):
    mats = R.from_quat(quats, scalar_first=True).as_matrix()
    return np.einsum('nij,nj->ni', mats, imu_coords) if imu_coords.ndim > 1 else mats @ imu_coords

def transform_scene_to_imu(coords, trans=np.array([0.0, -1.3, -6.62])):
    theta = np.deg2rad(-102)
    rot = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    coords = rot @ coords.T
    coords[0] += trans[0]
    coords[1] += trans[1]
    coords[2] += trans[2]
    return coords.T

def spherical_to_cartesian_scene(elev, azim):
    elev = np.deg2rad(elev) + np.pi / 2
    azim = -np.deg2rad(azim) + np.pi / 2
    return np.array([
        np.sin(elev) * np.cos(azim),
        np.cos(elev),
        np.sin(elev) * np.sin(azim),
    ]).T

def transform_scene_to_world(coords, quats, translation_in_imu=np.array([0.0, -1.3, -6.62])):
    coords_in_imu = transform_scene_to_imu(coords, translation_in_imu)
    return transform_imu_to_world(coords_in_imu, quats)

def vector_to_direction(vec, h_thresh=0.5, v_thresh=0.0):
    directions = []
    if vec[0] > h_thresh:
        directions.append("Right")
    elif vec[0] < -h_thresh:
        directions.append("Left")
    if vec[2] > v_thresh:
        directions.append("Up")
    elif vec[2] < v_thresh:
        directions.append("Down")
    return " & ".join(directions) if directions else "Neutral"

def plot_vectors_over_time(times, vectors, label_prefix, output_path):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[i].plot(times, vectors[:, i], label=f'{label_prefix} {axis}')
        axs[i].set_ylabel(f'{axis} component')
        axs[i].legend()
        axs[i].grid(True)
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{label_prefix} Vector Components Over Time")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--gaze', type=str, required=True)
    parser.add_argument('--imu', type=str, required=True)
    parser.add_argument('--info', type=str, required=True)
    parser.add_argument('--timestamps', type=str, required=True)
    parser.add_argument('--output', type=str, default="video_with_behaviors.mp4")
    parser.add_argument('--video_start_time_sec', type=float, default=None)
    parser.add_argument('--video_end_time_sec', type=float, default=None)
    parser.add_argument('--export_csv', type=str, default="gaze_head_behavior.csv")
    args = parser.parse_args()

    processor = GazeIMUProcessor(args)
    # processor.generate_plots()
    # processor.overlay_video()
    processor.export_csv(args.export_csv)


if __name__ == "__main__":
    main()
