import pandas as pd
import numpy as np
import argparse
import json
from scipy.spatial import cKDTree

class TrajectoryMerger:
    def __init__(self, tum_path):
        self.traj_df = self._load_tum_to_df(tum_path)

    @staticmethod
    def _load_tum_to_df(tum_file):
        df = pd.read_csv(tum_file, delim_whitespace=True, header=None)
        df.columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        df['timestamp_ns'] = (df['timestamp'] * 1e9).astype(np.int64)
        return df

    @staticmethod
    def _load_behavior_csv(csv_file):
        df = pd.read_csv(csv_file)
        df['timestamp_ns'] = df['timestamp_ns'].astype(np.int64)
        return df

    @staticmethod
    def _load_json_labels(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['timestamp_ns'] = df['timestamp_ns'].astype(np.int64)
        return df[['timestamp_ns', 'label', 'confidence']]

    @staticmethod
    def _nearest_join(source_df, ref_df, columns_to_copy, max_diff_ns=np.inf):
        tree = cKDTree(ref_df['timestamp_ns'].values.reshape(-1, 1))
        distances, indices = tree.query(source_df['timestamp_ns'].values.reshape(-1, 1), k=1)

        matched = ref_df.iloc[indices].reset_index(drop=True)
        for col in columns_to_copy:
            matched.loc[distances > max_diff_ns, col] = np.nan

        return matched[columns_to_copy]

    def fuse_behavior(self, behavior_path, max_diff_ms=100):
        behavior_df = self._load_behavior_csv(behavior_path)
        matched = self._nearest_join(self.traj_df, behavior_df,
                                     ['gaze_behavior', 'head_behavior'],
                                     max_diff_ns=max_diff_ms * 1e6)
        self.traj_df['gaze_behavior'] = matched['gaze_behavior']
        self.traj_df['head_behavior'] = matched['head_behavior']

    def fuse_labels(self, json_path, max_diff_ms=100):
        label_df = self._load_json_labels(json_path)
        matched = self._nearest_join(self.traj_df, label_df,
                                     ['label', 'confidence'],
                                     max_diff_ns=max_diff_ms * 1e6)
        self.traj_df['label'] = matched['label']
        self.traj_df['confidence'] = matched['confidence']

    def save(self, path):
        columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        optional = ['gaze_behavior', 'head_behavior', 'label', 'confidence']
        for col in optional:
            if col in self.traj_df.columns:
                columns.append(col)
        self.traj_df[columns].to_csv(path, index=False)
        print(f"✅ Merged CSV saved: {path}")

def main():
    parser = argparse.ArgumentParser(description="Flexible TUM trajectory merger.")
    parser.add_argument('--traj', required=True, help="TUM trajectory file")
    parser.add_argument('--behavior', help="CSV file with gaze/head behavior")
    parser.add_argument('--json_labels', help="JSON label file with timestamps")
    parser.add_argument('--output', required=True, help="Output merged CSV")
    parser.add_argument('--max_diff_ms', type=float, default=100.0, help="Max match time diff (ms)")

    args = parser.parse_args()

    merger = TrajectoryMerger(args.traj)

    if args.behavior:
        merger.fuse_behavior(args.behavior, args.max_diff_ms)

    if args.json_labels:
        merger.fuse_labels(args.json_labels, args.max_diff_ms)

    merger.save(args.output)
