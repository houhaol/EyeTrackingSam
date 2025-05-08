import pandas as pd
import numpy as np
import argparse
from scipy.spatial import cKDTree

def load_tum_to_df(tum_file):
    df = pd.read_csv(tum_file, delim_whitespace=True, header=None)
    df.columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    df['timestamp_ns'] = (df['timestamp'] * 1e9).astype(np.int64)
    return df

def load_behavior_csv(csv_file):
    return pd.read_csv(csv_file)

def merge_with_nearest(trajectory_df, behavior_df):
    # KD-tree nearest neighbor search
    tree = cKDTree(behavior_df['timestamp_ns'].values.reshape(-1, 1))
    distances, indices = tree.query(trajectory_df['timestamp_ns'].values.reshape(-1, 1), k=1)

    # Get matched rows
    matched_behaviors = behavior_df.iloc[indices].reset_index(drop=True)
    merged = trajectory_df.copy()
    merged['gaze_behavior'] = matched_behaviors['gaze_behavior']
    merged['head_behavior'] = matched_behaviors['head_behavior']
    return merged

def main():
    parser = argparse.ArgumentParser(description="Merge TUM trajectory with behavior annotations using nearest timestamp.")
    parser.add_argument('--traj', required=True, help="Path to TUM trajectory file")
    parser.add_argument('--behavior', required=True, help="Path to behavior CSV file")
    parser.add_argument('--output', required=True, help="Path to output merged CSV")

    args = parser.parse_args()

    traj_df = load_tum_to_df(args.traj)
    behavior_df = load_behavior_csv(args.behavior)
    merged_df = merge_with_nearest(traj_df, behavior_df)

    merged_df[['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw',
               'gaze_behavior', 'head_behavior']].to_csv(args.output, index=False)
    
    print(f"✅ Merged CSV saved: {args.output}")

if __name__ == "__main__":
    main()
