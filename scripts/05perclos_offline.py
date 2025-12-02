import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# --- Parameters ---
BASELINE_WINDOW = 5  # seconds for baseline calibration
ROLLING_WINDOW = 10  # seconds for rolling PERCLOS
# MAX_HISTORY = 60     # seconds to show in plot

def prune_buffer_by_time(
    buffer: deque,
    current_ts: float,
    window: float,
) -> None:
    """Removes items from the left of the buffer that are outside the time window."""
    while buffer and current_ts - buffer[0][0] > window:
        buffer.popleft()

# def main(csv_path, baseline_window=BASELINE_WINDOW, rolling_window=ROLLING_WINDOW, max_history=MAX_HISTORY):
def main(csv_path, baseline_window=BASELINE_WINDOW, rolling_window=ROLLING_WINDOW):
    # Load CSV
    df = pd.read_csv(csv_path)

    if not set(['timestamp [ns]', 'eyelid aperture left [mm]', 'eyelid aperture right [mm]']).issubset(df.columns):
        raise ValueError('CSV must contain timestamp [ns], eyelid aperture left [mm], eyelid aperture right [mm] columns')

    timestamps = df['timestamp [ns]'].values
    al = df['eyelid aperture left [mm]'].values
    ar = df['eyelid aperture right [mm]'].values

    # Convert timestamps from nanoseconds to seconds
    timestamps = timestamps / 1e9

    # Only use the first 5 minutes (300 seconds) of data
    max_duration = 300  # seconds
    elapsed_time_full = (timestamps - timestamps[0])
    use_mask = elapsed_time_full <= max_duration
    timestamps = timestamps[use_mask]
    al = al[use_mask]
    ar = ar[use_mask]

    # Baseline calibration: use first baseline_window seconds
    start_ts = timestamps[0]
    baseline_mask = (timestamps - start_ts) < baseline_window
    if not np.any(baseline_mask):
        raise ValueError('Not enough data for baseline calibration')
    baseline_left = np.percentile(al[baseline_mask], 95)
    baseline_right = np.percentile(ar[baseline_mask], 95)
    if baseline_left <= 3 or baseline_right <= 3:
        print('Warning: Eyes may have been closed during calibration. Baseline may be inaccurate.')
        baseline_left = max(baseline_left, 0.01)
        baseline_right = max(baseline_right, 0.01)

    # Calculate percent closed
    pc_left, pc_right = map(
                lambda ap, baseline: (
                    100 * (1 - ap / baseline) if baseline > 0 else 100.0
                ),
                [al, ar],
                [baseline_left, baseline_right],
            )
    percent_closed: float = np.clip((pc_left + pc_right) / 2, 0, 100)
    # pc_left = 100 * (1 - al / baseline_left)
    # pc_right = 100 * (1 - ar / baseline_right)
    # percent_closed = np.clip((pc_left + pc_right) / 2, 0, 100)

    # Calculate PERCLOS (percent of time percent_closed >= 80 in rolling window)
    perclos = np.zeros_like(percent_closed)
    for i in range(len(timestamps)):
        window_start = timestamps[i] - rolling_window
        mask = (timestamps >= window_start) & (timestamps <= timestamps[i])
        perclos[i] = np.mean(percent_closed[mask] >= 80) * 100 if np.any(mask) else 0

    # Plot (show all results at once)
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    elapsed_time = timestamps - timestamps[0]
    ax1.plot(elapsed_time, perclos, lw=2, color="red", label="PERCLOS")
    ax1.plot(elapsed_time, percent_closed, lw=1, color="blue", alpha=0.5, label="Percent Closed")
    ax1.axhline(80, color="gray", linestyle="--", alpha=0.7, label="80% Closure Threshold")
    ax1.set_ylim(0, 101)
    # ax1.set_xlim(0, min(max_history, elapsed_time[-1]))
    ax1.set_xlim(0, elapsed_time[-1])
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Percent (%)")
    ax1.set_title("Offline Eye Closure & PERCLOS")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(pad=2.0)
    plt.savefig("perclos_offline_plot.png", dpi=300)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline PERCLOS calculation and plot from CSV.")
    parser.add_argument("csv_path", type=str, help="Path to CSV file with timestamp, eyelid_aperture_left, eyelid_aperture_right columns.")
    parser.add_argument("--baseline-window", type=float, default=BASELINE_WINDOW, help="Seconds for baseline calibration.")
    parser.add_argument("--rolling-window", type=float, default=ROLLING_WINDOW, help="Seconds for rolling PERCLOS window.")
    # parser.add_argument("--max-history", type=float, default=MAX_HISTORY, help="Seconds to show in plot.")
    args = parser.parse_args()
    # main(args.csv_path, args.baseline_window, args.rolling_window, args.max_history)
    main(args.csv_path, args.baseline_window, args.rolling_window)
