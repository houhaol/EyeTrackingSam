import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11

# Load the CSV file
csv_path = "/home/houhao/workspace/EyeTrackingSam/data/BF004/seg_fixation_02.csv"
df = pd.read_csv(csv_path)

# Count occurrences and percentage of each voted_label_name (top 10)
label_counts = df['voted_label_name'].value_counts().head(10)
label_percent = df['voted_label_name'].value_counts(normalize=True).head(10) * 100
total_fixations = len(df)

# Create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # --- Plot 1: Bar chart with counts ---
# colors = sns.color_palette("husl", len(label_counts))
# bars = ax1.bar(range(len(label_counts)), label_counts.values, color=colors, 
#                edgecolor='black', linewidth=1.2, alpha=0.85)

# ax1.set_xlabel('Object Name', fontsize=13, fontweight='bold')
# ax1.set_ylabel('Fixation Count', fontsize=13, fontweight='bold')
# ax1.set_title('Top 10 Objects from Eye Fixations', fontsize=15, fontweight='bold', pad=20)
# ax1.set_xticks(range(len(label_counts)))
# ax1.set_xticklabels(label_counts.index, rotation=45, ha='right', fontsize=11)
# ax1.grid(axis='y', alpha=0.3, linestyle='--')
# ax1.set_axisbelow(True)

# # Add value labels on bars (count and percentage)
# for i, (count, percent) in enumerate(zip(label_counts.values, label_percent.values)):
#     ax1.text(i, count + max(label_counts) * 0.01, 
#              f'{int(count)}\n({percent:.1f}%)', 
#              ha='center', va='bottom', fontsize=10, fontweight='bold')

# # Add total fixations info
# ax1.text(0.02, 0.98, f'Total Fixations: {total_fixations}', 
#          transform=ax1.transAxes, fontsize=11, verticalalignment='top',
#          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- Plot 2: Horizontal bar chart with percentage ---
fig, ax2 = plt.subplots(1, 1, figsize=(14, 7))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(label_counts)))
ax2.set_xlim(0, max(label_percent.values) * 1.08)
ax2.barh(range(len(label_counts)), label_percent.values, color=colors,
         edgecolor='black', linewidth=1.2, alpha=0.85)

ax2.set_xlabel('Percentage (%)', fontsize=30)
ax2.set_ylabel('Object Name', fontsize=30)
ax2.set_title('Quantification of Top 10 Objects from Fixations', fontsize=30, pad=20)
ax2.set_yticks(range(len(label_counts)))
ax2.set_yticklabels(label_counts.index, fontsize=18, rotation=15)
ax2.tick_params(axis='x', labelsize=18)
ax2.invert_yaxis()  # Highest at top
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# Add percentage labels at the end of bars
for i, (percent, count) in enumerate(zip(label_percent.values, label_counts.values)):
    ax2.text(percent + 0.3, i, f' {percent:.1f}% \n(n={int(count)})', 
             va='center', fontsize=14)

plt.tight_layout()
plt.savefig('/home/houhao/workspace/EyeTrackingSam/data/BF004/voted_label_quantification.png', 
            dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: /home/houhao/workspace/EyeTrackingSam/data/BF004/voted_label_quantification.png")
plt.show()
