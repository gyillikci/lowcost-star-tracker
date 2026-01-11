"""Plot gyroscope data extracted from GoPro video."""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
from star_tracker.gyro_extractor import GyroExtractor
from pathlib import Path
import numpy as np

# Extract gyro data
extractor = GyroExtractor()
gyro_data = extractor.extract(Path('examples/GX010911.MP4'))

print(f"Extracted {gyro_data.num_samples} samples over {gyro_data.duration:.1f} seconds")

# Create plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

labels = ['X (roll)', 'Y (pitch)', 'Z (yaw)']
colors = ['red', 'green', 'blue']

for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
    ax.plot(gyro_data.timestamps, np.rad2deg(gyro_data.angular_velocity[:, i]), 
            color=color, linewidth=0.5)
    ax.set_ylabel(f'{label}\n(deg/s)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

axes[-1].set_xlabel('Time (seconds)')
axes[0].set_title('GoPro Gyroscope Data - GX010911.MP4')

plt.tight_layout()
plt.savefig('output/gyro_plot.png', dpi=150)
print('Plot saved to output/gyro_plot.png')
# plt.show()  # Commented out for non-interactive use
