import numpy as np
import matplotlib.pyplot as plt


## Problem 1

# Loading data
data = np.load('sentinel2_rochester.npy')
print(data.shape)

# Labels for each band
band_labels = [
    "1 - Coastal Aerosol (443nm)",
    "2 - Blue (490nm)",
    "3 - Green (560nm)",
    "4 - Red (665nm)",
    "5 - Red Edge 1 (705nm)",
    "6 - Red Edge 2 (740nm)",
    "7 - Red Edge 3 (783nm)",
    "8 - NIR (842nm)",
    "8A - Narrow NIR (865nm)",
    "9 - Water Vapor (945nm)",
    "11 - SWIR 1 (1610nm)",
    "12 - SWIR 2 (2190nm)"
]

# Creating image montage
rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

# Feeding images to montage
for i in range(rows * cols):
    ax = axes[i // cols, i % cols]
    ax.imshow(data[:, :, i], cmap='gray')
    ax.set_title(f'Band {i+1}')
    ax.set_title(band_labels[i])
    ax.axis('off')

# Displaying montage
plt.tight_layout()
plt.show()


## Problem 2

def calculate_band_statistics(args):
    args_flat = args.flatten()

    stats = {
        'mean': np.mean(args_flat),  # Average value of the band
        'std': np.std(args_flat),  # Standard deviation (measure of spread)
        'min': np.min(args_flat),  # Minimum value in the band
        'max': np.max(args_flat),  # Maximum value in the band
        'Q1': np.percentile(args_flat, 25),  # First quartile (25th percentile)
        'median': np.median(args_flat),  # Median (50th percentile)
        'Q3': np.percentile(args_flat, 75),  # Third quartile (75th percentile)
        'skewness': skew(args_flat),  # Measure of asymmetry
        'kurtosis': kurtosis(args_flat)  # Measure of "tailedness"
    }
    return stats

stats = calculate_band_statistics(data[ :, :, 10])
print(stats)

def standardize(args):
    band_mean = np.mean(args)  # Mean of the band
    band_std = np.std(args)
    z_matrix = (args - band_mean) / band_std

    return z_matrix

z_scores = standardize(data[ :, :, 10])
print("z-score" , z_scores[400, 350])







print('done')

