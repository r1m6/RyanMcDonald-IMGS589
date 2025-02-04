import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, gaussian_kde
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.interpolate import interp1d

#############################################################################################################
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
    ax.set_title(band_labels[i])
    ax.axis('off')

# Displaying montage
plt.tight_layout()
plt.show()

###############################################################################################
## Problem 2

# Band statistics
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

################################################################################################
# Standardize
def standardize(args):

    standardized_data = np.zeros_like(data)

    for band in range(data.shape[2]):
        band_data = data[:, :, band]
        band_mean = np.mean(band_data)
        band_std = np.std(band_data)

        # Standardize band
        standardized_data[:, :, band] = (band_data - band_mean) / band_std

    return standardized_data

z_scores = standardize(data)

# Histograms
def plot_histograms(data, standardized_data, band_labels):

    # Subplots
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

    for band in range(data.shape[2]):
        ax = axes[band // cols, band % cols]

        # Original data histogram
        ax.hist(data[:, :, band].flatten(), bins=50, alpha=0.5, label='Original', color='blue')

        # Standardized data histogram
        ax.hist(standardized_data[:, :, band].flatten(), bins=50, alpha=0.5, label='Standardized', color='red')

        # Outliers in standardized data
        outliers = standardized_data[:, :, band].flatten()
        outliers = outliers[(outliers > 3) | (outliers < -3)]  # Z-scores > 3 or < -3 are outliers
        if len(outliers) > 0:
            ax.hist(outliers, bins=50, alpha=0.5, label='Outliers', color='green')

        ax.set_title(band_labels[band])
        ax.legend()

    plt.tight_layout()
    plt.show()

plot_histograms(data, z_scores, band_labels)





################################################################################################
## Problem 3

#a


def correlation_matrix(args):

    bands_flattened = data.reshape(-1, data.shape[-1])
    corr_matrix = np.corrcoef(bands_flattened, rowvar=False)

    return corr_matrix

corr_matrix = correlation_matrix(data)

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='gray', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.title("Correlation Matrix of Sentinel-2 Bands")
plt.xlabel("Bands")
plt.ylabel("Bands")
plt.xticks(range(data.shape[-1]), range(1, data.shape[-1] + 1))
plt.yticks(range(data.shape[-1]), range(1, data.shape[-1] + 1))
plt.show()


############################################################################################
# b

# Pairplots

def correlation_plot(data, band_idx, band_label):

    band_data = [data[:, :, i].flatten() for i in band_idx]
    df = pd.DataFrame(dict(zip(band_label, band_data)))
    pairplot = sns.pairplot(df, diag_kind=None, plot_kws={"s": 1})
    pairplot.fig.suptitle("Pairwise Scatter Plots Between 10m Bands", y=1.02)
    plt.show()


band_idx_10m = [1, 2, 3, 7]
band_labels_10m = ["B2 - Blue", "B3 - Green", "B4 - Red", "B8 - NIR"]

correlation_plot(data, band_idx_10m, band_labels_10m)


############################################################################################
# Density
def pairwise_density_plots_statsmodels(data, band_idx, band_labels, sample_size=5000):
    num_bands = len(band_idx)
    fig, axes = plt.subplots(num_bands - 1, num_bands - 1, figsize=(12, 12), constrained_layout=True)



    # Loop through pairs of bands
    for i in range(num_bands - 1):
        for j in range(i + 1, num_bands):
            ax = axes[i, j - 1] if num_bands > 2 else axes  # Handle subplot layouts

            # Extract band data and sample
            band1 = data[:, :, band_idx[i]].flatten()
            band2 = data[:, :, band_idx[j]].flatten()
            sample_idx = np.random.choice(len(band1), size=sample_size, replace=False)
            band1_sample = band1[sample_idx]
            band2_sample = band2[sample_idx]

            # Scatter plot with KDE densities
            ax.scatter(band1_sample, band2_sample, s=2, alpha=0.5, label="Scatter Points")
            sns.kdeplot(x=band1_sample, y=band2_sample, ax=ax, cmap="viridis", fill=True, alpha=0.5)

            # axes and title
            ax.set_xlabel(band_labels[i])
            ax.set_ylabel(band_labels[j])
            ax.set_title(f"{band_labels[i]} vs {band_labels[j]}")

    plt.show()


pairwise_density_plots_statsmodels(data, band_idx_10m, band_labels_10m)

######################################################################################################

# Problem 4

Oakdf = pd.read_fwf("Oak.txt")  # Read fixed-width formatted file
Roaddf = pd.read_fwf("Road.txt")


oak_wavelengths = Oakdf.iloc[:, 0].values
oak_reflectance = Oakdf.iloc[:, 1].values / 100
road_wavelengths = Roaddf.iloc[:, 0].values
road_reflectance = Roaddf.iloc[:, 1].values / 100

# Omitting two sentinel bands
sentinel_wavelengths = np.array([0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 1.610, 2.190])

# Interpolate Oak and Road spectra to match Sentinel-2 bands
oak_interp = interp1d(oak_wavelengths, oak_reflectance, kind='linear', bounds_error=False, fill_value='extrapolate')
road_interp = interp1d(road_wavelengths, road_reflectance, kind='linear', bounds_error=False, fill_value='extrapolate')

oak_spectrum = oak_interp(sentinel_wavelengths)
road_spectrum = road_interp(sentinel_wavelengths)

data_adjusted = data[:, :, [1,2,3,4,5,6,7,8,10,11]]
# print(data_adjusted.shape)

# Flatten Sentinel-2 Data
pixels = data_adjusted.reshape(-1, data_adjusted.shape[-1])  # (954, 716, 10)


# Spectral angle mapper
def sam(v1, v2):

    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm_product)  # angle
    return angle


# Compute spectral angles for all pixels against Oak and Road
oak_angles = np.array([sam(pixel, oak_spectrum) for pixel in pixels])
road_angles = np.array([sam(pixel, road_spectrum) for pixel in pixels])

# indices of the 100 lowest spectral angles (best matches)
closest_oak_indices = np.argsort(oak_angles)[:100]
closest_road_indices = np.argsort(road_angles)[:100]

# Reshape indices back to 2D for visualization
oak_pixels = np.unravel_index(closest_oak_indices, (954, 716))
road_pixels = np.unravel_index(closest_road_indices, (954, 716))

# Plot the Spectra of 1st, 50th, and 100th closest matches for both Oak and Road
plt.figure(figsize=(12, 6))

for i, idx in enumerate([0, 49, 99]):
    oak_match = pixels[closest_oak_indices[idx]]
    road_match = pixels[closest_road_indices[idx]]

    plt.plot(sentinel_wavelengths, oak_match, label=f"Oak Match {idx + 1}", linestyle="-")
    plt.plot(sentinel_wavelengths, road_match, label=f"Road Match {idx + 1}", linestyle=":")

# Plot Reference Oak and Road Spectra
plt.plot(sentinel_wavelengths, oak_spectrum, label="Oak Spectrum (Reference)", linewidth=2, linestyle="-")
plt.plot(sentinel_wavelengths, road_spectrum, label="Road Spectrum (Reference)", linewidth=2, linestyle=":")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend()
plt.title("Comparison of Closest Matches with Oak and Road Spectra")
plt.show()

# Set cutoff angles for classification
threshold_oak = 0.45
classified_oak = oak_angles.reshape(954, 716) < threshold_oak  # Boolean mask for Oak
threshold_road = 0.4
classified_road = road_angles.reshape(954, 716) < threshold_road  # Boolean mask for Road

# Visualization of Oak Pixels
plt.figure(figsize=(8, 6))
plt.imshow(classified_oak, cmap="gray")
plt.colorbar(label="Oak Pixels (0: No, 1: Yes)")
plt.title("Identified Oak Pixels in Sentinel-2 Imagery")
plt.show()

# Visualization of Road Pixels
plt.figure(figsize=(8, 6))
plt.imshow(classified_road, cmap="gray")
plt.colorbar(label="Road Pixels (0: No, 1: Yes)")
plt.title("Identified Road Pixels in Sentinel-2 Imagery")
plt.show()





print('Done!')
