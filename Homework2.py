import numpy as np
import spectral as spy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from skimage.io import imread
import random

########################################################

# Problem 1
# a
hdr_file = "tait_hsi.hdr"
image = spy.open_image(hdr_file)

data = image.load()

blue_band = 13
green_band = 69
red_band = 110
nir_band = 250

# Plot individual bands
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(data[:, :, blue_band], cmap='Blues')
axs[0].set_title("Blue Band")
axs[1].imshow(data[:, :, green_band], cmap='Greens')
axs[1].set_title("Green Band")
axs[2].imshow(data[:, :, red_band], cmap='Reds')
axs[2].set_title("Red Band")

plt.show()

# Plot pseudocolor
pseudocolor = np.dstack((data[:, :, green_band], data[:, :, red_band], data[:, :, nir_band]))

plt.figure(figsize=(6, 6))
plt.imshow(pseudocolor / pseudocolor.max())
plt.title("Pseudocolor Image (Green, Red, NIR)")
plt.show()


print(image)
print("Data shape:", data.shape)

#-----------------------------------------------------------------------
#b
def correlation_matrix(args):

    bands_flattened = data.reshape(-1, data.shape[-1])
    corr_matrix = np.corrcoef(bands_flattened, rowvar=False)

    return corr_matrix

corr_matrix = correlation_matrix(data)
# plotting

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='gray', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.title("Correlation Matrix of Sentinel-2 Bands")
plt.xlabel("Bands")
plt.ylabel("Bands")

plt.show()

########################################################################################

#Problem 2
# a

def principal_component_analysis(data):

    # mean centering
    mean_arr = np.mean(data, axis=0)
    centered_data = data - mean_arr

    # standardization (Z-score normalization)
    std_devs = np.std(data, axis=0)
    standardized_data = centered_data / std_devs

    # SVD
    U, S, Vt = np.linalg.svd(standardized_data, full_matrices=False)

    # principle components from transpose
    pcs = Vt.T

    # eigenvalues
    eigenvalues = (S ** 2) / (data.shape[0] - 1)



    return pcs, eigenvalues, standardized_data

#---------------------------------------------------------------------
# b

rows, cols, bands = data.shape
reshaped_data = np.reshape(data, (rows * cols, bands))
print(reshaped_data.shape)

pcs, eigenvalues, standardized_data = principal_component_analysis(reshaped_data)

first_10_pcs = pcs[:, :10]
transformed_data = np.dot(standardized_data, first_10_pcs)

transformed_image = transformed_data.reshape((rows, cols, 10))

# plotting
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i in range(10):
    axes[i].imshow(transformed_image[:, :, i], cmap='gray')
    axes[i].set_title(f"Principal Component {i + 1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

#------------------------------------------------------------------
# c

def calculate_reconstruction_error(centered_data, pcs, num_components):

    selected_pcs = pcs[:, :num_components]
    projected_data = np.dot(centered_data, selected_pcs)
    transposed_data = np.dot(projected_data, selected_pcs.T)

    # mean squared L2 distance
    errors = np.linalg.norm(centered_data - transposed_data, axis=1) ** 2
    mean_error = np.mean(errors)
    return mean_error

# calucultaing errors for 1, 10, 50, 100 components
num_components_list = [1, 10, 50, 100, bands]  # Selected numbers of components
errors = [calculate_reconstruction_error(standardized_data, pcs, n) for n in num_components_list]

# plotting
plt.figure(figsize=(8, 6))
plt.plot(num_components_list, errors, marker='o', linestyle='--', color='b')
plt.title("Mean Reconstruction Error vs. Number of Principal Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Mean Reconstruction Error (L2 Distance)")
plt.grid(True)
plt.show()

#---------------------------------------------------------------------
# d

explained_variance_ratio = eigenvalues / np.sum(eigenvalues)


cumulative_variance = np.cumsum(explained_variance_ratio)
num_components_to_retain =  np.argmax(cumulative_variance >= 0.99) + 1
modified_pcs = pcs[:, :num_components_to_retain]


projected_data = np.dot(standardized_data, modified_pcs)
reconstructed_data = np.dot(projected_data, modified_pcs.T)
reconstructed_image = reconstructed_data.reshape((rows, cols, 273))

# selecting 'interesting pixels'
selected_pixel_indices = [(74, 426), (189, 411), (711, 599), (681, 683), (1000, 745)] #red, blue, yellow, green, brown targets
original_spectral_signatures = [data[x, y, :].flatten() for x, y in selected_pixel_indices]
reconstructed_spectral_signatures = [reconstructed_image[x, y, :].flatten() for x, y in selected_pixel_indices]

plt.figure(figsize=(10, 8))
for i, (original, reconstructed) in enumerate(zip(original_spectral_signatures, reconstructed_spectral_signatures)):
    plt.plot(original, label=f"Original Pixel {i + 1}", linestyle='-')
    plt.plot(reconstructed , label=f"Reconstructed Pixel {i + 1}", linestyle='--')
plt.title("Comparison of Original and Reconstructed Spectral Signatures")
plt.xlabel("Band")
plt.ylabel("Spectral Signal")
plt.ylim(0, 12)
plt.legend()
plt.grid(True)
plt.show()


def calculate_snr(data):
    mean_signal = np.mean(data[data > 0], axis=0)
    std_signal = np.std(data[data > 0], axis=0)
    snr = mean_signal * 100/ std_signal
    return np.mean(snr)  # Average SNR across all bands

snr_before = calculate_snr(reshaped_data)
snr_after = calculate_snr(reconstructed_data)

print(f"Signal-to-Noise Ratio (SNR) Before Transformation: {snr_before:.2f}")
print(f"Signal-to-Noise Ratio (SNR) After Transformation: {snr_after:.2f}")


###################################################################################
# Problem 3
# a

image = imread("jellybeans.tiff")
image_shape = image.shape
data_jb = image.reshape((-1, 3))  # Flatten image for clustering

# Standardize
scaler = StandardScaler()
standardized_data_jb = scaler.fit_transform(data_jb)

def k_means_clustering(data, k, max_iters=100, tol=1e-4):

    # K random cluster centers
    random_indices = random.sample(range(data.shape[0]), k)
    cluster_centers = data[random_indices]

    for _ in range(max_iters):

        distances = np.linalg.norm(data[:, np.newaxis] - cluster_centers, axis=2)
        labels = np.argmin(distances, axis=1)

        # Compute new centroids
        new_centers = np.array([data[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.linalg.norm(new_centers - cluster_centers) < tol:
            break

        cluster_centers = new_centers

    return cluster_centers, labels


k = 6
cluster_centers, labels = k_means_clustering(standardized_data_jb, k)


reconstructed_image = cluster_centers[labels].reshape(image_shape)

# Rescale
reconstructed_image = scaler.inverse_transform(reconstructed_image.reshape(-1, 3)).reshape(image_shape)
reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)  # Ensure pixel values are valid

cmaps = ['magenta','yellow', 'blue', 'green', 'red', 'cyan', 'white']

plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# Clustered (reconstructed) image
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap=mcolors.ListedColormap(cmaps[:k]))
plt.title(f"Clustered Image with K={k}")
plt.axis('off')

plt.tight_layout()
plt.show()

#-------------------------------------------------------------------
# b


data_st = np.load('sentinel2_rochester.npy')

# Flattening
rows_st, cols_st, bands_st = data_st.shape
reshaped_data_st = data_st.reshape((-1, bands_st))

# PCS for
pcs_st, eigenvalues_st, standardized_data_st = principal_component_analysis(reshaped_data_st)

def transform_data(standardized_data, pcs_st, num_components):
    selected_pcs_st = pcs_st[:, :num_components]
    transformed_data_st = np.dot(standardized_data, selected_pcs_st)
    return transformed_data_st

transformed_data_3 = transform_data(standardized_data_st, pcs_st, 3)
transformed_data_4 = transform_data(standardized_data_st, pcs_st, 4)
transformed_data_5 = transform_data(standardized_data_st, pcs_st, 5)
transformed_data_6 = transform_data(standardized_data_st, pcs_st, 6)

k = 4

cluster_centers_3, labels_3 = k_means_clustering(transformed_data_3, k)
cluster_centers_4, labels_4 = k_means_clustering(transformed_data_4, k)
cluster_centers_5, labels_5 = k_means_clustering(transformed_data_5, k)
cluster_centers_6, labels_6 = k_means_clustering(transformed_data_6, k)

labels_3_image = labels_3.reshape((rows_st, cols_st))
labels_4_image = labels_4.reshape((rows_st, cols_st))
labels_5_image = labels_5.reshape((rows_st, cols_st))
labels_6_image = labels_6.reshape((rows_st, cols_st))


plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(labels_3_image, cmap=mcolors.ListedColormap(cmaps[:k]))
plt.title("K-Means Clustering with 3 Principal Components")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(labels_4_image, cmap=mcolors.ListedColormap(cmaps[:k]))
plt.title("K-Means Clustering with 4 Principal Components")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(labels_5_image, cmap=mcolors.ListedColormap(cmaps[:k]))
plt.title("K-Means Clustering with 5 Principal Components")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(labels_6_image, cmap=mcolors.ListedColormap(cmaps[:k]))
plt.title("K-Means Clustering with 6 Principal Components")
plt.axis('off')

plt.show()

#--------------------------------------------------------------------
# c

x, y, width, height = 600, 400, 250, 250  # Example ROI starting at (20, 30) with 40x40 size

# Step 4: Extract the ROI from the data
roi = data[y:y + height, x:x + width, :]

print(roi.shape)

rows, cols, bands = roi.shape
reshaped_roi = np.reshape(roi, (rows * cols, bands))


pcs_roi, eigenvalues_roi, standardized_roi = principal_component_analysis(reshaped_roi)

transformed_roi_2 = transform_data(standardized_roi, pcs_roi, 2)
transformed_roi_5 = transform_data(standardized_roi, pcs_roi, 5)
transformed_roi_10 = transform_data(standardized_roi, pcs_roi, 10)
transformed_roi_50 = transform_data(standardized_roi, pcs_roi, 50)
transformed_roi_100 = transform_data(standardized_roi, pcs_roi, 100)

print(f"Transformed Data Shape (2 components): {transformed_roi_2.shape}")
print(f"Transformed Data Shape (5 components): {transformed_roi_5.shape}")
print(f"Transformed Data Shape (10 components): {transformed_roi_10.shape}")
print(f"Transformed Data Shape (50 components): {transformed_roi_50.shape}")
print(f"Transformed Data Shape (100 components): {transformed_roi_100.shape}")

k = 4

# original
cluster_centers_original, labels_original = k_means_clustering(standardized_roi, k)

# reduced
cluster_centers_2, labels_2 = k_means_clustering(transformed_roi_2, k)
cluster_centers_5, labels_5 = k_means_clustering(transformed_roi_5, k)
cluster_centers_10, labels_10 = k_means_clustering(transformed_roi_10, k)
cluster_centers_50, labels_50 = k_means_clustering(transformed_roi_50, k)
cluster_centers_100, labels_100 = k_means_clustering(transformed_roi_100, k)

labels_original_image = labels_original.reshape((rows, cols))
labels_2_image = labels_2.reshape((rows, cols))
labels_5_image = labels_5.reshape((rows, cols))
labels_10_image = labels_10.reshape((rows, cols))
labels_50_image = labels_50.reshape((rows, cols))
labels_100_image = labels_100.reshape((rows, cols))

plt.figure(figsize=(30, 15))
plt.rcParams.update({'font.size': 18})

plt.subplot(2, 3, 1)
plt.imshow(labels_2_image, cmap=mcolors.ListedColormap(cmaps[:k]))
plt.title("K-Means Clustering with 2 Principal Components")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(labels_5_image, cmap=mcolors.ListedColormap(cmaps[:k]))
plt.title("K-Means Clustering with 5 Principal Components")
plt.axis('off')


plt.subplot(2, 3, 3)
plt.imshow(labels_10_image, cmap=mcolors.ListedColormap(cmaps[:k]))
plt.title("K-Means Clustering with 10 Principal Components")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(labels_50_image, cmap=mcolors.ListedColormap(cmaps[:k]))
plt.title("K-Means Clustering with 50 Principal Components")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(labels_100_image, cmap=mcolors.ListedColormap(cmaps[:k]))
plt.title("K-Means Clustering with 100 Principal Components")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(labels_original_image, cmap=mcolors.ListedColormap(cmaps[:k]))
plt.title("Original Image")
plt.axis('off')

#plt.tight_layout()
plt.show()





