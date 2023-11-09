import numpy as np
import pandas as pd
from numpy.linalg import svd
import matplotlib.pyplot as plt


def load_data(file_name):
    return pd.read_csv(file_name, header=None).values

def buggy_pca(X, d):
    U, S, VT = svd(X, full_matrices=False)
    Z = np.dot(X, VT.T[:, :d])
    reconstruction = np.dot(Z, VT[:d, :])
    return Z, VT, reconstruction

def demeaned_pca(X, d):
    mean_X = np.mean(X, axis=0)
    demeaned_X = X - mean_X
    Z, VT, reconstruction = buggy_pca(demeaned_X, d)
    reconstruction += mean_X
    return Z, VT, reconstruction

def normalized_pca(X, d):
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    normalized_X = (X - mean_X) / std_X
    Z, VT, reconstruction = buggy_pca(normalized_X, d)
    reconstruction = reconstruction * std_X + mean_X
    return Z, VT, reconstruction

def dro(data, d):
    num_samples, original_dimension = data.shape
    
    offset = np.mean(data, axis=0)
    centered_data = data - offset
    
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
    
    U_reduced = U[:, :d]
    singular_values_reduced = S[:d]
    Vt_reduced = Vt[:d, :]
    A_transposed = np.dot(np.diag(singular_values_reduced), Vt_reduced)
    reduced_representations = U_reduced
    reconstructions = np.dot(reduced_representations, A_transposed) + offset
    
    return reduced_representations, A_transposed, U_reduced, reconstructions, np.diag(singular_values_reduced)

def reconstruction_error(X, reconstructed_X):
    return np.sum((X - reconstructed_X)**2)

data2D = load_data('data/data2D.csv')
data1000D = load_data('data/data1000D.csv')

chosen_d = 31

Z_buggy, _, rec_buggy = buggy_pca(data2D, d=1)
Z_demeaned, _, rec_demeaned = demeaned_pca(data2D, d=1)
Z_normalized, _, rec_normalized = normalized_pca(data2D, d=1)
d_dim_representations, At, Z, rec_dro, S_diag = dro(data2D, 1)

error_buggy = reconstruction_error(data2D, rec_buggy)
error_demeaned = reconstruction_error(data2D, rec_demeaned)
error_normalized = reconstruction_error(data2D, rec_normalized)
error_dro = reconstruction_error(data2D, rec_dro)

U, S, VT = svd(data1000D, full_matrices=False)

plt.figure(figsize=(10, 5))
plt.plot(S, 'o-', markersize=4, label='Singular Values')
plt.title('Singular Value Spectrum (1000D Dataset)')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig("2.3.kneePoint.png")


Z_buggy_1000D, _, rec_buggy_1000D = buggy_pca(data1000D, chosen_d)
Z_demeaned_1000D, _, rec_demeaned_1000D = demeaned_pca(data1000D, chosen_d)
Z_normalized_1000D, _, rec_normalized_1000D = normalized_pca(data1000D, chosen_d)
d_dim_representations, At, Z, rec_dro_1000D, S_diag = dro(data1000D, chosen_d)

error_buggy_1000D = reconstruction_error(data1000D, rec_buggy_1000D)
error_demeaned_1000D = reconstruction_error(data1000D, rec_demeaned_1000D)
error_normalized_1000D = reconstruction_error(data1000D, rec_normalized_1000D)
error_dro_1000D = reconstruction_error(data1000D, rec_dro_1000D)

print(f"Reconstruction error for Buggy PCA (2D): {error_buggy}")
print(f"Reconstruction error for Demeaned PCA (2D): {error_demeaned}")
print(f"Reconstruction error for Normalized PCA (2D): {error_normalized}")
print(f"Reconstruction error for DRO (2D): {error_dro}")

print(f"Reconstruction error for Buggy PCA (1000D): {error_buggy_1000D}")
print(f"Reconstruction error for Demeaned PCA (1000D): {error_demeaned_1000D}")
print(f"Reconstruction error for Normalized PCA (1000D): {error_normalized_1000D}")
print(f"Reconstruction error for DRO (1000D): {error_dro_1000D}")

fig, axs = plt.subplots(2, 2, figsize=(18, 12))

axs[0, 0].scatter(data2D[:, 0], data2D[:, 1], color='blue', label='Original Data')
axs[0, 0].scatter(rec_buggy[:, 0], rec_buggy[:, 1], color='red', label='Buggy PCA Reconstruction')
axs[0, 0].set_title('Buggy PCA Reconstruction')

axs[0, 1].scatter(data2D[:, 0], data2D[:, 1], color='blue', label='Original Data')
axs[0, 1].scatter(rec_demeaned[:, 0], rec_demeaned[:, 1], color='green', label='Demeaned PCA Reconstruction')
axs[0, 1].set_title('Demeaned PCA Reconstruction')

axs[1, 0].scatter(data2D[:, 0], data2D[:, 1], color='blue', label='Original Data')
axs[1, 0].scatter(rec_normalized[:, 0], rec_normalized[:, 1], color='purple', label='Normalized PCA Reconstruction')
axs[1, 0].set_title('Normalized PCA Reconstruction')

axs[1, 1].scatter(data2D[:, 0], data2D[:, 1], color='blue', label='Original Data')
axs[1, 1].scatter(rec_dro[:, 0], rec_dro[:, 1], color='orange', label='DRO Reconstruction')
axs[1, 1].set_title('DRO Reconstruction')

for ax in axs.flat:
    if not ax.axis()[0] == 0.0:
        ax.set(xlabel='X-axis', ylabel='Y-axis')
        ax.label_outer()
        ax.legend()

plt.tight_layout()
plt.savefig("2.3.png")
plt.show()