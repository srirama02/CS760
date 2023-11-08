import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Load the data from a CSV file
def load_data(file_name):
    return pd.read_csv(file_name, header=None).values

# Buggy PCA
def buggy_pca(X, d):
    U, S, VT = svd(X, full_matrices=False)
    Z = np.dot(X, VT.T[:, :d])
    reconstruction = np.dot(Z, VT[:d, :])
    return Z, VT, reconstruction

# Demeaned PCA
def demeaned_pca(X, d):
    mean_X = np.mean(X, axis=0)
    demeaned_X = X - mean_X
    Z, VT, reconstruction = buggy_pca(demeaned_X, d)
    reconstruction += mean_X
    return Z, VT, reconstruction

# Normalized PCA
def normalized_pca(X, d):
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    normalized_X = (X - mean_X) / std_X
    Z, VT, reconstruction = buggy_pca(normalized_X, d)
    reconstruction = reconstruction * std_X + mean_X
    return Z, VT, reconstruction

# DRO Implementation
def dro(X, d):
    n, D = X.shape
    one_n = np.ones((n, 1))

    # Objective function for DRO
    def objective(params):
        A = params[:D * d].reshape(D, d)
        b = params[D * d:D * d + D]
        Z = params[D * d + D:].reshape(n, d)
        return np.linalg.norm(X - Z @ A.T - one_n @ b.reshape(1, D), 'fro')**2 / n

    # Initial guess for A, b, and Z
    A_init = np.random.rand(D, d)
    b_init = np.random.rand(D)
    Z_init = np.random.rand(n, d)
    params_init = np.concatenate([A_init.ravel(), b_init, Z_init.ravel()])

    # Minimize the objective function
    result = minimize(objective, params_init, method='L-BFGS-B')

    # Extract the optimal values for A, b, and Z
    A_opt = result.x[:D * d].reshape(D, d)
    b_opt = result.x[D * d:D * d + D]
    Z_opt = result.x[D * d + D:].reshape(n, d)
    reconstruction = Z_opt @ A_opt.T + one_n @ b_opt.reshape(1, D)
    return Z_opt, (A_opt, b_opt), reconstruction

# Reconstruction error calculation
def reconstruction_error(X, reconstructed_X):
    return np.sum((X - reconstructed_X)**2)

# Load datasets
data2D = load_data('data/data2D.csv')
data1000D = load_data('data/data1000D.csv')

# Choose 'd' for the 1000D dataset
# For the sake of this example, let's choose d=3, but you should choose it based on the singular values analysis.
chosen_d = 3

# Apply methods on 2D dataset
Z_buggy, _, rec_buggy = buggy_pca(data2D, d=1)
Z_demeaned, _, rec_demeaned = demeaned_pca(data2D, d=1)
Z_normalized, _, rec_normalized = normalized_pca(data2D, d=1)
Z_dro, params_dro, rec_dro = dro(data2D, d=1)

# Calculate reconstruction errors for 2D dataset
error_buggy = reconstruction_error(data2D, rec_buggy)
error_demeaned = reconstruction_error(data2D, rec_demeaned)
error_normalized = reconstruction_error(data2D, rec_normalized)
error_dro = reconstruction_error(data2D, rec_dro)

# Apply methods on 1000D dataset
Z_buggy_1000D, _, rec_buggy_1000D = buggy_pca(data1000D, chosen_d)
Z_demeaned_1000D, _, rec_demeaned_1000D = demeaned_pca(data1000D, chosen_d)
Z_normalized_1000D, _, rec_normalized_1000D = normalized_pca(data1000D, chosen_d)
Z_dro_1000D, params_dro_1000D, rec_dro_1000D = dro(data1000D, chosen_d)

# Calculate reconstruction errors for 1000D dataset
error_buggy_1000D = reconstruction_error(data1000D, rec_buggy_1000D)
error_demeaned_1000D = reconstruction_error(data1000D, rec_demeaned_1000D)
error_normalized_1000D = reconstruction_error(data1000D, rec_normalized_1000D)
error_dro_1000D = reconstruction_error(data1000D, rec_dro_1000D)

# Print reconstruction errors
print(f"Reconstruction error for Buggy PCA (1000D): {error_buggy_1000D}")
print(f"Reconstruction error for Demeaned PCA (1000D): {error_demeaned_1000D}")
print(f"Reconstruction error for Normalized PCA (1000D): {error_normalized_1000D}")
print(f"Reconstruction error for DRO (1000D): {error_dro_1000D}")
# Applying DRO on the 2D dataset again
Z_dro, params_dro, rec_dro = dro(data2D, chosen_d)

# Visualization for 2D dataset including DRO
fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Adjusted for an additional method

# Original points
axs[0, 0].scatter(data2D[:, 0], data2D[:, 1], color='blue', label='Original Data')
axs[0, 0].set_title('Original Data')

# Buggy PCA reconstruction
axs[0, 1].scatter(data2D[:, 0], data2D[:, 1], color='blue', label='Original Data')
axs[0, 1].scatter(rec_buggy[:, 0], rec_buggy[:, 1], color='red', label='Buggy PCA Reconstruction')
axs[0, 1].set_title('Buggy PCA Reconstruction')

# Demeaned PCA reconstruction
axs[0, 2].scatter(data2D[:, 0], data2D[:, 1], color='blue', label='Original Data')
axs[0, 2].scatter(rec_demeaned[:, 0], rec_demeaned[:, 1], color='green', label='Demeaned PCA Reconstruction')
axs[0, 2].set_title('Demeaned PCA Reconstruction')

# Normalized PCA reconstruction
axs[1, 0].scatter(data2D[:, 0], data2D[:, 1], color='blue', label='Original Data')
axs[1, 0].scatter(rec_normalized[:, 0], rec_normalized[:, 1], color='purple', label='Normalized PCA Reconstruction')
axs[1, 0].set_title('Normalized PCA Reconstruction')

# DRO reconstruction
axs[1, 1].scatter(data2D[:, 0], data2D[:, 1], color='blue', label='Original Data')
axs[1, 1].scatter(rec_dro[:, 0], rec_dro[:, 1], color='orange', label='DRO Reconstruction')
axs[1, 1].set_title('DRO Reconstruction')

# Hide the last subplot (unused)
axs[1, 2].axis('off')

# Set labels and titles for each subplot
for ax in axs.flat:
    if not ax.axis()[0] == 0.0:  # Only set labels for non-hidden subplots
        ax.set(xlabel='X-axis', ylabel='Y-axis')
        ax.label_outer()
        ax.legend()

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()
# # Visualization for 2D dataset
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# # Original points
# axs[0, 0].scatter(data2D[:, 0], data2D[:, 1], color='blue', label='Original Data')
# axs[0, 0].set_title('Original Data')

# # Buggy PCA reconstruction
# axs[0, 1].scatter(rec_buggy[:, 0], rec_buggy[:, 1], color='red', label='Buggy PCA Reconstruction')
# axs[0, 1].set_title('Buggy PCA Reconstruction')

# # Demeaned PCA reconstruction
# axs[1, 0].scatter(rec_demeaned[:, 0], rec_demeaned[:, 1], color='green', label='Demeaned PCA Reconstruction')
# axs[1, 0].set_title('Demeaned PCA Reconstruction')

# # Normalized PCA reconstruction
# axs[1, 1].scatter(rec_normalized[:, 0], rec_normalized[:, 1], color='purple', label='Normalized PCA Reconstruction')
# axs[1, 1].set_title('Normalized PCA Reconstruction')

# for ax in axs.flat:
#     ax.set(xlabel='X-axis', ylabel='Y-axis')
#     ax.label_outer()

# plt.legend()
# plt.show()