import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.optimize import linear_sum_assignment

# Define the means and covariance matrices for the distributions
means = [np.array([-1, -1]), np.array([1, -1]), np.array([0, 1])]
covariances = [np.array([[2, 0.5], [0.5, 1]]), np.array([[1, -0.5], [-0.5, 2]]), np.array([[1, 0], [0, 2]])]

# Initialize the clustering objective and accuracy arrays
kmeans_objectives = []
kmeans_accuracies = []
gmm_objectives = []
gmm_accuracies = []
sigmas = [0.5, 1, 2, 4, 8]

def generate_data(sigma):
    # Generate 100 points from each Gaussian distribution
    P_a = np.random.multivariate_normal(means[0], covariances[0] * sigma, 100)
    P_b = np.random.multivariate_normal(means[1], covariances[1] * sigma, 100)
    P_c = np.random.multivariate_normal(means[2], covariances[2] * sigma, 100)
    X = np.vstack((P_a, P_b, P_c))
    y_true = np.array([0]*100 + [1]*100 + [2]*100)  # True labels for accuracy calculation
    return X, y_true

def compute_accuracy(true_labels, predicted_labels, cluster_centers):
    # Map predicted labels to true labels
    true_centers = np.vstack((means[0], means[1], means[2]))
    perm = pairwise_distances_argmin_min(true_centers, cluster_centers)[0]
    predicted_labels_mapped = np.zeros_like(predicted_labels)
    for i, p in enumerate(perm):
        predicted_labels_mapped[predicted_labels == i] = p
    accuracy = np.mean(true_labels == predicted_labels_mapped)
    return accuracy

# Perform clustering with K-means and GMM, and calculate objectives and accuracies
for sigma in sigmas:
    X, y_true = generate_data(sigma)

    # K-means clustering
    kmeans = KMeans(n_clusters=3, n_init=10, init='k-means++').fit(X)
    kmeans_objectives.append(kmeans.inertia_)
    kmeans_accuracy = compute_accuracy(y_true, kmeans.labels_, kmeans.cluster_centers_)
    kmeans_accuracies.append(kmeans_accuracy)

    # Gaussian Mixture Model clustering
    gmm = GaussianMixture(n_components=3, n_init=10).fit(X)
    gmm_labels = gmm.predict(X)
    gmm_objectives.append(-gmm.score(X) * len(X))  # Negative log likelihood
    gmm_accuracy = compute_accuracy(y_true, gmm_labels, gmm.means_)
    gmm_accuracies.append(gmm_accuracy)

# Plotting the results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot clustering objectives
ax[0].plot(sigmas, kmeans_objectives, label='K-means', marker='o')
ax[0].plot(sigmas, gmm_objectives, label='GMM', marker='x')
ax[0].set_title('Clustering Objective vs Sigma')
ax[0].set_xlabel('Sigma')
ax[0].set_ylabel('Objective')
ax[0].legend()

# Plot clustering accuracies
ax[1].plot(sigmas, kmeans_accuracies, label='K-means', marker='o')
ax[1].plot(sigmas, gmm_accuracies, label='GMM', marker='x')
ax[1].set_title('Clustering Accuracy vs Sigma')
ax[1].set_xlabel('Sigma')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.tight_layout()
plt.show()
