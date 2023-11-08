import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment

# Function to compute the clustering objective for K-means
def kmeans_objective(X, labels, centroids):
    distances = np.linalg.norm(X - centroids[labels])
    return np.sum(distances**2)

def evl_acc(means, centroids, labels, true_labels):
    # Compute the accuracy of the clustering
    # First map each centroid to a cluster based on the true means
    mapping = {}
    for i in range(len(centroids)):
        min_dist = np.inf
        for j in range(len(means)):
            dist = np.linalg.norm(centroids[i] - means[j])
            if dist < min_dist:
                min_dist = dist
                mapping[i] = j
    # Convert all the labels based on this mapping
    for i in range(len(labels)):
        labels[i] = mapping[labels[i]]
 
    # Compute the accuracy
    acc = 0
    for i in range(len(labels)):
        if labels[i] == true_labels[i]:
            acc += 1
 
    return acc / len(labels)

# # Function to compute the clustering objective for GMM
# def gmm_objective(X, labels, means, covariances):
#     total_log_likelihood = 0
#     for k in range(len(means)):
#         cluster_points = X[labels == k]
#         if len(cluster_points) > 0:
#             log_likelihood = multivariate_normal.logpdf(cluster_points, mean=means[k], cov=covariances[k])
#             total_log_likelihood += np.sum(log_likelihood)
#     return total_log_likelihood

def gmm_negative_log_likelihood(X, labels, means, covariances):
    total_negative_log_likelihood = 0
    for k in range(len(means)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            log_likelihood = multivariate_normal.logpdf(cluster_points, mean=means[k], cov=covariances[k])
            total_negative_log_likelihood -= np.sum(log_likelihood)  # Summing negative log likelihood
    return total_negative_log_likelihood

# K-means algorithm
def kmeans(X, K, max_iters=100):
    # Initialization step: pick random samples as initial centroids
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    for i in range(max_iters):
        # Assignment step: assign each point to the closest centroid
        # labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        # new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        # Check for convergence
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids
    return labels, centroids

# GMM algorithm
def gmm(X, K, max_iters=100, tol=1e-3):
    n, d = X.shape
    means = X[np.random.choice(n, K, replace=False)]
    covariances = [np.eye(d) * np.var(X, axis=0) for _ in range(K)]
    pis = np.ones(K) / K
    r = np.zeros((n, K))
    prev_log_likelihood = -np.inf

    for iteration in range(max_iters):
        # E-step: compute responsibilities
        for k in range(K):
            r[:, k] = pis[k] * multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])

        log_likelihood = np.sum(np.log(np.dot(r, pis)))
        
        # Check for convergence
        if np.abs(log_likelihood - prev_log_likelihood) <= tol:
            break
        prev_log_likelihood = log_likelihood

        r /= r.sum(axis=1, keepdims=True)

        # M-step: update parameters
        N_k = r.sum(axis=0)
        for k in range(K):
            means[k] = np.dot(r[:, k], X) / N_k[k]
            X_centered = X - means[k]
            covariances[k] = np.dot(r[:, k] * X_centered.T, X_centered) / N_k[k]
            pis[k] = N_k[k] / n

    # Assign points to clusters
    labels = np.argmax(r, axis=1)
    return labels, means, covariances


# Generate the synthetic data
def generate_data(sigma):
    n = 100  # number of points per cluster
    P_a = np.random.multivariate_normal([-1, -1], [[2*sigma, 0.5*sigma], [0.5*sigma, 1*sigma]], n)
    P_b = np.random.multivariate_normal([1, -1], [[1*sigma, -0.5*sigma], [-0.5*sigma, 2*sigma]], n)
    P_c = np.random.multivariate_normal([0, 1], [[1*sigma, 0], [0, 2*sigma]], n)
    X = np.concatenate((P_a, P_b, P_c), axis=0)
    true_labels = np.array([0]*n + [1]*n + [2]*n)
    true_means = [[-1,-1], [1,-1], [0,1]]
    return X, true_labels, true_means

# Values of sigma for which to generate data and perform clustering
sigmas = [0.5, 1, 2, 4, 8]

# Arrays to hold the objective and accuracy values for each sigma
kmeans_objectives = []
kmeans_accuracies = []
gmm_objectives = []
gmm_accuracies = []

# Main loop over sigma values
for sigma in sigmas:
    # Generate data
    X, true_labels, true_means = generate_data(sigma)
    
    # Perform K-means clustering
    kmeans_labels, kmeans_centroids = kmeans(X, K=3)
    kmeans_objectives.append(kmeans_objective(X, kmeans_labels, kmeans_centroids))
    kmeans_accuracies.append(evl_acc(true_means, kmeans_centroids, kmeans_labels, true_labels))
    
    # Perform GMM clustering
    gmm_labels, gmm_means, gmm_covariances = gmm(X, K=3)
    gmm_objectives.append(gmm_negative_log_likelihood(X, gmm_labels, gmm_means, gmm_covariances))
    gmm_accuracies.append(evl_acc(true_means, gmm_means, gmm_labels, true_labels))

    # gmm_accuracies.append(clustering_accuracy(true_labels, gmm_labels, K=3))

# Plotting the results
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot clustering objectives
axs[0].plot(sigmas, kmeans_objectives, marker='o', label='K-means Objective')
axs[0].plot(sigmas, gmm_objectives, marker='x', label='GMM Objective')
axs[0].set_xlabel('Sigma')
axs[0].set_ylabel('Clustering Objective')
axs[0].set_title('Clustering Objective vs Sigma')
axs[0].legend()

# Plot clustering accuracies
axs[1].plot(sigmas, kmeans_accuracies, marker='o', label='K-means Accuracy')
axs[1].plot(sigmas, gmm_accuracies, marker='x', label='GMM Accuracy')
axs[1].set_xlabel('Sigma')
axs[1].set_ylabel('Clustering Accuracy')
axs[1].set_title('Clustering Accuracy vs Sigma')
axs[1].legend()

plt.tight_layout()
plt.show()
