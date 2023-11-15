import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def calculate_kmeans_objective(data_points, cluster_labels, cluster_centroids):
    point_to_centroid_distances = np.linalg.norm(data_points - cluster_centroids[cluster_labels], axis=1)
    return np.sum(point_to_centroid_distances**2)

def evaluate_clustering_accuracy(ideal_means, determined_centroids, assigned_labels, true_cluster_labels):
    centroid_to_mean_mapping = {}
    for centroid_index in range(len(determined_centroids)):
        minimum_distance = np.inf
        for mean_index in range(len(ideal_means)):
            distance_to_mean = np.linalg.norm(determined_centroids[centroid_index] - ideal_means[mean_index])
            if distance_to_mean < minimum_distance:
                minimum_distance = distance_to_mean
                centroid_to_mean_mapping[centroid_index] = mean_index
    for label_index in range(len(assigned_labels)):
        assigned_labels[label_index] = centroid_to_mean_mapping[assigned_labels[label_index]]

    accurate_count = sum(assigned_label == true_label for assigned_label, true_label in zip(assigned_labels, true_cluster_labels))
    return accurate_count / len(assigned_labels)

def compute_gmm_negative_log_likelihood(data_points, gmm_labels, gmm_means, gmm_covariances):
    neg_log_likelihood_total = 0
    for mean_index, mean in enumerate(gmm_means):
        data_in_cluster = data_points[gmm_labels == mean_index]
        if data_in_cluster.size > 0:
            cluster_log_likelihood = multivariate_normal.logpdf(data_in_cluster, mean=mean, cov=gmm_covariances[mean_index])
            neg_log_likelihood_total -= np.sum(cluster_log_likelihood)
    return neg_log_likelihood_total

def perform_kmeans_clustering(data_points, num_clusters, max_iterations=100):
    centroids = data_points[np.random.choice(data_points.shape[0], num_clusters, replace=False)]
    for iteration in range(max_iterations):
        distances_to_centroids = np.linalg.norm(data_points[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
        closest_centroid_labels = np.argmin(distances_to_centroids, axis=1)
        updated_centroids = np.array([data_points[closest_centroid_labels == k].mean(axis=0) for k in range(num_clusters)])

        if np.all(updated_centroids == centroids):
            break
        centroids = updated_centroids
    return closest_centroid_labels, centroids

def perform_gmm_clustering(data_points, num_clusters, max_iterations=100, convergence_tolerance=1e-3):
    num_points, num_dimensions = data_points.shape
    gmm_means = data_points[np.random.choice(num_points, num_clusters, replace=False)]
    gmm_covariances = [np.eye(num_dimensions) * np.var(data_points, axis=0) for _ in range(num_clusters)]
    mixing_coefficients = np.ones(num_clusters) / num_clusters
    responsibilities = np.zeros((num_points, num_clusters))
    previous_log_likelihood = -np.inf

    for iteration in range(max_iterations):
        for cluster_index in range(num_clusters):
            responsibilities[:, cluster_index] = mixing_coefficients[cluster_index] * multivariate_normal.pdf(data_points, mean=gmm_means[cluster_index], cov=gmm_covariances[cluster_index])

        log_likelihood = np.sum(np.log(np.dot(responsibilities, mixing_coefficients)))
        
        if np.abs(log_likelihood - previous_log_likelihood) <= convergence_tolerance:
            break
        previous_log_likelihood = log_likelihood

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        cluster_sizes = responsibilities.sum(axis=0)
        for cluster_index in range(num_clusters):
            gmm_means[cluster_index] = np.dot(responsibilities[:, cluster_index], data_points) / cluster_sizes[cluster_index]
            data_points_centered = data_points - gmm_means[cluster_index]
            gmm_covariances[cluster_index] = np.dot(responsibilities[:, cluster_index] * data_points_centered.T, data_points_centered) / cluster_sizes[cluster_index]
            mixing_coefficients[cluster_index] = cluster_sizes[cluster_index] / num_points

    final_labels = np.argmax(responsibilities, axis=1)
    return final_labels, gmm_means, gmm_covariances

def generate_synthetic_data(variability_factor):
    samples_per_cluster = 100
    cluster_A_mean = [-1, -1]
    cluster_B_mean = [1, -1]
    cluster_C_mean = [0, 1]
    cluster_A_covariance = [[2*variability_factor, 0.5*variability_factor], [0.5*variability_factor, 1*variability_factor]]
    cluster_B_covariance = [[1*variability_factor, -0.5*variability_factor], [-0.5*variability_factor, 2*variability_factor]]
    cluster_C_covariance = [[1*variability_factor, 0], [0, 2*variability_factor]]
    
    points_A = np.random.multivariate_normal(cluster_A_mean, cluster_A_covariance, samples_per_cluster)
    points_B = np.random.multivariate_normal(cluster_B_mean, cluster_B_covariance, samples_per_cluster)
    points_C = np.random.multivariate_normal(cluster_C_mean, cluster_C_covariance, samples_per_cluster)
    
    combined_points = np.concatenate((points_A, points_B, points_C), axis=0)
    actual_labels = np.array([0]*samples_per_cluster + [1]*samples_per_cluster + [2]*samples_per_cluster)
    true_centroid_means = [cluster_A_mean, cluster_B_mean, cluster_C_mean]
    
    return combined_points, actual_labels, true_centroid_means

variability_factors = [0.5, 1, 2, 4, 8]

kmeans_objective_values = []
kmeans_accuracy_scores = []
gmm_objective_values = []
gmm_accuracy_scores = []

for variability in variability_factors:
    synthetic_data, actual_labels, ideal_means = generate_synthetic_data(variability)
    
    kmeans_assigned_labels, kmeans_determined_centroids = perform_kmeans_clustering(synthetic_data, num_clusters=3)
    kmeans_objective_values.append(calculate_kmeans_objective(synthetic_data, kmeans_assigned_labels, kmeans_determined_centroids))
    kmeans_accuracy_scores.append(evaluate_clustering_accuracy(ideal_means, kmeans_determined_centroids, kmeans_assigned_labels, actual_labels))
    
    gmm_assigned_labels, gmm_determined_means, gmm_determined_covariances = perform_gmm_clustering(synthetic_data, num_clusters=3)
    gmm_objective_values.append(compute_gmm_negative_log_likelihood(synthetic_data, gmm_assigned_labels, gmm_determined_means, gmm_determined_covariances))
    gmm_accuracy_scores.append(evaluate_clustering_accuracy(ideal_means, gmm_determined_means, gmm_assigned_labels, actual_labels))




fig, axis = plt.subplots(1, 2, figsize=(14, 6))

axis[0].plot(variability_factors, kmeans_objective_values, marker='o', label='K-means Objective')
axis[0].plot(variability_factors, gmm_objective_values, marker='x', label='GMM Objective')
axis[0].set_xlabel('Variability Factor (Sigma)')
axis[0].set_ylabel('Clustering Objective Value')
axis[0].set_title('Clustering Objective vs Variability Factor')
axis[0].legend()

axis[1].plot(variability_factors, kmeans_accuracy_scores, marker='o', label='K-means Accuracy')
axis[1].plot(variability_factors, gmm_accuracy_scores, marker='x', label='GMM Accuracy')
axis[1].set_xlabel('Variability Factor (Sigma)')
axis[1].set_ylabel('Clustering Accuracy Score')
axis[1].set_title('Clustering Accuracy vs Variability Factor')
axis[1].legend()

plt.tight_layout()
plt.savefig("1.2.png")
plt.show()
