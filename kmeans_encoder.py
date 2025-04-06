import numpy as np
from sklearn.cluster import KMeans

# Step 5: Use KMeans to create a visual vocabulary (cluster all descriptors)
def build_visual_vocabulary(descriptors_list, num_clusters=120):
    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans

# Step 6: For each image, compute a histogram of visual words
def compute_histogram(descriptors, kmeans, num_clusters):
    histogram = np.zeros(num_clusters)
    if descriptors is not None:
        cluster_indices = kmeans.predict(descriptors)
        for idx in cluster_indices:
            histogram[idx] += 1
    return histogram