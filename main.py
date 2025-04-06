# Step 0: Main script to coordinate all steps
from image_loader import load_data
from feature_extractor import extract_sift_features
from kmeans_encoder import build_visual_vocabulary
from svm_classifier import train_and_evaluate
import numpy as np

np.random.seed(42)

if __name__ == '__main__':
    # Step 1-2: Load and label images
    dataset_dir = 'brain_tumor_dataset'
    images, labels = load_data(dataset_dir)

    # Step 3-4: Extract SIFT features from images
    descriptors_list = extract_sift_features(images)

    # Step 5: Build a KMeans model to form visual words (clusters)
    for cluster in range(100,200,10):
        num_clusters = cluster
        kmeans = build_visual_vocabulary(descriptors_list, num_clusters=num_clusters)

        # Step 6-9: Train SVM on histogram representations and evaluate
        train_and_evaluate(descriptors_list, labels, kmeans, num_clusters)