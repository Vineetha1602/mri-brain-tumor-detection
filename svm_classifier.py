import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from kmeans_encoder import compute_histogram

# Step 9: Save the accuracy score to a file for record keeping
def append_accuracy_to_file(accuracy, num_clusters, filename="accuracy.txt"):
    with open(filename, "a") as f:
        f.write(f"Accuracy with {num_clusters} clusters: {accuracy * 100:.2f}%\n")
    print(f"Accuracy appended to {filename}")

# Step 8: Train an SVM classifier using the histograms and evaluate performance
def train_and_evaluate(descriptors_list, labels, kmeans, num_clusters):
    image_histograms = np.array([compute_histogram(desc, kmeans, num_clusters) for desc in descriptors_list])

    # Step 7: Standardize the histogram features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(image_histograms)

    # Step 8.1: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    # Step 8.2: Train the SVM classifier
    classifier = svm.SVC(kernel='linear', max_iter=50000)
    classifier.fit(X_train, y_train)

    # Step 8.3: Predict and calculate accuracy
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    append_accuracy_to_file(accuracy,num_clusters)