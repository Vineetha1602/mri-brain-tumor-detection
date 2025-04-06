# Brain Tumor Classification Pipeline

This project implements an image classification pipeline to detect brain tumors using SIFT feature extraction, KMeans clustering, and an SVM classifier. The dataset used for training and testing consists of MRI images classified into 'normal' and 'cancer' categories.

### Features

SIFT (Scale-Invariant Feature Transform): Extracts distinctive features from images.

KMeans Clustering: Groups the extracted features into visual words (clusters) to create a histogram for each image.

SVM (Support Vector Machine): Classifies the images based on the visual words histogram.

### Structure

The project is split into the following four parts:

Image Loader: Handles image preprocessing (grayscale conversion) and loading.

Feature Extractor: Extracts SIFT descriptors from the images.

KMeans Encoder: Applies KMeans clustering to create a visual vocabulary and constructs histograms for each image.

SVM Classifier: Standardizes the histograms, splits the data, and trains the SVM classifier.

### Dataset

The dataset used in this project is the [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) dataset available on Kaggle. This dataset contains brain MRI images categorized into two classes:

- Normal: MRI images without any tumor.

- Tumor: MRI images with tumor presence.

### Requirements

To run this project, you need the following dependencies:

- Python 3.x

- OpenCV (for SIFT feature extraction)

- scikit-learn (for KMeans, SVM, and data preprocessing)

- NumPy (for handling data arrays)

- OS and other basic Python libraries

Install dependencies using pip:

```bash
pip install opencv-python scikit-learn numpy
```

<!------------------>

### Directory Structure

```bash
/brain-tumor-classification
├── brain_tumor_dataset/  # Your image dataset directory
│   ├── cancer/           # Folder containing cancerous images
│   ├── normal/           # Folder containing normal images
├── image_loader.py       # Image loading and preprocessing
├── feature_extractor.py  # SIFT feature extraction
├── kmeans_encoder.py     # KMeans clustering and histogram computation
├── svm_classifier.py     # SVM classification and evaluation
├── main.py               # Main script to run the pipeline
├── accuracy.txt          # File to store accuracy scores
└── README.md             # Project documentation
```

<!------------------>

### Files

1. image_loader.py
   Contains functions to preprocess and load images from the dataset.

- Key Functions:

  - preprocess_image(image_path): Converts images to grayscale.

  - load_data(dataset_dir): Loads images from the specified dataset directory and labels them ('normal' = 0, 'cancer' = 1).

2. feature_extractor.py
   Contains functions to extract features from images using the SIFT algorithm.

- Key Functions:

  - extract_sift_features(images): Extracts SIFT descriptors from each image.

3. kmeans_encoder.py
   Contains functions to apply KMeans clustering and convert descriptors to histograms.

- Key Functions:

  - build_visual_vocabulary(descriptors_list, num_clusters): Applies KMeans clustering on the descriptors.

  - compute_histogram(descriptors, kmeans, num_clusters): Converts an image's descriptors into a histogram of visual words based on the KMeans model.

4. svm_classifier.py
   Contains functions for training and evaluating the SVM classifier.

- Key Functions:

  - train_and_evaluate(descriptors_list, labels, kmeans, num_clusters): Trains the SVM classifier using the extracted feature histograms and evaluates the performance.

  - append_accuracy_to_file(accuracy, filename="accuracy.txt"): Saves the accuracy score to a text file.

5. main.py
   Coordinates the workflow by calling the functions from all the above files. Loads data, extracts features, applies KMeans clustering, and trains/evaluates the SVM classifier.

### Usage

1. Prepare your dataset

   Organize your dataset into two subfolders: _normal_ and _cancer_. Each folder should contain the corresponding images.

   Example directory structure:

   ```bash
   brain_tumor_dataset/
   ├── normal/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── cancer/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

   <!------------------>

2. Run the pipeline

   Execute the main.py script to run the pipeline. It will load images, extract features, apply KMeans clustering, and train the SVM classifier.

   ```bash
   python main.py
   ```

    <!------------------>

3. View the results

   After running the pipeline, the accuracy score will be printed on the console and saved in the accuracy.txt file.

   Example output:

   ```bash
   Accuracy: 86.27%
   Accuracy appended to accuracy.txt
   ```

   <!------------------>

### Results

The accuracy of the classifier can vary based on the number of clusters used in the KMeans step, the quality of the data, and other factors. To improve the accuracy, you can:

- Experiment with the number of clusters in the KMeans model.

- Tune the SVM parameters (e.g., kernel, regularization).

- Use additional feature extraction techniques or other machine learning algorithms.
