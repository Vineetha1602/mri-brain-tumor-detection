import os
import cv2
import numpy as np

# Step 1: Preprocess image by converting to grayscale
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

# Step 2: Load images and assign labels (0 for normal, 1 for cancer)
def load_data(dataset_dir):
    images = []
    labels = []
    for label in ['normal', 'cancer']:
        folder_path = os.path.join(dataset_dir, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            processed_img = preprocess_image(img_path)
            images.append(processed_img)
            labels.append(1 if label == 'cancer' else 0)
    return images, np.array(labels)