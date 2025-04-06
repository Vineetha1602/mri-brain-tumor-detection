import cv2

# Step 3: Initialize SIFT feature extractor
sift = cv2.SIFT_create()

# Step 4: Extract SIFT descriptors from each image
def extract_sift_features(images):
    descriptors_list = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list