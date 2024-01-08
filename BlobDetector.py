import cv2
import os

class BlobDetector:
    def __init__(self, input_dir='pictures'):
        self.input_dir = input_dir
        self.image_files = sorted(os.listdir(self.input_dir))
        self.blobs = []

    def detect_blobs(self):
        for i, image_file in enumerate(self.image_files):
            image_path = os.path.join(self.input_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            _, thresholded = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            self.blobs.append(contours)

    def get_blobs(self):
        return self.blobs