import os
from BlobDetector import BlobDetector
from BlobEntityTracker import BlobEntityTracker
import cv2
import numpy as np

class BlobAnimation:
    def __init__(self, tracker=None, input_dir='pictures', delay=500, color=None):  
        self.tracker = tracker
        self.input_dir = input_dir
        self.delay = delay
        self.image_files = sorted(os.listdir(self.input_dir))
        self.detector = BlobDetector(input_dir)
        self.entity_color = color if color else (255, 255, 255)

    def run(self):
        cv2.namedWindow('Animation', cv2.WINDOW_NORMAL)
        self.detector.detect_blobs()

        self.entity_color = self.entity_color

        for i, image_file in enumerate(self.image_files):
            if cv2.getWindowProperty('Animation', 0) < 0:
                break

            image_path = os.path.join(self.input_dir, image_file)
            image = cv2.imread(image_path)

            frame = np.zeros_like(image)
            frame = cv2.add(frame, image)

            if self.tracker is not None:
                self.tracker = BlobEntityTracker(self.detector.get_blobs()[:i+1], self.entity_color)
                self.tracker.track_entities()
                self.tracker.draw_tracks(frame)

            cv2.imshow('Animation', frame)
            cv2.waitKey(self.delay)

        cv2.destroyAllWindows()