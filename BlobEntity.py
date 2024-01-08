import cv2

class BlobEntity:
    def __init__(self, initial_blob, color):
        self.track = [cv2.minEnclosingCircle(initial_blob)[0]]
        self.color = color

    def add_blob(self, blob):
        self.track.append(cv2.minEnclosingCircle(blob)[0])

    def get_track(self):
        return self.track

    def distance_to(self, blob):
        blob_center = cv2.minEnclosingCircle(blob)[0]
        last_known_position = self.track[-1]
        return ((blob_center[0] - last_known_position[0]) ** 2 + (blob_center[1] - last_known_position[1]) ** 2) ** 0.5