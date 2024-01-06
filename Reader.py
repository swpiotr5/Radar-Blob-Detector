import cv2
import os
import numpy as np

class Entity:
    def __init__(self, initial_blob):
        self.track = [cv2.minEnclosingCircle(initial_blob)[0]]  # Store the center of the blob

    def add_blob(self, blob):
        self.track.append(cv2.minEnclosingCircle(blob)[0])  # Store the center of the blob

    def get_track(self):
        return self.track

    def distance_to(self, blob):
        blob_center = cv2.minEnclosingCircle(blob)[0]
        last_known_position = self.track[-1]
        return ((blob_center[0] - last_known_position[0]) ** 2 + (blob_center[1] - last_known_position[1]) ** 2) ** 0.5


class EntityTracker:
    def __init__(self, blobs):
        self.entities = [Entity(blob) for blob in blobs[0]]
        self.blobs = blobs[1:]
        self.SOME_THRESHOLD = 300  # Define the missing constant

    def track_entities(self):
        for blobs in self.blobs:
            for blob in blobs:
                closest_entity = min(self.entities, key=lambda entity: entity.distance_to(blob))

                # Check if the blob is close enough to the closest entity to be considered part of it
                if closest_entity.distance_to(blob) < self.SOME_THRESHOLD:  # Use the defined constant
                    closest_entity.add_blob(blob)
                else:
                    # If the blob is not close enough to any existing entity, create a new entity for it
                    self.entities.append(Entity(blob))

class Animation:
    def __init__(self, tracker=None, input_dir='pictures', delay=1000):
        self.tracker = tracker
        self.input_dir = input_dir
        self.delay = delay
        self.image_files = sorted(os.listdir(self.input_dir))
        self.detector = BlobDetector(input_dir)

    def run(self):
        cv2.namedWindow('Animation', cv2.WINDOW_NORMAL)
        self.detector.detect_blobs()

        for i, image_file in enumerate(self.image_files):
            if cv2.getWindowProperty('Animation', 0) < 0:
                break

            image_path = os.path.join(self.input_dir, image_file)
            image = cv2.imread(image_path)

            frame = np.zeros_like(image)
            frame = cv2.add(frame, image)

            # Draw circles around the detected blobs
            for contour in self.detector.get_blobs()[i]:
                # Calculate the center and radius of the minimum enclosing circle
                center, radius = cv2.minEnclosingCircle(contour)
                center = tuple(map(int, center))  # Convert the center coordinates to integers
                radius = int(radius)  # Convert the radius to an integer

                # Draw the circle on the frame
                cv2.circle(frame, center, radius, (255, 0, 0), 2)
            
            # Track entities for the current frame
            if self.tracker is not None:
                self.tracker = EntityTracker(self.detector.get_blobs()[:i+1])
                self.tracker.track_entities()

                for entity in self.tracker.entities:
                    track = entity.get_track()
                    for j in range(1, len(track)):
                        cv2.line(frame, tuple(map(int, track[j-1])), tuple(map(int, track[j])), (0, 255, 0), 2)

            cv2.imshow('Animation', frame)
            cv2.waitKey(self.delay)

        cv2.destroyAllWindows()

class BlobDetector:
    def __init__(self, input_dir='pictures'):
        self.input_dir = input_dir
        self.image_files = sorted(os.listdir(self.input_dir))
        self.blobs = []

    def detect_blobs(self):
        for i, image_file in enumerate(self.image_files):
            image_path = os.path.join(self.input_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Threshold the image to get only white blobs
            _, thresholded = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Save the contours (blobs) for this image
            self.blobs.append(contours)

    def get_blobs(self):
        return self.blobs

class JPDA:
    def __init__(self, blobs):
        self.blobs = blobs
        self.probabilities = []
        self.tracks = []

    def calculate_probabilities(self):
        for i in range(1, len(self.blobs)):
            current_blobs = self.blobs[i]
            previous_blobs = self.blobs[i-1]
            probabilities_for_current_blobs = []

            for blob in current_blobs:
                # Calculate the distance from this blob to all blobs in the previous frame
                distances = [np.linalg.norm(np.mean(blob, axis=0) - np.mean(previous_blob, axis=0)) for previous_blob in previous_blobs]

                # Convert distances to probabilities
                probabilities = [1 / distance for distance in distances]

                # Normalize probabilities so they sum to 1
                total = sum(probabilities)
                probabilities = [probability / total for probability in probabilities]

                probabilities_for_current_blobs.append(probabilities)

            self.probabilities.append(probabilities_for_current_blobs)

    def get_tracks(self):
        # For simplicity, we'll just assign each blob to the track with the highest probability
        for i in range(len(self.probabilities)):
            tracks_for_current_blobs = [np.argmax(probabilities) for probabilities in self.probabilities[i]]
            self.tracks.append(tracks_for_current_blobs)

        return self.tracks

# Create an Animation instance
animation = Animation()

# Detect blobs before getting them
animation.detector.detect_blobs()

# Get the detected blobs
blobs = animation.detector.get_blobs()

if blobs:  # Check if blobs is not empty
    tracker = EntityTracker(blobs)
    tracker.track_entities()

    # Run the animation with tracker
    animation = Animation(tracker)
    animation.run()

    # Create a JPDA instance and calculate probabilities
    jpda = JPDA(blobs)
    jpda.calculate_probabilities()

    # Get the tracks
    tracks = jpda.get_tracks()
else:
    print("No blobs detected.")