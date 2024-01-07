import cv2
import os
import numpy as np
import random
from scipy.stats import norm

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

class BlobEntityFactory:
    @staticmethod
    def create_blob_entity(initial_blob, color):
        return BlobEntity(initial_blob, color)

class BlobEntityIterator:
    def __init__(self, entities):
        self.entities = entities
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.entities):
            entity = self.entities[self.index]
            self.index += 1
            return entity
        else:
            raise StopIteration

class BlobEntityTracker:
    def __init__(self, blobs, colors):
        self.entities = [BlobEntityFactory.create_blob_entity(blob, color) for blob, color in zip(blobs[0], colors)]
        self.blobs = blobs[1:]
        self.colors = colors
        self.SOME_THRESHOLD = 500

    def track_entities(self):
        for blobs in self.blobs:
            unassigned_entities = self.entities.copy()

            blobs = list(blobs)
            blobs.sort(key=lambda blob: min(entity.distance_to(blob) for entity in unassigned_entities))

            for blob in blobs:
                if not unassigned_entities:
                    self.entities.append(BlobEntityFactory.create_blob_entity(blob, random.choice(self.colors)))
                    continue

                closest_entity = min(unassigned_entities, key=lambda entity: entity.distance_to(blob))

                if closest_entity.distance_to(blob) < self.SOME_THRESHOLD:
                    closest_entity.add_blob(blob)
                    unassigned_entities.remove(closest_entity)
                else:
                    self.entities.append(BlobEntityFactory.create_blob_entity(blob, random.choice(self.colors)))

            for entity in unassigned_entities:
                self.entities.remove(entity)

    def get_entities_iterator(self):
        return BlobEntityIterator(self.entities)

class BlobAnimation:
    def __init__(self, tracker=None, input_dir='pictures', delay=1000):
        self.tracker = tracker
        self.input_dir = input_dir
        self.delay = delay
        self.image_files = sorted(os.listdir(self.input_dir))
        self.detector = BlobDetector(input_dir)
        self.entity_colors = []

    def run(self):
        cv2.namedWindow('Animation', cv2.WINDOW_NORMAL)
        self.detector.detect_blobs()

        num_entities = sum(len(blobs) for blobs in self.detector.get_blobs())
        self.entity_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_entities)]

        for i, image_file in enumerate(self.image_files):
            if cv2.getWindowProperty('Animation', 0) < 0:
                break

            image_path = os.path.join(self.input_dir, image_file)
            image = cv2.imread(image_path)

            frame = np.zeros_like(image)
            frame = cv2.add(frame, image)

            for contour in self.detector.get_blobs()[i]:
                center, radius = cv2.minEnclosingCircle(contour)
                center = tuple(map(int, center))
                radius = int(radius)

                if self.tracker is not None:
                    self.tracker = BlobEntityTracker(self.detector.get_blobs()[:i+1], self.entity_colors)
                    self.tracker.track_entities()

                    entities_iterator = self.tracker.get_entities_iterator()

                    for entity in entities_iterator:
                        track = entity.get_track()
                        entity_color = entity.color
                        for j in range(1, len(track)):
                            cv2.line(frame, tuple(map(int, track[j-1])), tuple(map(int, track[j])), entity_color, 1)
                        
                        cv2.circle(frame, tuple(map(int, track[-1])), radius, entity_color, 1)

            cv2.imshow('Animation', frame)
            cv2.waitKey(self.delay)

        cv2.destroyAllWindows()

class AssociationManager:
    @staticmethod
    def associate_blobs(previous_probabilities, current_probabilities):
        associations = []

        for j in range(len(current_probabilities)):
            max_probability = -1
            best_match = -1

            for k in range(len(previous_probabilities)):
                probability_ij = current_probabilities[j]
                prev_prob = previous_probabilities[k]

                prob = prev_prob * probability_ij
                if prob > max_probability:
                    max_probability = prob
                    best_match = k

            associations.append((j, best_match))

        return associations

class JPDA:
    def __init__(self, blobs):
        self.blobs = blobs
        self.probabilities = []
        self.tracks = []

    def calculate_probabilities(self):
        for i in range(1, len(self.blobs)):
            current_blobs = self.blobs[i]
            previous_blobs = self.blobs[i - 1]
            probabilities_for_current_blobs = []

            for blob in current_blobs:
                distances = [np.linalg.norm(np.mean(blob, axis=0) - np.mean(previous_blob, axis=0)) for previous_blob in previous_blobs]
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)

                probabilities = [norm.pdf(distance, loc=mean_distance, scale=std_distance) for distance in distances]
                threshold = mean_distance + 3 * std_distance
                probabilities = [prob if dist < threshold else 0.0 for dist, prob in zip(distances, probabilities)]

                total = sum(probabilities)
                probabilities = [probability / total for probability in probabilities]

                probabilities_for_current_blobs.append(probabilities)

            self.probabilities.append(probabilities_for_current_blobs)

    def get_tracks(self):
        association_manager = AssociationManager()

        for i in range(len(self.probabilities)):
            tracks_for_current_blobs = []
            previous_probabilities = []
            if i > 0:
                previous_probabilities = self.probabilities[i - 1]

            for j in range(len(self.probabilities[i])):
                try:
                    associations = association_manager.associate_blobs(previous_probabilities, self.probabilities[i][j])
                    tracks_for_current_blobs.extend(associations)
                except TypeError:
                    pass

            self.tracks.append(tracks_for_current_blobs)

        return self.tracks

def main():
    animation = BlobAnimation()

    animation.detector.detect_blobs()
    blobs = animation.detector.get_blobs()

    if blobs:
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(blobs))]
        tracker = BlobEntityTracker(blobs, colors)
        tracker.track_entities()

        animation = BlobAnimation(tracker)
        animation.run()

        jpda = JPDA(blobs)
        jpda.calculate_probabilities()
        tracks = jpda.get_tracks()
    else:
        print("No blobs detected.")

if __name__ == "__main__":
    main()
