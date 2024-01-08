from AssociationManager import AssociationManager
import numpy as np
from scipy.stats import norm

class JPDA:
    def __init__(self, blobs):
        self.blobs = blobs
        self.probabilities = []
        self.tracks = []

    def calculate_distances(self, blob, previous_blobs):
        return [np.linalg.norm(np.mean(blob, axis=0) - np.mean(previous_blob, axis=0)) for previous_blob in previous_blobs]

    def calculate_probabilities(self, distances, mean_distance, std_distance):
        return [norm.pdf(distance, loc=mean_distance, scale=std_distance) for distance in distances]

    def apply_three_sigma_rule(self, distances, probabilities, mean_distance, std_distance):
        threshold = mean_distance + 3 * std_distance
        return [prob if dist < threshold else 0.0 for dist, prob in zip(distances, probabilities)]

    def normalize_probabilities(self, probabilities):
        total = sum(probabilities)
        return [probability / total for probability in probabilities]

    def calculate_probabilities_main(self):
        for i in range(1, len(self.blobs)):
            current_blobs = self.blobs[i]
            previous_blobs = self.blobs[i - 1]
            probabilities_for_current_blobs = []

            for blob in current_blobs:
                distances = self.calculate_distances(blob, previous_blobs)
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)

                probabilities = self.calculate_probabilities(distances, mean_distance, std_distance)
                probabilities = self.apply_three_sigma_rule(distances, probabilities, mean_distance, std_distance)
                probabilities = self.normalize_probabilities(probabilities)

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