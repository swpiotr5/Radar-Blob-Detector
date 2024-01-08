import random
from BlobEntityFactory import BlobEntityFactory
from BlobEntityIterator import BlobEntityIterator
import cv2
import numpy as np

class BlobEntityTracker:
    def __init__(self, blobs, colors):
        self.entities = [BlobEntityFactory.create_blob_entity(blob, color) for blob, color in zip(blobs[0], colors)]
        self.blobs = blobs[1:]
        self.colors = colors
        self.SOME_THRESHOLD = 100

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

    def draw_tracks(self, frame):
            for entity in self.get_entities_iterator():
                track = entity.get_track()
                entity_color = entity.color
                track_points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [track_points], isClosed=False, color=entity_color, thickness=1, lineType=cv2.LINE_AA)