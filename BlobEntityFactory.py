from BlobEntity import BlobEntity

class BlobEntityFactory:
    @staticmethod
    def create_blob_entity(initial_blob, color):
        return BlobEntity(initial_blob, color)