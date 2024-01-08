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