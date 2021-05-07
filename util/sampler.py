from torch.utils.data import Sampler
import random
import math


class ShuffledFilterSampler(Sampler[int]):
    def __init__(self, indexes=None, indexes_file=None):
        self.indexes = None
        if indexes and indexes_file:
            raise RuntimeError("Only one argument has to be defined!")

        if indexes:
            self.indexes = indexes
        if indexes_file:
            self.indexes = []
            with open(indexes_file, 'r') as file:
                for line in file:
                    self.indexes.append(int(line.strip()))
        if self.indexes is None:
            raise RuntimeError("One of two arguments has to be defined!")
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.indexes)

    def __iter__(self):
        self.shuffle()
        return (i for i in self.indexes)


class SubsetFilterSampler(Sampler[int]):
    def __init__(self, samples, indexes=None, indexes_file=None):
        self.indexes = None
        if indexes and indexes_file:
            raise RuntimeError("Only one argument has to be defined!")

        if indexes:
            self.indexes = indexes
        if indexes_file:
            self.indexes = []
            with open(indexes_file, 'r') as file:
                for line in file:
                    self.indexes.append(int(line.strip()))
        if self.indexes is None:
            raise RuntimeError("One of two arguments has to be defined!")

        assert len(self.indexes) > samples

        self.total_samples = len(self.indexes)
        self.samples = samples
        self.increment = int(math.floor(self.total_samples / self.samples))
        self.max_i = self.samples * self.increment
        self.indexes = self.indexes[0:self.max_i:self.increment]

    def __iter__(self):
        return (i for i in self.indexes)

