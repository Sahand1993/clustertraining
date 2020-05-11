import numpy as np
from itertools import chain
from scipy.sparse import csr_matrix

from typing import List, Iterable, Set
import random
from dssm.config import *
from dssm.model_dense import NO_OF_TRIGRAMS

DEFAULT_IRRELEVANT_SAMPLES = 4
DEFAULT_BATCH_SIZE = 5

CSV_SEPARATOR = ";"

def toints(strings) -> List[int]:
    return list(map(lambda s: int(s), strings))


def readCsvLines(file) -> List[List[str]]:
    return list(map(lambda line: line.split(CSV_SEPARATOR),
        file.readlines()))


def sample_numbers(end: int, size: int, exclude: int) -> List[int]:
    """
    Samples uniformly from 1 to end (inclusive) and excludes exclude.
    :param end:
    :param size:
    :param exclude:
    :return:
    """
    numbers: List[int] = np.random.choice(end, replace=False, size=size + 1) + 1
    numbers: List[int] = filter(lambda number: number != exclude, numbers)
    numbers: List[int] = list(numbers)[:size]
    return numbers


def filterWords(wordIndices: Iterable) -> np.ndarray:
    return np.array(list(filter(lambda wordIndex: wordIndex in FILTERED_WORD_INDICES, wordIndices)))


class DataPoint():

    def __init__(self, _id: int, query_ngrams: np.ndarray, relevant_ngrams: np.ndarray, irrelevant_ngrams: np.ndarray):
        """
        :param query_ngrams: vector of integers
        :param relevant_ngrams: as above
        :param irrelevant_ngrams: matrix of integers, shape is [no_of_irrelevants, None]
        """
        self._id = _id
        self._query_ngrams: np.ndarray = query_ngrams
        self._relevant_ngrams: np.ndarray = relevant_ngrams
        self._irrelevant_ngrams: np.ndarray = irrelevant_ngrams


    def get_query_ngrams(self) -> np.ndarray:
        return self._query_ngrams


    def get_relevant_ngrams(self) -> np.ndarray:
        return self._relevant_ngrams

    def get_irrelevant_ngrams(self) -> np.ndarray:
        return self._irrelevant_ngrams


class DataPointFactory():

    @staticmethod
    def fromNGramsData(_id: int, query_ngrams: str, relevant_ngrams: str, irrelevant_ngrams: List[str]) -> DataPoint:
        return DataPoint(
            _id,
            np.array(toints(query_ngrams.split(","))),
            np.array(toints(relevant_ngrams.split(","))),
            np.array([toints(ngrams.split(",")) for ngrams in irrelevant_ngrams])
        )

    @staticmethod
    def fromWordIndicesData(_id: int, queryWordIndices: str, relevantWordIndices: str, irrelevantWordIndices: List[str]) -> DataPoint:
        return DataPoint(
            _id,
            filterWords(np.array(toints(queryWordIndices.split(",")))),
            filterWords(np.array(toints(relevantWordIndices.split(",")))),
            np.array([filterWords(toints(ngrams.split(","))) for ngrams in irrelevantWordIndices])
        )


class DataPointBatch():

    def __init__(self, data_points: List[DataPoint], no_of_irrelevant_samples = 4):
        self.data_points = data_points
        self._no_of_irrelevant_samples = no_of_irrelevant_samples


    def get_q_indices(self) -> np.ndarray:
        return self.create_batch(list(map(lambda data_point: data_point.get_query_ngrams(), self.data_points))) # TODO: Really necessary to turn into list? (For all create_batch* function calls.


    def get_q_dense(self) -> np.ndarray:
        return self.create_batch_dense(list(map(lambda data_point: data_point.get_query_ngrams(), self.data_points)))


    def get_relevant_indices(self) -> np.ndarray:
        return self.create_batch(list(map(lambda data_point: data_point.get_relevant_ngrams(), self.data_points)))


    def get_relevant_dense(self) -> np.ndarray:
        return self.create_batch_dense(list(map(lambda data_point: data_point.get_relevant_ngrams(), self.data_points)))


    def get_irrelevant_indices(self) -> List[np.ndarray]:
        irrelevants_batches = [] # [irr1_batch, irr2_batch, irr3_batch]
        for i in range(self._no_of_irrelevant_samples):
            irrelevants_batches.append(self.create_batch(list(map(lambda data_point: data_point.get_irrelevant_ngrams()[i], self.data_points))))

        return irrelevants_batches


    def get_irrelevant_dense(self) -> List[np.ndarray]:
        irrelevants_batches = []
        for i in range(self._no_of_irrelevant_samples):
            irrelevants_batches.append(self.create_batch_dense(list(map(lambda data_point: data_point.get_irrelevant_ngrams()[i], self.data_points))))

        return irrelevants_batches


    def create_batch(self, dataPointIndices: List[np.ndarray]) -> np.ndarray:
        indices = np.empty((0, 2), np.int64)
        for i, indices in enumerate(dataPointIndices):
            new_indices = np.array([[i, index] for index in indices])
            indices = np.concatenate((indices, new_indices))
        return indices


    def create_batch_dense(self, batchIndices: List[np.ndarray]) -> np.ndarray:
        data = []
        row_ind = []
        col_ind = []
        for row, indices in enumerate(batchIndices):
            for index in indices:
                row_ind.append(row)
                col_ind.append(index)
                data.append(1)
        return csr_matrix(
            (data,
            (row_ind, col_ind)), shape=(len(batchIndices), NO_OF_TRIGRAMS)).toarray()


class RandomBatchIterator():
    """
    Randomly calls the __next__() methods of the given iterators.
    """
    def __init__(self, *args):
        """

        :param args: iterators to uniformly sample from.
        """
        self.iterators = list(args) # TODO: Create traversal order.
        self._turns = []
        for iterator in self.iterators:
            self._turns += list((iterator for _ in range(len(iterator))))

        random.shuffle(self._turns)


    def __iter__(self):
        return self


    def __next__(self):
        if self.iterators:
            try:
                iterator = self._turns.pop()
                return iterator.__next__()
            except IndexError:
                raise StopIteration
        else:
            raise StopIteration


    def restart(self):
        for iterator in self.iterators:
            iterator.restart()

        self._turns = []
        for iterator in self.iterators:
            self._turns += list((iterator for _ in range(len(iterator))))

        random.shuffle(self._turns)


    def getNoOfDataPoints(self):
        return sum(map(lambda it: it.getNoOfDataPoints(), self.iterators))


    def __len__(self):
        return sum(map(lambda iterator: len(iterator), self.iterators))
