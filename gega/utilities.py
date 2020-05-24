import numpy
from copy import copy

def generate_indices_randomly(array_size, num_indices):
    """
    Generate a list of unique indexes that are randomly sampled
    :param array_size: max index
    :param num_indices: number of indices to return
    :return: list of indices that can itself be used as an index to an array
    """
    indices = []
    while len(indices) < num_indices:
        r = numpy.random.randint(0, array_size)
        if r not in indices:
            indices.append(r)
    # print(entry_indices)
    return indices


def get_n_best(array, number_of_indices, minimise=True):
    """
    :param array: 1D numpy array of floats
    :param number_of_indices: number of indexes to return
    :param minimise: whether to take the number_of_indices as minimum or maximum values
    :return: a list with the indexes of the number_of_indices minimum or maximum values
    """
    parents = []
    for i in range(number_of_indices):
        best = 99999999999 if minimise else -99999999999
        best_idx = -1
        for index, value in zip(range(len(array)), array):
            value = value[0]
            # print("evaluating {0}, {1}".format(index, value))
            is_better_condition = value < best if minimise else value > best
            if is_better_condition and index not in parents:
                best_idx = copy(index)
                best = value
        if best_idx > -1:
            parents.append(best_idx)
    return parents


def get_row_index(array, row, atol):
    """
    Utility method to get the row index of a numpy array by matching a 1-Dim array to it.
    Note, this is not very efficient so should not be used for large arrays
    :param array: The numpy array to search in, size: [N, M]
    :param row: The numpy array to search for, size: [M x 1]
    :param atol: A numpy array that indicates the absolute tolerance for each gene, of size: [M x 1], that a match must lie within
    :return: integer of the row index if the row is present in the array, else None
    """
    if type(atol) is list or type(atol) is numpy.ndarray:
        assert len(atol) == len(row), "Absolute tolerance must either be an int, float, or array of the same" \
                                      "length as is being compared"
    for idx, _row in zip(range(array.shape[0]), array):
        if numpy.all(abs(array[idx, :] - row) < atol):
            return idx
    return None


def is_close(array1, array2, atol):
    return numpy.all(abs(array1 - array2) < atol)
