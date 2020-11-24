import numpy as np
from copy import deepcopy
from gega.utilities import check_for_past_result


class ApproximateFitness(object):
    def __init__(self):
        self.fitness = 0
        self._history = []

    def update(self, fitness):
        self._history.append(fitness)
        return np.mean(self._history)


class FitnessByApproximation(object):
    def __init__(self, op=lambda x, y: x + y):
        # map for individual and running fitness total
        self.individual_fitness_map = {}
        # lambda function to compute the approximated fitness
        self.operation = op
        self.num_repeated = 0

    def add(self, individual, fitness, solution_description):
        key = deepcopy(individual)
        if not isinstance(key, tuple):
            key = tuple(key)
        closest = check_for_past_result(self.individual_fitness_map, key, solution_description.atol)
        if closest is not None:
            print("INDIVIDUAL HAS FITNESS SCORES ALREADY!!!!")
            current_value = self.individual_fitness_map[closest]
            self.num_repeated += 1
        else:
            current_value = ApproximateFitness()
            self.individual_fitness_map[key] = current_value

        return current_value.update(fitness)
