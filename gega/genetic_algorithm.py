import pandas as pd
import numpy as np
import gega.utilities as utilities
import gega.ga as ga
from math import log10


class GeneticAlgorithm(object):
    def __init__(self, result_dir, solution_description, termination_fitness_threshold, num_gens_before_termination,
                 population_size=8,
                 generations=400,
                 crossover_probability=0.8,
                 mutation_probability=0.2,  # 0.05
                 elitism=True,
                 minimise_fitness=True,
                 skip_known_solutions=False,
                 load_past_data=False):
        # __metaclass__ = abc.ABCMeta

        # Make sure that we have all the properties of a solution that we need
        # Note: this does not guaranteed they are the correct/assumed shapes
        assert population_size > 2, "Population size must be greater than 2"
        assert hasattr(solution_description, "num_genes"), "num_genes missing from solution_description"
        assert hasattr(solution_description, "gene_bounds"), "gene_bounds missing from solution_description"
        assert hasattr(solution_description, "gene_init_range"), "gene_init_range missing from solution_description"
        assert hasattr(solution_description, "gene_sigma"), "gene_sigma missing from solution_description"
        assert hasattr(solution_description, "gene_mutation_prob"), \
            "gene_mutation_prob missing from solution_description"

        self._fitness_threshold = termination_fitness_threshold
        self._num_gens_before_termination = num_gens_before_termination
        self.solution_description = solution_description
        self._population_size = population_size
        self._max_generations = generations
        self._crossover_probability = crossover_probability
        self._mutation_probability = mutation_probability
        self._elitism = elitism
        self._minimise_fitness = minimise_fitness
        self._skip_known_solutions = skip_known_solutions
        self._load_past_data = load_past_data
        self._past_data = None

        self.results_dir = result_dir

        self.solution_lookup = {}
        self.solution_idx = 0

        self._POPULATION_HISTORY_FILENAME = "/population_history.csv"
        self._EVOLUTION_HISTORY_FILENAME = "/evolution_history.csv"
        self._GENERATION_HISTORY_FILENAME = "/generation_history.csv"
        self._LIVE_POPULATION_FILENAME = "/population.csv"

        if self._load_past_data:
            self._past_data = np.loadtxt("{0}{1}".format(self.results_dir, self._EVOLUTION_HISTORY_FILENAME),
                                         dtype=float,
                                         comments='#',
                                         delimiter="\t")
        file_operation = "w" if not self._load_past_data else "a"
        self.f_evolution_history = open("{0}{1}".format(self.results_dir, self._EVOLUTION_HISTORY_FILENAME), file_operation, 1)
        self.f_generation_max = open("{0}{1}".format(self.results_dir, self._GENERATION_HISTORY_FILENAME), file_operation, 1)
        self.f_population_history = open("{0}{1}".format(self.results_dir, self._POPULATION_HISTORY_FILENAME), file_operation,
                                         1)

        # TODO - save to file the GA setup (max generations, solution description, etc.) so that can compare when loading data

    # def seed_population(self):
    #     population = None
    #     for gene_idx, (lower_bound, upper_bound) in zip(range(self.solution_description.num_genes),
    #                                                     self.solution_description.gene_init_range):
    #         gene_weights = np.random.uniform(low=lower_bound, high=upper_bound, size=(self._population_size, 1))
    #         population = np.concatenate((population, gene_weights), axis=1) if population is not None else gene_weights
    #
    #     return population

    def seed_population(self):
        population = None
        for gene_idx, (lower_bound, upper_bound) in zip(range(self.solution_description.num_genes),
                                                        self.solution_description.gene_init_range):
            # gene_weights = 10 ** (-1 * np.random.uniform(low=-log10(lower_bound), high=-log10(upper_bound), size=(self._population_size, 1)))
            gene_weights = np.random.uniform(low=lower_bound, high=upper_bound, size=(self._population_size, 1))
            population = np.concatenate((population, gene_weights), axis=1) if population is not None else gene_weights
        return population

    # @abc.abstractmethod
    def calculate_fitness(self, solution):
        """
        [5, 7] is the optimal solution for this fitness function, equating to a fitness of 0
        :param solution:
        :return:
        """
        return abs((solution[0] - 1e-4)) + abs(solution[1] - 1e-5)
        # return abs(((2*solution[0]**2) + solution[1]) - 57)

    def update_population_logs(self, generation_id, population, fitness):
        # Overwrite the "live" population file
        df = pd.DataFrame(population)
        idx = population.shape[1]
        df[idx] = fitness
        df.to_csv("{0}{1}".format(self.results_dir, self._LIVE_POPULATION_FILENAME), sep="\t", header=False)

        # Append the population to the population history log
        self.f_population_history.write("====== {0} ======\n".format(generation_id))
        self.f_population_history.write(df.to_string(header=False))
        self.f_population_history.write("\n")

    def run(self):
        generation_idx = 0
        generations_since_improvement = 0

        if not self._load_past_data:
            population = self.seed_population()
            print("Initial population:")
            print(population)

            fitness = np.zeros(shape=(self._population_size, 1))
            for solution_idx, solution in zip(range(self._population_size), population):
                solution_fitness = self.calculate_fitness(solution)
                self.log_solution(1, solution, solution_fitness)

                fitness[solution_idx, 0] = solution_fitness
                self.solution_idx += 1

            self.update_population_logs(generation_idx, population, fitness)
        else:
            # populate the in-memory store of past solutions that have been run
            for solution_idx in range(self._past_data.shape[0]):
                indv = self._past_data[solution_idx, :]
                is_repeat = not bool(indv[1])
                if not is_repeat:
                    self.solution_idx += 1
                    # ignore the solution index and fitness for the solution
                    self.solution_lookup[tuple(indv[2:-1])] = indv[-1]

            # update the generation info
            last_fitness = None
            _past_generations = np.loadtxt("{0}{1}".format(self.results_dir, self._GENERATION_HISTORY_FILENAME),
                                           comments='#',
                                           dtype=float,
                                           delimiter="\t")
            for gen_idx in range(_past_generations.shape[0]):
                gen = _past_generations[gen_idx, :]
                # if gen_idx > self._population_size:
                generation_idx += 1
                generations_since_improvement += 1

                curr_fitness = gen[-1]
                if last_fitness is not None and last_fitness != curr_fitness:
                    generations_since_improvement = 0
                last_fitness = curr_fitness

            # TODO - remove this debug
            # for value in self.solution_lookup.items():
            #     print(value)

            # Update the population and fitness arrays
            population_from_file = np.loadtxt("{0}{1}".format(self.results_dir, self._LIVE_POPULATION_FILENAME),
                                              comments='#',
                                              dtype=float,
                                              delimiter="\t")
            population = population_from_file[:, 1:-1]
            fitness = population_from_file[:, -1].reshape((self._population_size, 1))

            print("Loading past data, starting from:")
            print(population)
            print(fitness)
            print("solution idx: {0}".format(self.solution_idx))
            print("generation idx: {0}".format(generation_idx))
            print("gens since improvement: {0}".format(generations_since_improvement))
            print("============ STARTING NEW GENERATIONS ===============")
        # exit(0)

        while generation_idx < self._max_generations:
            print("Population:\n{0}".format(population))
            print("Fitness:\n{0}".format(fitness))
            parents, parents_fitness = ga.select_mating_pool_tournament(population, fitness, 4,
                                                                        minimise=self._minimise_fitness)
            # TODO - check if parents are same
            if utilities.is_close(parents[0, :], parents[1, :], self.solution_description.atol):
                pass
            print("Parents:\n{0}".format(parents))
            child = ga.crossover_random_chromosones(parents)
            print("Child after crossover:\n{0}".format(child))
            # child = ga.mutation_gaussian(child, self.solution_description.gene_sigma,
            #                              self.solution_description.gene_mutation_prob,
            #                              self.solution_description.gene_bounds)
            child = ga.mutation(child, self.solution_description)
            print("Child after mutation:\n{0}".format(child))

            call_fitness_function = True

            # Check if the individual has been run before
            if self._skip_known_solutions:
                closest = self.check_for_past_result(child)
                if closest is not None:
                    # print("Similar individual run in past")
                    # print("new  individual: {0}".format(individual))
                    # print("past individual: {0}".format(closest))
                    child_fitness = self.solution_lookup[closest]
                    call_fitness_function = False

            if call_fitness_function:
                child_fitness = self.calculate_fitness(child)
                if ga.update_population_using_elitism(population, fitness,
                                                      parents, parents_fitness,
                                                      child, child_fitness,
                                                      self.solution_description.atol,
                                                      minimise=self._minimise_fitness):
                    self.update_population_logs(generation_idx, population, fitness)
                    generations_since_improvement = 0  # TODO - move this reset!!!!!
                self.solution_idx += 1

            self.log_solution(call_fitness_function, child, child_fitness)
            self.log_best_in_generation(generation_idx, population, fitness)
            generation_idx += 1
            print("")
            print("result directory: {0}".format(self.results_dir))
            print("Generation idx: {0}".format(generation_idx))
            print("Num generations since improvement: {0}".format(generations_since_improvement))
            generations_since_improvement += 1


            # best_fitness = utilities.get_n_best(fitness, 1, minimise=self._minimise_fitness)[0]
            # Check for early end conditions
            print("Checking termination condition, num generations since improvement: {0}".format(generations_since_improvement))
            best_fitness = fitness[utilities.get_n_best(fitness, 1, minimise=self._minimise_fitness)][0][0]
            print("best fitness in generation: {0}".format(best_fitness))
            fitness_threshold_condition = best_fitness < self._fitness_threshold if self._minimise_fitness else best_fitness > self._fitness_threshold
            print("fitness_threshold_condition: {0}".format(fitness_threshold_condition))
            print("")

            if fitness_threshold_condition and generations_since_improvement > self._num_gens_before_termination:
                print("EARLY TERMINATION CONDITIONS MET")
                break
        print("MAX GENERATIONS REACHED!")

    def log_best_in_generation(self, generation_idx, population, fitness):
        best_idx = utilities.get_n_best(fitness, 1, minimise=self._minimise_fitness)
        best_fitness, best_solution = fitness[best_idx][0][0], population[best_idx, :][0]
        log_entry_individual = '\t'.join(map(str, best_solution))
        log_entry = str(generation_idx) + "\t" + log_entry_individual + "\t" + str(best_fitness) + "\n"
        self.f_generation_max.write(log_entry)
        self.f_generation_max.flush()

    def log_solution(self, executed, solution, fitness):
        log_entry_individual = '\t'.join(map(str, solution))
        log_entry = str(self.solution_idx) + "\t" + str(int(executed)) + "\t" + log_entry_individual + "\t" + str(
            fitness) + "\n"
        self.f_evolution_history.write(log_entry)
        self.solution_lookup[tuple(solution)] = fitness
        self.f_evolution_history.flush()

    def check_for_past_result(self, individual):
        closest = None
        for key in self.solution_lookup.keys():
            match = utilities.is_close(np.array(key), np.array(individual), self.solution_description.atol)
            if match:
                print("Individual {0} matched with {1}".format(individual, key))
                closest = key
        return closest
