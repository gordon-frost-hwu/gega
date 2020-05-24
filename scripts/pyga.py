#! /usr/bin/python
import numpy as np
import gega

if __name__ == '__main__':
    num_genes = 2
    # gene_bounds = np.array([[0, 10] for gene in range(num_genes)])
    # gene_init_range = np.array([[0, 10] for gene in range(num_genes)])
    # gene_sigma = np.array([0.5 for gene in range(num_genes)])
    # gene_mutation_probability = np.array([0.2 for gene in range(num_genes)])
    # atol = np.array([0.01 for gene in range(num_genes)])


    gene_bounds = np.array([[1e-6, 1e-1] for gene in range(num_genes)])
    gene_init_range = np.array([[1e-6, 1e-1] for gene in range(num_genes)])
    gene_sigma = np.array([0.1 for gene in range(num_genes)])
    gene_mutation_probability = np.array([0.2 for gene in range(num_genes)])
    gene_mutation_type = ["log", "log"]
    atol = np.array([1e-6 for gene in range(num_genes)])

    solution_description = gega.SolutionDescription(num_genes, gene_bounds,
                                                   gene_init_range, gene_sigma,
                                                   gene_mutation_probability,
                                                    gene_mutation_type,
                                                    atol)

    test_ga = gega.GeneticAlgorithm("/tmp/", solution_description, generations=40, skip_known_solutions=True)
    test_ga.run()
