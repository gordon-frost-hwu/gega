# Generalised Genetic Algorithm (gega)

This package provides a generalised flexible genetic algorithm implementation
that can be used to optimise a set of parameters.
For an example of applied usage, see the optimisation scripts in the
Reinforcement Learning repo [ALLAgents](https://github.com/gordon-frost-hwu/ALLAgents.git)
that extend the [autonomous-learning-library](https://github.com/cpnota/autonomous-learning-library).

## Terminology and Genetic Algorithm (GA) Pseudocode:
1) A **"solution"** or **"individual"** is a vector containing all of the parameters
to be optimised. Each parameter is called a **"gene"**, where there are **N** genes
2) The **"Population"** is created by randomly creating **M** solutions, giving 
the population a size of M rows x N columns
3) The overridden fitness_function gets called for each
 solution in the population giving a fitness vector of size Mx1
4) Two **"Parents"** get selected using a **"Selection Strategy"** from the population, creating an 2xN array
5) An **"Offspring"** is created by a **"Crossover"** function of the two parents
6) The Offspring has a stochastic **"Mutation"** applied to one or all of it's genes
7) The overridden fitness_function gets called for the new offspring to get the 
offspring's fitness value
8) An **"Update Strategy"** occurs, whereby the offspring is inserted into the population
under certain conditions. e.g. If elitest, the offspring will only replace a solution
in the population if it has a better fitness value than one of them
9) Repeat back to step (4) for desired number of generations

##Package components
 1) The **SolutionDescription** class - this defines the properties of the parameters/genes
 to be optimised. i.e. for each gene, what it's valid range is, and what type of mutation to
 apply to it etc.
 2) the **ga**  module contains self-contained functions for selection strategies,
  mutation, and update strategies.
 3) the **utility** module contains useful methods for interacting with numpy
  arrays needed in typical ga actions
 4) the main **GeneticAlgorithm** class takes as input a SolutionDescription instance,
  and internally uses the ga and utility modules to execute the algorithm outlined
  above
  
  ## Usage
Clone the repository using:
```
git clone https://github.com/gordon-frost-hwu/gega.git
```

Install the local cloned repository as a python module:
```
cd gega
pip install -e .
```