import math
import random
import numpy as np
import matplotlib.pyplot as plt


def euclidian_distance(p, q):
    return math.dist(p, q)


class TravelSalesmanGeneticAlgorithm:
    """Combinatorial genetic algorithm without repetition, this can be used in combianatorial problems that the
        points do not need to be repeatable, ex. travel salesman problem

    Parameters
    ----------
    X: Dictionary
        Problem to solve, this represents the cities and their location in a 2-D plane
        Ex. {'A':[1,3],  'B':[3,6]}
    """

    def __init__(self, X):
        self.next_sprint_test = None
        self.population_fitness = list()
        self.initial_pop = None
        self.iter = None
        self.population = None
        self.num_mut_gen = None
        self.mutation_prob = None
        self.crossover_prob = None
        self.crossover_point = None
        self.X = X
        self.city_names = list(self.X.keys())
        self.num_cities = len(self.city_names)
        self.fitness_population = list()

    def find_best_path(self, initial_pop, crossover_point, crossover_prob, mutation_prob, num_mut_gen, iter):
        """main method to find the best city.

         Parameters
         ----------
         initial_pop : Number
             Number size of the population.

         crossover_point: Number
             crossover point in which the cross will take place, between 0 and 1.

         crossover_prob: Number,
             Probability of crossover, between 0 and 1.

         mutation_prob: Number,
             Probability of mutation, between 0 and 1.

         num_mut_gen: Number,
             Number of gens to mute, between 0 and num_cities

         iter: Number,
             Number of max iterations
         """
        self.initial_pop = self.num_cities / 3 if initial_pop is None else initial_pop
        self.crossover_point = 0.5 if crossover_point is None else crossover_point
        self.crossover_prob = 0.5 if crossover_prob is None else crossover_prob
        self.mutation_prob = 0.1 if mutation_prob is None else mutation_prob
        self.num_mut_gen = 1 if num_mut_gen is None else num_mut_gen
        self.population_fitness = [0] * self.initial_pop
        self.iter = 10 if iter is None else iter

        # Validate the parameters
        if self.initial_pop < 1:
            raise AttributeError(f'initial_pop should be more than 1 actual: {self.initial_pop}')

        if self.crossover_point <= 0 or self.crossover_point >= 1:
            raise AttributeError(f'crossover point should be between 0 and 1, actual: {self.crossover_point}')

        if self.crossover_prob <= 0 or self.crossover_prob >= 1:
            raise AttributeError(f'crossover_prob point should be between 0 and 1, actual: {self.crossover_prob}')

        if self.mutation_prob <= 0 or self.mutation_prob >= 1:
            raise AttributeError(f'mutation_prob point should be between 0 and 1, actual: {self.mutation_prob}')

        if self.num_mut_gen < 0 or self.num_mut_gen > self.num_cities:
            raise AttributeError(f'prob_cross point should be between 0 and 1, actual: {crossover_prob}')

        # get initial population.
        self.population = self._generate_random_multiclass_individual(self.initial_pop, self.city_names)

        # Calculate fitness.
        self.population_fitness = self._calculate_population_fitness(self.X, self.population)

        # population_fitness = zip(self.population, self.)
        plt.figure()
        plt.ion()
        fig, ax = plt.subplots()
        first_child = self.population[0]
        x_tmp = [self.X[city][0] for city in first_child]
        y_tmp = [self.X[city][1] for city in first_child]
        graph = ax.plot(x_tmp, y_tmp, color='g')[0]
        for city in self.city_names:
            x = self.X[city][0]
            y = self.X[city][1]
            plt.scatter(x, y, marker="o")
            plt.annotate(city, (x, y))
        plt.show()

        num_individuals = math.floor(self.initial_pop / 2)
        for i in range(self.iter):
            print(f"Iteration {i +1 } of {self.iter}\n")
            selected_individual_idx = list()

            # Select individual as self.initial_pop / 2

            while len(selected_individual_idx) < num_individuals:
                # Available individuals
                available_individuals_idx = [idx for idx in range(self.initial_pop)
                                             if idx not in selected_individual_idx]

                individuals_to_fight = random.sample(available_individuals_idx, num_individuals - 1)
                selected_individual_idx.append(self._tournament(individuals_to_fight, self.population_fitness))

            new_offspring = self._new_offspring(selected_individual_idx)

            ## Select the new population
            temporal_population = np.append(self.population, new_offspring, axis=0)
            temporal_fitness = self._calculate_population_fitness(self.X, temporal_population)
            next_pop_idx = np.argsort(temporal_fitness)[:self.initial_pop]
            self.next_sprint_test = new_offspring

            self.population = np.array([temporal_population[idx] for idx in next_pop_idx])
            self.population_fitness = self._calculate_population_fitness(self.X, self.population)

            best_path_tmp = np.argmin(self.population_fitness)
            worst_path = np.argmax(self.population_fitness)
            print(f"best path so far... \n{self.population[best_path_tmp]}\n")
            print(f"best fitness: {self.population_fitness[best_path_tmp]}")
            print(f"worst fitness: {self.population_fitness[worst_path]}")

            x_best = [self.X[city][0] for city in self.population[best_path_tmp]]
            y_best = [self.X[city][1] for city in self.population[best_path_tmp]]
            graph.set_xdata(x_best)
            graph.set_ydata(y_best)
            plt.pause(0.02)

        # final selection of best path

        # Draw best path....
        best_path = np.argmin(self.population_fitness)

        print(f"BEST PATH : {self.population[best_path]}\nFITNESS : {self.population_fitness[best_path]}")
        input()

    def _new_offspring(self, selected_individuals_idx):
        children = []
        num_parents = len(selected_individuals_idx) - 1 if len(selected_individuals_idx) % 2 else len(
            selected_individuals_idx)
        random_parents_idx = random.sample(selected_individuals_idx, num_parents)

        for parent_idx in range(0, len(random_parents_idx), 2):
            # Random number for prob of crossover
            is_crossover = random.random() < self.crossover_prob
            if is_crossover:
                new_children1, new_children2 = self._crossover(self.population[parent_idx],
                                                               self.population[parent_idx + 1])
                new_children1 = self._mutation(new_children1)
                new_children2 = self._mutation(new_children2)
            else:
                # In case there is no crossover, add the parents and mutate them.
                new_children1 = self._mutation(self.population[parent_idx])
                new_children2 = self._mutation(self.population[parent_idx + 1])
                children.append(new_children1)
                children.append(self.population[parent_idx + 1])

            children.append(new_children1)
            children.append(new_children2)
        return children

    def _mutation(self, child):
        temp_child = np.copy(child)
        child_mutated = np.copy(child)

        for i in range(self.num_mut_gen):
            is_muting = random.random() < self.mutation_prob
            if is_muting:
                gens_to_change = random.sample(range(len(child)), 2)
                child_mutated[gens_to_change[0]] = temp_child[gens_to_change[1]]
                child_mutated[gens_to_change[1]] = temp_child[gens_to_change[0]]
                temp_child = np.copy(child_mutated)
        return child_mutated

    def _tournament(self, fighters_idx, population_fitness):
        fighters_fitness = [population_fitness[idx] for idx in fighters_idx]
        max_idx = np.argmax(fighters_fitness)

        return fighters_idx[max_idx]

    def _crossover(self, p1, p2):
        """Crossover for permutations.
         """
        n_gens = len(p1)
        cross_idx = math.floor(n_gens * self.crossover_point)
        s1 = p1[:cross_idx]
        s2 = p2[:cross_idx]

        s1 = np.append(s1, [gen for gen in p2 if gen not in s1])
        s2 = np.append(s2, [gen for gen in p1 if gen not in s2])

        return s1, s2

    def _calculate_population_fitness(self, X, population):
        fitness = list()
        for individual in population:
            fitness.append(self._fitness(X, individual))
        return fitness

    def _fitness(self, X, individual):
        individual_fitness = 0
        previous_city = None
        for city in individual:
            if previous_city is None:
                previous_city = city
                continue

            p = X[previous_city]
            q = X[city]
            individual_fitness += euclidian_distance(p, q)
            previous_city = city
        return individual_fitness

    def _generate_random_multiclass_individual(self, num_pop, available_classes=None):
        if available_classes is None:
            available_classes = list()
        random_pob = list()
        num_gen = len(available_classes)
        for x in range(num_pop):
            new_individual_idx = random.sample(range(num_gen), num_gen)
            new_individual = [available_classes[i] for i in new_individual_idx]
            random_pob.append(new_individual)

        return np.array(random_pob)
