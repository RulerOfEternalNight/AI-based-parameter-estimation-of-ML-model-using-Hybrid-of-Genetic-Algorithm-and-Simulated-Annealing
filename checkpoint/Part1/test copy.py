# https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6

import numpy
import ga

equation_inputs = [4, -2, 3.5, 5, -11, -4.7]
num_weights = len(equation_inputs)


sol_per_pop = 8
num_parents_mating = 4

pop_size = (sol_per_pop, num_weights)

new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
# print(new_population)


num_generations = 1000
for generation in range(num_generations):

    fitness = ga.cal_pop_fitness(equation_inputs, new_population)

    parents = ga.select_mating_pool(new_population, fitness,  num_parents_mating)

    offspring_crossover = ga.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)

    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

fitness = ga.cal_pop_fitness(equation_inputs, new_population)

best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])
