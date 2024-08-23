#Maximize x^2-x 
import random
import numpy as np
x=[]
def fobject(x):
    return -x**2+x
#First, initialize a populñation
def initialize(pop_size,min,max):
    return [random.uniform(min,max) for _ in range (pop_size)]
#Here we make a list, that has random numbers, from a min to max value, and therefore,make the population with 
#a for with size of the population passed as a parameter or the function


#Let's evaluate it
def evaluate(population):
    return [fobject(x) for x in  population]
#Here we'll pass the values cretaed in the population to the objective function 

#We are going to select randomly the values accoridng a propbability that is based according with the fitness
def select(population,fitness,num_parents):
    probabilities = np.abs(fitness) / np.sum(np.abs(fitness))
    parents=np.random.choice(population,num_parents,p=probabilities)
    return parents
def crossover(parents,offspring_size):
    offspring=[]
    for k in range(offspring_size):
        parent1=random.choice(parents)
        parent2=random.choice(parents)
        crossover_point=random.uniform(0,1)
        child=crossover_point*parent1+(1-crossover_point)*parent2
        offspring.append(child)
    return offspring
#Let's evolve it 
def mutate(offspring,mutation_rate,xmin,xmax):
    for i in range(len(offspring)):
        if random.uniform(0,1) <mutation_rate:
            offspring[i]=random.uniform(xmin,xmax)
    return offspring
def genetic(pop_size,generations,xmin,xmax,mutation_rate):
    population=initialize(pop_size,xmin,xmax)
    for gen in range(generations):
        fitness=evaluate(population)
        parents=select(population,fitness,pop_size//2)
        offspring=crossover(parents,pop_size-len(parents))
        offspring=mutate(offspring,mutation_rate,xmin,xmax)
        population=parents.tolist()+offspring
    best=max(population,key=fobject)
    return best,fobject(best)

best_solution,best_value=genetic(100,50,-10,10,0.05)
print(f"Mejor solución: x = {best_solution}, f(x) = {best_value}")