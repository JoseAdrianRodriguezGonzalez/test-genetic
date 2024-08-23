# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:47:53 2024

@author: endea
"""

import pandas as pd
import numpy as np
import math as mt
from matplotlib import pyplot as plt
from IPython.display import clear_output

def init_population(df, q):
    #Empty array
    perm_array=np.empty((q,28), dtype=object)
    for i in range(q):
        #Permutation of initial dataframe
        perm=np.random.permutation(df)
        perm_array[i]=perm.T[0]
    return perm_array

def get_coordinates(array, df):
    #Empty arrays
    x_vec=[]
    y_vec=[]
    for ind in array:
        #Get city
        city=df.loc[df['Ciudad']==ind]
        #Get city coordinates
        p=[city.iloc[0]['x'], city.iloc[0]['y']]
        x_vec.append(p[0])
        y_vec.append(p[1])
        
    #Stacking coordinates in rows at the end of "array" (0: Letters, 1: X values, 2: Y values)
    matrix=np.vstack((array,x_vec,y_vec))
    
    return matrix

def compute_fitness(array, df):
    #array=np.array(array)
    #Empty array
    total_dist=[]
    for ind in array:
        dist=0
        for i in range(ind.shape[0]-1):
            #print(i)
            if i<ind.shape[0]-1:
                #print(i)
                #Get cities
                city_1=df.loc[df['Ciudad']==ind[i]]
                city_2=df.loc[df['Ciudad']==ind[i+1]]
                #Get X and Y values of cities
                p1=[city_1.iloc[0]['x'], city_1.iloc[0]['y']]
                p2=[city_2.iloc[0]['x'], city_2.iloc[0]['y']]
                #Computing euclidean distance
                dist+=mt.dist(p1, p2)
        total_dist.append(dist)
            
    return total_dist

def combat(df):
    #Empty matrix
    winners=np.empty((len(df),29), dtype=object)
    for i in range(len(df)):
        #Random indexes
        random_index_1=np.random.randint(0,len(df)-1)
        random_index_2=np.random.randint(0,len(df)-1)
        #Fighters
        fighter_1=df.loc[random_index_1]
        fighter_2=df.loc[random_index_2]
        #Compare fighters fitness
        if fighter_1.loc['fitness']<fighter_2.loc['fitness']:
            winners[i]=fighter_1.T
        else:
            winners[i]=fighter_2.T
    return winners[:,:-1]

def roulette_wheel(df):
    #Empty matrix
    winners=np.empty((len(df),29), dtype=object)
    #Get fitness values
    fitness_values = df['fitness'].values
        
    #Total fitness
    total_fitness = np.sum(fitness_values)
    #Probabilities of each ind
    selection_probs = fitness_values / total_fitness
    
    for i in range(len(df)):
        #Get winner
        winner_fitness = np.random.choice(fitness_values, 1, p=selection_probs)
        
        #Get all ind that have the winner fitness
        winner_candidates = df.loc[df['fitness'] == winner_fitness[0]]
        
        #If exists more than one candidate, select the first one
        if len(winner_candidates) > 1:
            winner = winner_candidates.sample(n=1)
        else:
            winner = winner_candidates
        
        #Transform winner into 1x28 array and add it to winners matrix
        winners[i] = np.array(winner.values.flatten())
    return winners[:,:-1]

def recombination(df):
    #Empty matrix
    children=np.empty((len(df),df.shape[1]), dtype=object)
    #Recombination probability
    recomb_prob=0.9
    
    
    for i in range(int(len(df)/2)):
        #Random value between 0 and 1
        recomb_rvalue=np.random.uniform(0,1)
        #print(recomb_rvalue)
        #Get 2 random indexes
        random_index_1 = np.random.randint(0, len(df))
        random_index_2 = np.random.randint(0, len(df))
        #Select parents
        parent_1 = np.array(df.loc[random_index_1]).reshape(1, df.shape[1])
        parent_2 = np.array(df.loc[random_index_2]).reshape(1, df.shape[1])
    
        
        if(recomb_rvalue<recomb_prob): #Do recombination
            size = df.shape[1]
        
            #Cross point
            cx_point = np.random.randint(1, size)
        
            #Empty children arrays
            child_1 = np.full((1, size), '', dtype=parent_1.dtype)
            child_2 = np.full((1, size), '', dtype=parent_1.dtype)
            
            #From cx_point to the end of array equals to cx_point to the end of parent 2
            child_1[0, cx_point:] = parent_2[0, cx_point:]
            #From cx_point to the end of array equals to cx_point to the end of parent 1
            child_2[0, cx_point:] = parent_1[0, cx_point:]
        
            
            for j in range(size):
                if parent_1[0, j] not in child_1:
                    #Get first empty index
                    idx = np.where(child_1 == '')[1][0]
                    #Put letter in position idx
                    child_1[0, idx] = parent_1[0, j]
                if parent_2[0, j] not in child_2:
                    #Get first empty index
                    idx = np.where(child_2 == '')[1][0]
                    #Put letter in position idx
                    child_2[0, idx] = parent_2[0, j]
            #child_1 and child_2 into children arrray
            children[2*i]=child_1
            children[2*i+1]=child_2
        else:#Dont do recombination
            children[2*i]=parent_1
            children[2*i+1]=parent_2
            
    return children


def mutation(df_original):
    #Mutation probability
    mutation_prob=0.1
    
    df=df_original.copy()
    #Empty matrix
    mutation_array=np.empty((len(df),df.shape[1]), dtype=object)
    
    for i in range(len(df)):
        #Random mutation value between 0 and 1
        mutation_rvalue=np.random.uniform(0,1)
        
        if mutation_rvalue<mutation_prob: #Do mutation
            #Random indexes
            random_index_1=np.random.randint(0, df.shape[1])
            random_index_2=np.random.randint(0, df.shape[1])
            #Letter 1 and 2 for mutation
            mut_1=df[i][random_index_1]
            mut_2=df[i][random_index_2]
            #Swap letters
            df[i][random_index_1]=mut_2
            df[i][random_index_2]=mut_1
            #Adding mutated ind to array
            mutation_array[i]=df[i]
        else:
            mutation_array[i]=df[i]
        
    return mutation_array


def start_GA(population, df):
    #Compute fitness
    fitness=compute_fitness(population, dataset)
    #fitness to DF
    fitness=pd.DataFrame(fitness, columns=['fitness'])
    #Population to DF
    population=pd.DataFrame(population)
    #Concat fitness col to population DF
    population=pd.concat([population,fitness], axis=1)
    print(population)
    
    ####Combat
    #parents=combat(population)
    
    #####Roulette
    parents=roulette_wheel(population)
    #Parents to DF
    parents=pd.DataFrame(parents)

    #Recombination
    new_generation=recombination(parents)
    #new_generation=pd.DataFrame(new_generation)
    
    #Mutation
    new_generation_mut=mutation(new_generation)
    #New generation to DF
    new_generation=pd.DataFrame(new_generation)
    #New generation muy to DF
    new_generation_mut=pd.DataFrame(new_generation_mut)

    #Concat old population and new generation mut
    new_population=pd.concat([population.loc[:, 0:27], new_generation_mut], axis=0, ignore_index=True)
    #New population to array
    new_population=np.array(new_population)
    
    #Compute new population fitness
    new_population_fitness=compute_fitness(new_population, dataset)
    #Set name to fitness column
    new_population_fitness=pd.DataFrame(new_population_fitness, columns=['fitness'])
    #New population to DF
    new_population=pd.DataFrame(new_population)
    #Concat new population and its fitness
    new_population=pd.concat([new_population, new_population_fitness], axis=1)
    #Sort new population by fitness
    sorted_new_population=new_population.sort_values(by='fitness')
    sorted_new_population.reset_index(inplace=True, drop=True)
    
    #Replacement
    new_population=sorted_new_population.loc[:len(sorted_new_population)/2-1,:]
    #print(new_population)
    
    return np.array(new_population.iloc[:,:-1]), np.array(new_population.iloc[:,-1])
        


#main
dataset=pd.read_csv('./Ciudades.csv')

it=100
population=init_population(dataset, 100)
population_fitness=compute_fitness(population, dataset)

fitness_values=[]
x_axis=np.linspace(0,it,num=it)

for i in range(it):
    print('Iteration:', i, '\n')
    #Get coordinates
    matrix = get_coordinates(population[0], dataset)
    
    #Clear previous route
    clear_output(wait=True)
    
    #Plot config
    plt.figure(figsize=(10, 8))
    #Plotting
    plt.plot(matrix[1], matrix[2], '-o')
    
    #Superior title
    plt.suptitle(f'Generation: {i}, fitness: {population_fitness[0]}')
    fitness_values.append(population_fitness[0])
    
    
    #Letter of each city
    for j, val in enumerate(matrix[0]):
        plt.annotate(val, (matrix[1][j] + 0.5, matrix[2][j] + 0.5), fontsize=12)
    
    #Show
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.show()
    
    #Update population
    population, population_fitness = start_GA(population, dataset)

#Last best fitness plot
plt.figure(figsize=(10,8))
plt.suptitle('Fitness plot (low is better)')
plt.plot(x_axis, fitness_values)
plt.xlabel("Iterations")
plt.ylabel("Fitness")
















