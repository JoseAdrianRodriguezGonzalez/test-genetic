import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzzy
import numpy as np
#Read the dataset
df=pd.read_csv('./classifier_cancer/data.csv')
selected_features = [
    'concave points_worst', 'area_worst', 'perimeter_worst', 'radius_worst',
    'concave points_mean', 'concavity_mean', 'perimeter_mean', 'radius_mean'
]
#Change the values to 0 and 1
def conversion(df_):
    df_=df_.astype('category')
    df_=df_.cat.codes
    return df_
df['diagnosis']=conversion(df['diagnosis'])
#I eliminate the index
df.drop(columns='id',axis=1)
#With a heatmap I can know the most adecuateted variables
def plotCorr(df_):
    plt.figure(figsize=(10,20))
    sns.heatmap(df_.corr(),annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('heat map')
    plt.show()

y = df['diagnosis']

def scaling(df_,columns):
    scaler=MinMaxScaler()
    scaled_df=scaler.fit_transform(df_)
    scaled_df=pd.DataFrame(scaled_df,columns=columns)
    return scaled_df
norm=scaling(df[selected_features],selected_features)
plotCorr(norm)
##Creating the fuzzy sets
def fuzzySets(df_):
    fuzzy_data={}
    low = fuzzy.trimf(df_, [0.00, 0.10, 0.30])
    low_m = fuzzy.trimf(df_, [0.15, 0.30, 0.50])
    medium = fuzzy.trimf(df_, [0.35, 0.50, 0.65])
    medium_h = fuzzy.trimf(df_, [0.50, 0.65, 0.85])
    high = fuzzy.trimf(df_, [0.70, 0.85, 1.00])
    fuzzy_data[0] = low
    fuzzy_data[1] = low_m
    fuzzy_data[2] = medium
    fuzzy_data[3] = medium_h
    fuzzy_data[4] = high
    # Crear un DataFrame con los conjuntos difusos
    fuzzy_df = pd.DataFrame(fuzzy_data)
    return fuzzy_df
x1=fuzzySets(norm['concave points_worst'])
x2=fuzzySets(norm['area_worst'])
x3=fuzzySets(norm['perimeter_worst'])
x4=fuzzySets(norm['radius_worst'])
x5=fuzzySets(norm['concave points_mean'])
x6=fuzzySets(norm['concavity_mean'])
x7=fuzzySets(norm['perimeter_mean'])
x8=fuzzySets(norm['radius_mean'])

set_fuzzy=[x1,x2,x3]

###I'll need to create a vector that can have all the posible states of each feature
#print(fuzzy_df)
"""
* Se generan los vectores aleatorios  de numeros enteros
* se evaluan los valores de 
###So i'll count the values fo each genetical fuzzy set.
"""
#print(len(fuzzy_df.columns)//5)
##### Intiate a population
def initializePopulation():
    return [int(random.uniform(0,5)) for _ in range(len(set_fuzzy))]

def eval(individual):
    print(individual)
    id0=df[df['diagnosis']==0].index.tolist()
    id1=df[df['diagnosis']==1].index.tolist()
    bc0=1
    bct0=[]
    bct1=[]
    bc1=1
    for i in  id0:
        for j in range(len(individual)):
            bc0=bc0*set_fuzzy[j][individual[j]][i]
            print(set_fuzzy[j][individual[j]][i])
            
        bct0.append(bc0)
    for i in  id1:
        for j in range(len(individual)):
            bc1=bc1*set_fuzzy[j][individual[j]][i]
        bct1.append(bc1)
    print(bct0)
    
    #print(bct0)
    #print(bct1)
    #print(set_fuzzy[0][1][id0])
eval(initializePopulation())
    #Calculate the Bct of each class, so multiply each value of the vector 
    #Then, find the high value of both classe
    #Therefore you nee dto make the next operation:
    # CF=(TheGreatest-TheSumOfTheOtherClasses)/(The sum of each Beta's class)
#def selection():
    #Here, you can apply different views to select the better values
    
#def crossover():
    #Apply different crossover operations,/May be could be useful just a swaping values
#def mutation():
    #Apply a mutation, maybe just changing one value of the vector
#def compare():
    #Compare the values of the fathers and the children and select the best ones
#def repeat():
    #Repeat this process over and over again
