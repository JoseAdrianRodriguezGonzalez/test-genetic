import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzzy
import random
#Read the dataset
df=pd.read_csv('./classifier_cancer/data.csv')
#Change the values to 0 and 1
def conversion(df):
    df=df.astype('category')
    df=df.cat.codes
    return df
df['diagnosis']=conversion(df['diagnosis'])
#I eliminate the index
df.drop(columns='id',axis=1)
#With a heatmap I can know the most adecuateted variables
plt.figure(figsize=(10,20))
sns.heatmap(df.corr(),annot=True, fmt=".2f", cmap='coolwarm')
plt.title('heat map')
#plt.show()

selected_features = [
    'concave points_worst', 'area_worst', 'perimeter_worst', 'radius_worst',
    'concave points_mean', 'concavity_mean', 'perimeter_mean', 'radius_mean'
]
y = df['diagnosis']

def scaling(df,columns):
    scaler=MinMaxScaler()
    scaled_df=scaler.fit_transform(df)
    scaled_df=pd.DataFrame(scaled_df,columns=columns)
    return scaled_df
norm=scaling(df[selected_features],selected_features)

##Creating the fuzzy sets

def fuzzySets(df):
    fuzzy_data={}
    low=fuzzy.trimf(df,[0,.10,.20])
    low_m=fuzzy.trimf(df,[0.10,.25,.40])
    medium=fuzzy.trimf(df,[0.30,.45,.60])
    medium_h=fuzzy.trimf(df,[.5,.65,.80])
    high=fuzzy.trimf(df,[0.7,.85,1])
    fuzzy_data[0] = low
    fuzzy_data[1] = low_m
    fuzzy_data[2] = medium
    fuzzy_data[3] = medium_h
    fuzzy_data[4] = high
    # Crear un DataFrame con los conjuntos difusos
    fuzzy_df = pd.DataFrame(fuzzy_data)
    return fuzzy_df
x1=fuzzySets(df['concave points_worst'])
print(x1)



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
    return [int(random.uniform(0,5)) for _ in range(len(selected_features))]
#def eval():
    #* First step: Divide the dataset in 2 classes
    #*Then, divide the columns according its feature
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