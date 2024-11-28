#!/usr/bin/env python
# coding: utf-8

# In[310]:


import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzzy
import numpy as np
#Read the dataset
df=pd.read_csv('data.csv')


# In[311]:


# 1. Eliminar la columna 'id' si está presente (es irrelevante para el análisis)
if 'id' in df.columns:
    df = df.drop(columns=['id'])


# 4. Convertir la columna 'diagnosis' en valores numéricos (si aún no está convertida)
# 'M' (maligno) lo codificamos como 1 y 'B' (benigno) como 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})


# Verificar si alguna columna tiene varianza cero (constantes)


# 1. Calcular la matriz de correlación


# In[312]:


corr_matrix = df.corr()

# 2. Generar el heatmap de correlación
plt.figure(figsize=(22, 10))  # Ajustar el tamaño de la figura
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Mapa de Correlación del Dataset de Cáncer de Mama')
plt.show()
# El dataset ahora está limpio y listo para su análisis


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
features=['perimeter_mean', 'concave points_mean', 'radius_worst',
       'perimeter_worst', 'concave points_worst']

X = df[features]  # Características seleccionadas
y = df['diagnosis']  # Variable objetivo

# 2. Dividir el dataset en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar un modelo de Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. Hacer predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# 5. Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Mostrar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.show()


# In[344]:


from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=3)
X_selected = selector.fit_transform(X, y)

# Obtener los nombres de las características seleccionadas
selected_features = X.columns[selector.get_support()]
print("Características seleccionadas:", selected_features)


# In[357]:


from sklearn.preprocessing import MinMaxScaler

# Seleccionar las características que deseas normalizar
features_2 =['radius_mean', 'perimeter_mean', 'area_mean', 'concave points_mean',
       'radius_worst', 'perimeter_worst', 'area_worst',
       'concave points_worst']
features=['perimeter_mean', 'concave points_mean', 'radius_worst',
       'perimeter_worst', 'concave points_worst']

# Crear una instancia de MinMaxScaler
scaler = MinMaxScaler()

# Ajustar y transformar las características
df[features] = scaler.fit_transform(df[features])

# Verifica el resultado
print(df[features].head())


# In[408]:


import numpy as np
import skfuzzy as fuzz
def spliting(df):
    features=['perimeter_mean', 'concave points_mean', 'radius_worst',
       'perimeter_worst', 'concave points_worst']

    X = df[features]  # Características seleccionadas
    y = df['diagnosis']  # Variable objetivo

# 2. Dividir el dataset en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def fuzzy_sets(data_):
    low = fuzz.trimf(data_, [0.0, 0.0, 0.2])
    very_low = fuzz.trimf(data_, [0.0, 0.20, 0.4])
    medium = fuzz.trimf(data_, [0.2, 0.4, 0.6])
    high = fuzz.trimf(data_, [0.4, 0.60, 0.8])
    very_high = fuzz.trimf(data_, [0.6, 0.8, 1.0])
    
    fuzzy_data = {
        1: very_low,
        2: low,
        3: medium,
        4:high,
        5:very_high
    }

    return fuzzy_data


# In[413]:


X_train, X_test, y_train, y_test = spliting(df)

# Paso 2: Aplicar los conjuntos difusos a las características de X_train y X_test
def apply_fuzzy_sets(X):
    fuzzy_sets_ = {}  # Diccionario para almacenar los conjuntos difusos
    for i, column in enumerate(X.columns):  # Iterar sobre las columnas de X
        fuzzy_sets_[column] = fuzzy_sets(X[column].values)  # Aplicar fuzzy_sets a cada columna
    return fuzzy_sets_

fuzzy_sets_train = apply_fuzzy_sets(X_train) 
fuzzy_sets_test = apply_fuzzy_sets(X_test)  


# In[318]:


print(y_test.iloc[6])


# # GA

# ## Inicializar la poblacion

# 

# In[410]:


def initializePopulation(pop_size):
    return [[int(random.randint(0,2)) for _ in range(len(features))] for _ in range(pop_size)]
print(initializePopulation(5))


#  <!-- evaluar -->

# In[381]:


lista=[4, 2, 0, 2, 0]
fuzzy_results_test[features[0]]


# In[411]:


def intialization(population):
    return [[random.randint(0, 5) for _ in range(4)] for _ in range(population)]


# ## Evaluation

# In[419]:


def Credibility(individual, x_train, y_train):
    C1 = y_train[y_train == 0].index.tolist()
    C2 = y_train[y_train == 1].index.tolist()
    features=['perimeter_mean', 'concave points_mean', 'radius_worst',
       'perimeter_worst', 'concave points_worst']

    bct1, bct2= [], [] 
    for i in C1:
        bc1 = 1
        for j in range(len(individual)):
            if individual[j] != 0:
                bc1 *= x_train[features[j]][individual[j]][i]
        bct1.append(bc1)
    for i in C2:
        bc2 = 1
        for j in range(len(individual)):
            if individual[j] != 0:
                bc2 *= x_train[features[j]][individual[j]][i]
        bct2.append(bc2)
    Bct = [sum(bct1), sum(bct2)]
    Beta = sum(value for value in Bct if value != max(Bct)) / (len(Bct) - 1)
    CF = (max(Bct) - Beta) / sum(Bct)
    # If CF is 1, all patterns are compatible with the linguistic rule R_ij belonging to the same class
    return CF, Bct.index(max(Bct)) + 1


# In[324]:


print(type(y_test))


# In[416]:


def eval(population, x_train, y_train, x_test, y_test):
    predictions = []  # Lista para almacenar las predicciones
    features=['perimeter_mean', 'concave points_mean', 'radius_worst',
       'perimeter_worst', 'concave points_worst']
  # Nombres de las características
    rule_predict = []
    for i in range(len(fuzzy_sets_test['perimeter_mean'][1])):  # Iterar sobre cada instancia de prueba
        Alpha_T = []  # Almacenar valores Alpha para esta instancia
        Classess = []  # Almacenar clases asociadas a cada regla
        new_rule = []

        for rules in population:  # Iterar sobre cada regla en la población
            CF, Class = Credibility(rules, x_train, y_train)  # Calcular CF y clase de la regla
            Alpha = 1  # Reiniciar Alpha para cada regla
            if not(np.isnan(CF)):
                for x in range(len(rules)):  # Multiplicar las funciones de membresía para las características
                    if rules[x] != 0:
                        Alpha *= x_test[features[x]][rules[x]][i]
                new_rule.append(rules)
                # Guardar Alpha y CF
                Alpha_T.append(Alpha * CF)
                Classess.append(Class)

        # Determinar la clase de la instancia con el Alpha más alto
        max_alpha = max(Alpha_T)
        Classification_class = Classess[Alpha_T.index(max_alpha)]
        predictions.append(Classification_class)
        rule_predict.append(new_rule[Alpha_T.index(max_alpha)])

    return predictions, rule_predict


# In[ ]:


def fitness(rules, predictions, y_test, population):
    fitness_dictio = {}
    for index, rule in enumerate(population):
        indexes = [i for i, r in enumerate(rules) if r == rule]
        fitness = 0
        correct = 0
        wrong = 0
        
        for i in indexes:
            if predictions[i] == y_test.iloc[i]:
                correct += 1
            else:
                wrong += 1
        fitness = correct
        fitness_dictio[index] = fitness
    
    return fitness_dictio,correct,wrong


# ## selecton

# In[396]:


def selection(fitness_dict, population, tournament_size=5):
    """
    Selección por torneo: selecciona individuos a través de competencias entre un subconjunto de la población.
    
    Args:
        fitness_dict (dict): Diccionario con índices de la población como claves y su fitness como valores.
        population (list): Lista de individuos de la población actual.
        tournament_size (int): Tamaño del torneo (número de individuos que compiten en cada torneo).
    
    Returns:
        list: Una nueva población seleccionada mediante el método de torneo.
    """
    selected = []
    population_indices = list(range(len(population)))  # Indices de la población
    
    for _ in range(len(population)):
        # Seleccionar `tournament_size` individuos al azar para el torneo
        tournament_indices = random.sample(population_indices, tournament_size)
        # Obtener el índice del ganador (el de mayor fitness)
        winner_index = max(tournament_indices, key=lambda idx: fitness_dict[idx])
        # Añadir el ganador a la población seleccionada
        selected.append(population[winner_index])
    
    return selected


# # Cruza

# In[399]:


def crossover(parent1, parent2):
    mask = np.random.randint(0, 2, size=len(parent1))
    child1 = [p1 if m else p2 for p1, p2, m in zip(parent1, parent2, mask)]
    child2 = [p2 if m else p1 for p1, p2, m in zip(parent1, parent2, mask)]
    return child1, child2


# In[400]:


def mutation(dictionary, population, mutation_rate=0.05):
    pairs = selection(dictionary, population)
    children = []
    for p in range(len(pairs) - 1):
        child1, child2 = crossover(pairs[p], pairs[p + 1])
        for i in range(len(child1)):
            if np.random.random() < mutation_rate:
                child1[i] = np.random.randint(0, 5)
        for i in range(len(child2)):
            if np.random.random() < mutation_rate:
                child2[i] = np.random.randint(0, 5)
        children.extend([child1, child2])
    return children


# In[401]:


def compare_and_replace(parents, children, fitness_dict, x_train, x_test, y_train, y_test):
    new_population = []
    # Comparar fitness de los padres y los hijos y seleccionar los mejores
    children_predictions, children_rules = eval(children, x_train, y_train, x_test, y_test)
    children_fitness,_,_ = fitness(children_rules, children_predictions, y_test, children)
    
    # Comparar fitness de los padres y los hijos
    for i in range(len(parents)):
        parent_fitness = fitness_dict.get(i, 0)  # Fitness de los padres
        child_fitness = children_fitness.get(i, 0)  # Fitness de los niños

        # Selecciona el mejor entre el padre y el hijo
        if child_fitness > parent_fitness:
            new_population.append(children[i])
        else:
            new_population.append(parents[i])

    return new_population



# In[ ]:


nueva_poblacion = generar_nueva_poblacion(padres_seleccionados, k=2)
print("Nueva población:", nueva_poblacion)


# In[402]:


def evolucionar_poblacion(population, num_generaciones=10, k=2):
    """
    Evoluciona la población durante un número determinado de generaciones.
    """
    # Evaluamos la población inicial
    poblacion,fitness_poblacion = eval_population_sort(population)

    # Iteramos sobre el número de generaciones
    for generacion in range(num_generaciones):
        print(f"\nGeneración {generacion + 1}")

        # Selección, cruce y reemplazo para formar la nueva generación
        nueva_poblacion = generar_nueva_poblacion(poblacion, k)
        
        # Opcional: imprime el mejor individuo de la generación actual
        mejor_individuo = max(nueva_poblacion, key=lambda ind: ind[1])
        print(f"Mejor individuo de la generación {generacion + 1}: {mejor_individuo[0]} con fitness {mejor_individuo[1]}")

        # Actualizamos la población con los nuevos individuos y sus fitness
        poblacion = [ind[0] for ind in nueva_poblacion]
        fitness_poblacion = [ind[1] for ind in nueva_poblacion]

    # Retornamos la última población con los individuos y sus fitness
    return poblacion, fitness_poblacion
    # Retornamos la última generación
  


# In[422]:


def evolution(population, x_train, y_train, x_test, y_test, mutation_rate=0.1, generations=10):
    for gen in range(generations):
        print(f"\nGeneración {gen + 1}")
        
        # Evaluación de la población
        predictions, rules = eval(population, x_train, y_train, x_test, y_test)
        
        # Calcular el fitness
        fitness_dict,correct,wrong = fitness(rules, predictions, y_test, population)
        
        # Selección de padres
        sorted_fitness = sorted(fitness_dict.items(), key=lambda x: x[1], reverse=True)

        # Imprimir las 5 mejores reglas y sus CF
        print("Mejores 5 reglas y sus fitness:")
        for idx, fitness_value in sorted_fitness[:20]:
            best_rule = population[idx]
            CF, class_associated = Credibility(best_rule, x_train, y_train)
            print(f"Regla {best_rule} - Fitness: {fitness_value}, CF: {CF:.4f}, Class:{class_associated}")
        
        # Selección de padres
        selected_population = selection(fitness_dict, population)
        
        # Generación de hijos mediante cruce y mutación
        children = mutation(fitness_dict, selected_population, mutation_rate)
        
        # Comparar padres e hijos y formar la nueva población
        population = compare_and_replace(selected_population, children, fitness_dict, x_train, x_test, y_train, y_test)
    return population

# Ejecución del algoritmo
population = intialization(1000)
final_population = evolution(population, fuzzy_sets_train, y_train, fuzzy_sets_test, y_test)


# In[430]:


pred,rules=eval([[0,0,0,0],[3,1,0,0],[0,1,4,1],[4,5,1,1],[1,1,0,2]],fuzzy_sets_train,y_train,fuzzy_sets_test,y_test)
print(fitness(rules,pred,y_test,[[0,0,0,0],[3,1,0,0],[0,1,4,1],[4,5,1,1],[1,1,0,2]]))

