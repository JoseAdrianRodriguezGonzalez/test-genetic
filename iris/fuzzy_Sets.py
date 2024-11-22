import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz
import numpy as np
df=pd.read_csv('./iris/iris.csv')
df['y']=df['y'].map({'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3})
def corr_heat():
    corr_matrix = df.corr()

    # 2. Generar el heatmap de correlación
    plt.figure(figsize=(22, 10))  # Ajustar el tamaño de la figura
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title('Mapa de Correlación del Dataset de Cáncer de Mama')
    plt.show()
def normalize():

    scaler = MinMaxScaler()
    features=['x1','x2','x3','x4']
# Ajustar y transformar las características
    df[features] = scaler.fit_transform(df[features])

# Verifica el resultado
    
normalize()
def box(df):
    features=['x1','x2','x3','x4']
    plt.figure(figsize=(22, 10))
    sns.boxplot(df[features])
    plt.show()
def spliting(df):
    features=['x1','x2','x3','x4']
    X = df[features]  # Características seleccionadas
    y = df['y']  # Variable objetivo

# 2. Dividir el dataset en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test

#There are 30 rows on test
#And in train there are 120


def fuzzy_sets(data_):
    low = fuzz.trimf(data_, [0.0, 0.1, 0.2])
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
X_train, X_test, y_train, y_test = spliting(df)

# Paso 2: Aplicar los conjuntos difusos a las características de X_train y X_test
def apply_fuzzy_sets(X):
    fuzzy_sets_ = {}  # Diccionario para almacenar los conjuntos difusos
    for i, column in enumerate(X.columns):  # Iterar sobre las columnas de X
        fuzzy_sets_[column] = fuzzy_sets(X[column].values)  # Aplicar fuzzy_sets a cada columna
    return fuzzy_sets_


fuzzy_sets_train = apply_fuzzy_sets(X_train) 
fuzzy_sets_test = apply_fuzzy_sets(X_test)  
"""
print("\nConjuntos difusos para el conjunto de entrenamiento:")
for feature, sets in fuzzy_sets_train.items():
    print(f"\nFeature: {feature}")
    for set_name, set_values in sets.items():

print("\nConjuntos difusos para el conjunto de prueba:")
for feature, sets in fuzzy_sets_test.items():
    print(f"\nFeature: {feature}")
    for set_name, set_values in sets.items():
        print(f"  {set_name}: {set_values[:5]}...")
"""



def intialization(population):
    return [[random.randint(1, 5) for _ in range(4)] for _ in range(population)]

def Credibility(individual, x_train, y_train):
    C1 = y_train[y_train == 1].index.tolist()
    C2 = y_train[y_train == 2].index.tolist()
    C3 = y_train[y_train == 3].index.tolist()
    features = ['x1', 'x2', 'x3', 'x4']
    bct1, bct2, bct3 = [], [], []
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
    for i in C3:
        bc3 = 1
        for j in range(len(individual)):
            if individual[j] != 0:
                bc3 *= x_train[features[j]][individual[j]][i]
        bct3.append(bc3)
    Bct = [sum(bct1), sum(bct2), sum(bct3)]
    Beta = sum(value for value in Bct if value != max(Bct)) / (len(Bct) - 1)
    CF = (max(Bct) - Beta) / sum(Bct)
    # If CF is 1, all patterns are compatible with the linguistic rule R_ij belonging to the same class
    return CF, Bct.index(max(Bct)) + 1

def eval(population, x_train, y_train, x_test, y_test):
    predictions = []  # Lista para almacenar las predicciones
    features = ['x1', 'x2', 'x3', 'x4']  # Nombres de las características
    rule_predict = []
    for i in range(len(fuzzy_sets_test['x1'][1])):  # Iterar sobre cada instancia de prueba
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
pred,rule=eval([[0,0,5,0],[0,0,2,0]],fuzzy_sets_train,y_train,fuzzy_sets_test,y_test)
print(fitness(rule,pred,y_test,[[0,0,5,0],[0,0,2,0],[0,0,4,4]]))
print(y_test)
print(pred)
def selection(fitness_dict, population, tournament_size=3):
    selected = []
    valid_indices = list(fitness_dict.keys())  # Extraer índices válidos del diccionario de fitness
    for _ in range(len(population)):
        # Elegir índices de torneo del rango actual de la población
        if len(valid_indices) >= tournament_size:
            tournament_indices = random.sample(valid_indices, tournament_size)
        else:
            tournament_indices = random.sample(valid_indices, len(valid_indices))
        # Seleccionar el índice del ganador del torneo
        winner_index = max(tournament_indices, key=lambda idx: fitness_dict[idx])
        selected.append(population[winner_index])
    return selected

def crossover(parent1, parent2):
    mask = np.random.randint(0, 2, size=len(parent1))
    child1 = [p1 if m else p2 for p1, p2, m in zip(parent1, parent2, mask)]
    child2 = [p2 if m else p1 for p1, p2, m in zip(parent1, parent2, mask)]
    return child1, child2

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
def penalized_fitness(rules, predictions, y_test, population):
    fitness_dict = {}
    for i, rule in enumerate(rules):
        base_fitness,_,_ = fitness(rule, predictions[i], y_test,population)
        zero_count = rule.count(0)  # Contar ceros en la regla
        penalty = zero_count / len(rule)  # Penalización proporcional
        fitness_dict[i] = base_fitness[i] - penalty  # Penalizar reglas con muchos ceros
    return fitness_dict
def reintroduce_diversity(population, fitness_dict, generation, interval=10, random_count=5):
    """
    Reintroducir diversidad cada 'interval' generaciones.
    
    Args:
        population (list): Población actual.
        fitness_dict (dict): Diccionario de fitness.
        generation (int): Número de la generación actual.
        interval (int): Intervalo de generaciones para reiniciar.
        random_count (int): Número de individuos aleatorios a agregar.
    
    Returns:
        list: Nueva población con mayor diversidad.
    """
    if generation % interval == 0:  # Cada cierto número de generaciones
        print("Reintroduciendo diversidad en la población...")
        new_individuals = intialization(random_count)
        # Reemplazar los peores individuos con nuevos aleatorios
        worst_indices = sorted(fitness_dict, key=fitness_dict.get)[:random_count]
        for idx, new_individual in zip(worst_indices, new_individuals):
            population[idx] = new_individual
    return population
def evolution(population, x_train, y_train, x_test, y_test, mutation_rate=0.1, generations=10):
    for gen in range(generations):
        print(f"\nGeneración {gen + 1}")
        
        # Evaluación de la población
        predictions, rules = eval(population, x_train, y_train, x_test, y_test)
        
        # Calcular el fitness con penalización
        fitness_dict,_,_ = fitness(rules, predictions, y_test, population)
        
        # Imprimir las 5 mejores reglas y sus CF
        sorted_fitness = sorted(fitness_dict.items(), key=lambda x: x[1], reverse=True)
        print("Mejores 5 reglas y sus fitness:")
        for idx, fitness_value in sorted_fitness[:50]:
            best_rule = population[idx]
            CF, class_associated = Credibility(best_rule, x_train, y_train)
            print(f"Regla {best_rule} - Fitness: {fitness_value}, CF: {CF:.4f}, Class:{class_associated}")
        
        # Reintroducción de diversidad cada 10 generaciones
        population = reintroduce_diversity(population, fitness_dict, gen, interval=2, random_count=5)
        
        # Selección de padres con torneo
        selected_population = selection(fitness_dict, population, tournament_size=3)
        
        # Generación de hijos mediante cruce y mutación
        children = mutation(fitness_dict, selected_population, mutation_rate)
        
        # Comparar padres e hijos y formar la nueva población
        population = compare_and_replace(selected_population, children, fitness_dict, x_train, x_test, y_train, y_test)
    
    return population

# Ejecución del algoritmo
population = intialization(1000)
final_population = evolution(population, fuzzy_sets_train, y_train, fuzzy_sets_test, y_test)