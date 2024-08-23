# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:40:40 2024

@author: JJJJJEEEESSS
"""

import pandas as pd
import numpy as np
import itertools #librería para permutaciones
import math
import random
import matplotlib.pyplot as plt #liberia para graficas

#La siguiente funcion busca la coordenada de una letra
#que se encuentra en la lista de vectores, especificamente en un vector 
#y luego la compara con el dataframe para encontrar la coordenada de
#esa letra, hace eso hasta terminar de recorrer el primer individuo y asi 
#hasta terminar todos los individuos, y entrega una lista de arreglos
# los primeros 28 valores le pertenecen al primer individuo y asi 
#sucesivamente

def coordenada(df_City,df):
    Vec_coord=list()
    for vec_lista in df_City:#individuos (vector que se mueve en la lista)
        vec_lista = vec_lista[0] #Para convertir de 2D a 1D
        #No estás usando una variable como índice en este punto porque simplemente quieres seleccionar la fila completa, no un elemento dentro de esa fila.
        #Una vez que has convertido vec_lista a un array 1D (por ejemplo, primera_fila), puedes moverte a través de sus elementos usando un bucle for.
        #Cuando haces vec_lista[0], estás accediendo a la primera fila de este array 2D(matriz).
        for vec_int in vec_lista:#vec_int es un iterador con valor de los caracteres de vec_lista. vec_lista es el vector de Caracteres/Ciudades que se mueve dentro del individuo
            for i in range(len(df)):
                if vec_int==df['Ciudad'][i]:
                    #coincidencia='son iguales'
                    res_coord=df.loc[i][['x','y']]
                    Vec_coord.append(np.array(res_coord).reshape(1,2))#Obtengo una serie por cada par de ccordenadas y era un vector de 2x1 y lo pase a 1x2 por eso el reshape
                    #print('City',vec_int)
                    #print('Dataframe',df['Ciudad'][i])
                    #print(res_coord)
    return Vec_coord

#ES IMPORTANTE MARCAR EL TAMAÑO DEL INDIVIDUO
def distance(lista_coord, ind_size=28): #ANTES ERA 28
    # Crear una matriz vacía para guardar las distancias
    matriz_distancias = []
    
    # Iterar sobre cada conjunto de coordenadas de tamaño ind_size
    for inicio in range(0, len(lista_coord), ind_size):
        distancias = []
        fin = min(inicio + ind_size, len(lista_coord))
        for i in range(inicio, fin - 1):  # Calcular distancias entre pares consecutivos dentro del individuo
            c1 = lista_coord[i]
            c2 = lista_coord[i + 1]
            # Calcula la distancia euclidiana por cada par de coordenadas
            distancia = math.sqrt(sum((a - b) ** 2 for a, b in zip(c1.T, c2.T)))
            distancias.append(distancia)
        
        # Agregar la lista de distancias calculadas para el individuo a la matriz
        matriz_distancias.append(distancias)
    
    return np.array(matriz_distancias)


#Esta función sirve para hacer la suma de cada renglon, es decir por individuo
def sum_individuos(M_individuos):
    sum_ind = []
    for i in range(len(M_individuos)):
        # Sumar los elementos de la fila i
        suma_fila = sum(M_individuos[i])
        sum_ind.append(suma_fila)
    
    # Convertir la lista sum_ind en un vector n*1 usando NumPy
    vector_sum_ind = np.array(sum_ind).reshape(-1, 1)
    
    return vector_sum_ind


def fitness(df,df_City):
    lista_coord=coordenada(df_City,df)
    M_individuos=distance(lista_coord)
    #print('Respuesta:',M_individuos)
    #print('*******SUMA DE DISTANCIAS POR INDIVIDUO********')
    SumaxIND=sum_individuos(M_individuos)
    return SumaxIND


#TORNEO
def boxing_ring(Fitness_sum,size):#entra un vector de distancias de los individuos
    
    competitors=Fitness_sum
    comp_rd=[]
    winners=[]
    winners_ind=[]
    i=2
    #T=10 o size es la misma cantidad  size es el numero de torneos
    for num_T in range(size):
        for inicio in range(i):#Tomo un vector de 2 individuos al azar #empiezas en cero y siempre es hasta n-1
            comp_rd.append(float(random.choice(competitors)))#arreglo de un valor flotante en un valor flotante
        min_values = min(comp_rd)
        min_ind=np.where(Fitness_sum==min_values)[0]
        # min_ind=Fitness_sum.index(min_values)
        comp_rd=[]
        #print('individuo ganador(futuro padre):',min_values)        
        winners.append(min_values)
        winners_ind.append(min_ind[0])#de min_values toma solo el primero si se repiten 
    winners=np.array(winners)
    winners_ind=np.array(winners_ind)#obtener los individuos que tienen ese fitnes
    return winners_ind 

#Funcion de cruza#
def cross_breeding(winners_ind,size,df_City):
    padres=[]
    indv=[]
    d1=[]
    d2=[]
    h1=[]
    h2=[]
    Lista_h1=[]
    Lista_h2=[]
    dotC=14
    PC=0.92 #probabilidad de cruce
    #For para recuperar el valor de cada invividuo ganador mediante su indice
    for i in range(len(winners_ind)):
        vec_dad=df_City[int(winners_ind[i])]   # Recupera el vector del individuo en df_City que corresponde al índice en winners_ind
        indv.append(vec_dad)  # Guarda el valor en la lista indv es mi lista de individuos (letras)
    #print('individuo:',indv)
    #indv es mi matriz de vectores de individuos con valor de letra por ejemplo: IJGHS    
    #opc2 For para seleccionar los en parejas de 2.
    # For para seleccionar los vectores en parejas del centro hacia afuera
    for i in range(size // 2):
        d1.append(indv[len(indv) // 2 - 1 - i])  # Tomar del centro hacia la izquierda
        d2.append(indv[len(indv) // 2 + i])      # Tomar del centro hacia la derecha
        val_ale=random.random() #Probabilidad de que se crucen 
        if val_ale<PC:
            #Crear hijo 1
            for j in range(dotC):
                h1.append(d1[i][0,j]) #[0,j] renglo y j se mueve en las columnas
            for L in d2[i][0]:
                if len(h1) < 28 and L not in h1:
                    h1.append(L)
                    
            #Crear hijo 2
            for j in range(dotC):
                h2.append(d2[i][0,j]) #[0,j] renglo y j se mueve en las columnas
            for L in d1[i][0]:
                if len(h2) < 28 and L not in h2:
                    h2.append(L)    
                    
            Lista_h1.append(h1)
            Lista_h2.append(h2)
            h1=[]
            h2=[]
        else:#Pasen los papas directos
            Lista_h1.append(np.squeeze(d1[i]).tolist())
            Lista_h2.append(np.squeeze(d2[i]).tolist())
            h1=[]
            h2=[]
    

        
    #print("Lista hijos 1:", Lista_h1)
    #print("Lista hijos 2:", Lista_h2)
            
    return Lista_h1,Lista_h2



#######FUNCION MUTACION###### cruza1 son mis hijos 1
def mutation(Cruza1,Cruza2):
    pm=0.45
    Hijos_sin_mutar=Cruza1+Cruza2
    H_mutados=[]
    for z in range(len(Hijos_sin_mutar)):  
        val_ale=random.random() #Probabilidad de que se muten   
        if val_ale<pm:
            i, j = random.sample(range(len(Hijos_sin_mutar[z])), 2)# z 0 a 27
            Hijos_sin_mutar[z][i],Hijos_sin_mutar[z][j]=Hijos_sin_mutar[z][j],Hijos_sin_mutar[z][i]
            H_mutados.append(Hijos_sin_mutar[z])
        else:
            H_mutados.append(Hijos_sin_mutar[z])
            #print('holi')
            
    return H_mutados

###### FUNCION REEMPLAZO#####
def replace(Mutar,df_City,df,size):
    #Ind_F=np.array(Mutar)+np.squeeze(df_City)
    Ind_F = np.vstack((np.array(Mutar), np.squeeze(df_City)))
    # Convertir 'array_of_objects' a una lista de arreglos NumPy
    Ind_F = [np.array(obj).reshape(1,28) for obj in Ind_F]
    Best_D=fitness(df,Ind_F)#Mejores distancias de mis individuos
    #Aqui empiezo a modificar para unir a Ind_F con Best_D
    Ind_F = np.array([np.array(obj).reshape(1, -1) for obj in Ind_F])
    # Asegura que Ind_F tenga una forma adecuada (n, 28)
    Ind_F = np.squeeze(Ind_F)
    Ind_F = np.hstack((Best_D, Ind_F))
    indices_ordenados = np.argsort(Ind_F[:, 0])
    # Reordenar el arreglo usando los índices de menor a mayor
    Ind_F = Ind_F[indices_ordenados]
    Best_Ind_Fit=Ind_F[:size]
    Best_Ind=Best_Ind_Fit[:,1:] #para quitarle la columna de resultados del fitness

    return Best_Ind,Best_Ind_Fit





                          ############ MAIN ###########
# leemos los datos
#RECUERDA DESCARGAR EL CSV ACTUALIZADO
df = pd.read_csv("C:/Users/JJJJJEEEESSS/OneDrive - Universidad de Guanajuato/MAESTRIA 2 CUATRI/Computación flexible/Soft_Computer_Code/P4/Ciudades.csv")



#Inicialización de la población.
City=np.array(df['Ciudad'])#OFICIAl PARA 28 individuos
#City=np.array(df['Ciudad'].iloc[:6])#PRUEBA
size=250#Son 2 individuos pero puedo poner más
df_City=[]
mov_intern=0
G=100
Number_ONEs_graf=[]

#Creacion de individuos
for k in range(size):
    df_City.append(np.random.permutation(City).reshape(1,len(City)))#Cada individuo es una lista y cada lista tiene un vector por lo que mis operaciones seran para las listas
    

                        ####Ciclo de generaciones####
#extrañamente no funciona con el ciclo REVISAR ESO linea 106 al parecer 
for g in range(G):
    #Utilizar funcion fitness
    Fitness_sum=fitness(df,df_City)
    #for i, suma in enumerate(Fitness_sum, start=1):
    #    print(f"Individuo {i}: {suma[0]}")
        
    #Implementamos la seleccion y torneo
    winners_ind=boxing_ring(Fitness_sum,size)
    #print('Indice de los individuos random Ganadores de los combates:',winners_ind)
        
    Cruza1,Cruza2=cross_breeding(winners_ind,size,df_City)#Regresa la lista de los hijos
    #print('Cruza total:',Cruza1,Cruza2)
    
    Mutar=mutation(Cruza1, Cruza2)
    #print('MUTACION TOTAL:',Mutar)
    
    #Funcion replace
    Best_individuos,Best_Fit=replace(Mutar,df_City,df,size)
    #El mejor de todos es el primer valor que regresa de la funcion replace
    Number_ONE=Best_Fit[0,0]
    #Guardar valores para graficar posteriormente
    Number_ONEs_graf.append(Number_ONE)
    #Se adapta el tipo de dato que arroja Best_individuos para que tenga
    #el mismo tipo el nuevo df_City.
    df_City = [np.array(row).reshape(1, 28) for row in Best_individuos]


#Graficar los valores de Number_ONE(contiene la mejor distancia por generacion)
plt.figure(figsize=(10, 6))
plt.plot(range(1, G + 1), Number_ONEs_graf, marker='o', linestyle='-')
plt.title('Evolución de Number_ONE a lo largo de las generaciones')
plt.xlabel('Generación')
plt.ylabel('Distancia')
plt.grid(True)
plt.show()

