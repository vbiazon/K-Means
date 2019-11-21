# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:14:13 2019

@author: Victor Biazon
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import math
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder

def DistEuclid(pontoA, pontoB, eixos):
    accum = 0
    for i in range(0, eixos):
        accum += (pontoA[i]-pontoB[i])**2
    return math.sqrt(accum)


def K_means(data, K, dim, dimx, dimy):
    
    #separando variaveis independentes e dependentes
    X = np.asanyarray(data[:,:]) #separa as variaveis independentes no vetor X 
    
    #Atribuindo K's aleatorios
    Km = np.zeros((K,len(X[0])),float)
    
    #Atribuindo os centroides a pontos aleatorios na observações
    Xsamples = np.copy(X)        
    for i in range(0, len(Km)):
        Index = random.randint(0,len(Xsamples)-1)
        Km[i] = Xsamples[Index]
        np.delete(Xsamples, Index, 0)
    del Xsamples
    
    #   plot de dados sem divisao de classes e ponto Kmean aleatorios   
    plt.figure()
    plt.scatter(X[:,dimx], X[:,dimy], color = 'red', cmap = 'rainbow')
    plt.scatter(Km[:,dimx], Km[:,dimy], color = 'black', marker = 'X')
    plt.title('K-means - Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()   
    
    execute_kmean = True
    
    #executa enquanto nao se estabiliza a posição dos centroides
    while(execute_kmean):   
        Km2 = np.copy(Km)
        # atribuição de classes de acordo com a proximidade do ponto de outros pontos Kmean
        Y_class = np.zeros_like(X[:,0], float)
        for i in range(0, len(X)):
            minD = math.inf
            for j in range(0, len(Km)):
               D = DistEuclid(X[i,:], Km[j], dim)
               if minD > D:
                   minD = D
                   Y_class[i] = j + 1
                   
        
        #ajuste das posições dos pontos Kmean
        for i in range(0, len(Km)):
            for j in range(0, len(X[0])):
                Km[i,j] = np.mean(X[Y_class == i + 1][:,j])
      
        #   plot de dados com divisao de classes e ponto Kmean ajustados 
        plt.figure()
        plt.scatter(X[:,dimx], X[:,dimy], c = Y_class, cmap = 'rainbow')
        plt.scatter(Km[:,dimx], Km[:,dimy], color = 'black', marker = 'X')
        plt.title('K-means - Data')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
        
        #verifica se a posição anterior dos centroides foi modificada.
        if (Km == Km2).all(): execute_kmean = False
    
    return
    
    
    
#main
    
data = pd.read_table('Iris.txt', decimal  = ",")

dataset = np.asarray(data.iloc[:,:-1])

K_means(dataset, 3, 3, 0, 1)

data = pd.read_table('Mall_Customers.csv', sep = ",")

dataset = np.asarray(data.iloc[:,[3,4]]) 

K_means(dataset, 5, 2, 0, 1)

data = pd.read_table('wine.csv', decimal = ".", sep = ";", header = None)

dataset = np.asarray(data.iloc[:,:-6]) 

K_means(dataset, 3, 7, 0, 1)