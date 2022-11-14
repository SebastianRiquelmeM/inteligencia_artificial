import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
dataset = pd.read_csv("./dataset/train22k.csv")
#print(dataset.head())
dataset.shape
from sklearn.neighbors import KNeighborsClassifier
""" el siguiente es para asociar strings a numeros """
from sklearn.preprocessing import LabelEncoder


#print(type(dataset.iloc[0,1]))
#print(isinstance(dataset.iloc[0,1], str))
#print(dataset.head())
""" Declaro el encoder de string a numeros """
encoder = LabelEncoder()

""" For para iterar el dataset """
for i in range(15):
     """  
     En realidad el dataset es un dataframe de pandas, este tiene propiedades para
     acceder a él. 'iloc' sirve para acceder a él como una matriz con indices
     de fila y columna [x,y].

     Con isisntance puedo ver si la primera fila de datos en cada columna es un string,
     ya que en estos casos debo aplicar la conversion a números """
     if isinstance(dataset.iloc[0,i], str):
          #print(dataset.columns[i], dataset.iloc[0,i])
          #print("type column: ", type(dataset.columns[i])," column:", dataset.columns[i])
          #dataset[dataset.columns[i]] = encoder.fit_transform(dataset.columns[i])
          """
          Con dataset.columns[i] obtengo el nombre de la columna,
          luego convierto esa columna completa en números """
          string = dataset.columns[i]
          dataset[string] = encoder.fit_transform(dataset[string])

#print(dataset.head())
#print(dataset)
#encoder.classes_

def entrenar(dataset, vecinos):

     x = dataset.iloc[:, :14].values

     y = dataset['salary'].values

     knn = KNeighborsClassifier(n_neighbors=vecinos)

     knn.fit(x, y)
     #Realizo una predicción

     dataset_validation = pd.read_csv("./dataset/validation.csv")

     for i in range(15):
          if isinstance(dataset_validation.iloc[0,i], str):
               string = dataset_validation.columns[i]
               dataset_validation[string] = encoder.fit_transform(dataset_validation[string])

     #print(dataset_validation.head())

     x_validation = dataset_validation.iloc[:, :14].values
     y_validation = dataset_validation['salary'].values

     """ Los datos de salario <=50k son 0 y los >50 son 1 """

     """ for i in range(20): 
          if(knn.predict(x_validation[i].reshape(1,-1))[0] == 1):
               print(">50")
          else:
               print("<=50k") """
          #print(knn.predict(x_validation[i].reshape(1,-1))[0])
     prediccion_correcta = []

     for i in range(len(x_validation)): 
          resultado = knn.predict(x_validation[i].reshape(1,-1))[0]
          if resultado == y_validation[i]:
               prediccion_correcta.append(resultado)
          
     #print(  (len(prediccion_correcta)/len(y_validation))*100, "% de presición"  )  
     return (len(prediccion_correcta)/len(y_validation))*100

actual = 0
i_actual = 0

for i in range(100000):
     nuevo = entrenar(dataset, i+1)
     if actual < nuevo:
          actual = nuevo
          i_actual = i+1
          print("n_vecinos:optimo: ", i_actual, " precisión: ", actual )
     print("procesando n_actual: ", i+1, "n_ptimo: ",i_actual," Precisión_actual: ", actual )
print("Cantidad de vecinos con mayor precision: ")
print("i: ", i_actual, " n_vecinos: ", actual )



#python myscript > output.txt
#python3 vecino_optimo.py > output.txt