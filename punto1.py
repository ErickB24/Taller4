# IMPORTS
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# RECONSTRUIR IMAGEN
def recreate_image(centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))  # Inicializar en un arreglo de zeros la imagen
    label_idx = 0
    for i in range(rows):                       # Para cada fila
        for j in range(cols):                   # Para cada columna
            image_clusters[i][j] = centers[labels[label_idx]]   # Asignar color
            label_idx += 1                      # label +1
    return image_clusters


print('inserte el directorio de la imagen:')    # Mensaje en consola de insertar directorio de la imagen
path = input()                                  # Directorio ingresado por el usuario
print(
    'escriba el metodo de segmentacion a utilizar (gmm o kmeans):')  # Mensaje en consola de elección del metodo de segmentación
metodo = input()                                # Metodo de segmentación a utilizar

path_file = os.path.join(path)                  # Lectura del directorio de la imagen
image = cv2.imread(path_file)                   # Lectura de la imagen
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Re-ordenar para matplot
image = np.array(image, dtype=np.float64) / 255 # Normalizar imagen
rows, cols, ch = image.shape                    # Tamaño de la imagen en filas, columnas y capas
image_array = np.reshape(image, (
rows * cols, ch))                   # Cambiar el tamaño de la imagen con respecto a las filas,columnas y en cual capa
image_array_sample = shuffle(image_array, random_state=0)[:10000]  # Tomar 10k muestras de la matriz de la imagen

C_r = []                                        # Inicializar el vector en 0
n_centroides = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Vector con cada uno de los centroides que se obtendran

for i in range(0, 10):                          # For para cambiar el numero de centros
    n_colors = i + 1        # Numero del centro que se evalua y se le adiciona 1 para que empiece desde 1 y no desde 0

    if (metodo == 'gmm'):                                            # Metodo gaussian mean
        modelo = GMM(n_components=n_colors).fit(image_array_sample)  # Exraer el modelo gmm con pocas muestras
        labels = modelo.predict(image_array)                         # Etiquetas para cada uno de los pixeles
        centros = modelo.means_                                      # Centroides

    if (metodo == 'kmeans'):                                         # Metodo kmeans
        modelo = KMeans(n_clusters=n_colors, random_state=0).fit(
            image_array_sample)                 # Extraer el modelo kmean con pocas muestras
        labels = modelo.predict(image_array)    # Etiquetas para cada uno de los pixeles
        centros = modelo.cluster_centers_       # Centros de los clousters

    tam = len(image_array)   # Extraer el tamaño de la imagen
    DIS = 0                  # Inicializar DIS en 0
    for m in range(0, tam):  # For para recorrer cada uno de los pixeles y cada uno de los labels
        dis = abs(centros[labels[m]] - image_array[m])      # Distancia entre el centro y cada uno de los pixeles
        DIS = DIS + np.linalg.norm(dis)                     # Norma de la distancia
    C_r.append(DIS)          # Acumular el valor de DIS

print(C_r)                   # Imprimir el valor de C_r

plt.figure(0)
plt.plot(n_centroides,C_r)   # Grafica de la suma de distancias entre clusters con respecto al numero de centros
plt.xlabel("Numero de Centros")                         # Titulo X
plt.ylabel("Suma de distancias intra-cluster")          # Titulo Y
plt.title('Suma de Distancias vs Numero de centros')    # Titulo de la imagen

# Visualizar imagen original
plt.figure(1)                           # Numero de la figura
plt.clf()
plt.axis('off')                         # Quitar ejes
plt.title('Original image')             # Titulo
plt.imshow(image)                       # Mostrar Imagen Original

#Visualizar imagen segmentada
plt.figure(2)                           # Numero de la figura
plt.clf()
plt.axis('off')                         # Quitar ejes
plt.title('Quantized image ')           # Titulo de la Imagen
plt.imshow(recreate_image(centros, labels, rows, cols)) # Recrear Imagen y mostrarla
plt.show()                              # Mostrar todas las Imagenes en Pantalla

# PATH de prueba C:\Users\Erick\Desktop\OCTAVO SEMESTRE\PROC DE IMAGENES\bandera.png