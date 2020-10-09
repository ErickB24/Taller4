# Imports
import cv2
import numpy as np
import os



pos = []                                # Vector vacio para guardar posicion de la primera imagen
pos1 = []                               # Vector vacio para guardar posicion de la segunda imagen

# Metodos para la entrada del Mouse
def raton(event,x,y,flags,param):       # evento del raton
    global pos, posNp                   # Variables de posicion como globales
    if event == cv2.EVENT_LBUTTONDOWN:  # Si el boton izquierdo es presionado
        pos.append((x,y))               # acumular puntos
        posNp = np.array(pos)           # pasar posicion como arreglo
        print(posNp)                    # Mostrar las coordenadas presionadas

def raton2(event,x,y,flags,param):      # Evento del raton para la segunda imagen
    global pos1,posNp1                  # Variables de posicion como globales

    if event == cv2.EVENT_LBUTTONDOWN:  # Si se presiona el boton derecho
        pos1.append((x,y))              # Acumular el valor de coordenadas x,y
        posNp1 = np.array(pos1)         # Valor en arreglo
        print(posNp1)                   # Mostrar el valor acumulado


# IMAGEN 1
print('inserte el directorio de la imagen:')    # Aviso para el usuario
path = input()                                  # Path insertado por el usuario
path_file = os.path.join(path)                  # Entrar al path
image = cv2.imread(path_file)                   # Leer la imagen
print('seleccione SOLO 3 PUNTOS en la imagen y luego cierrela:') # Aviso para el usuario
cv2.imshow('Image', image)                      # Mostrar Imagen
cv2.namedWindow('Image')                        # Nombre de la imagen
cv2.setMouseCallback('Image', raton)            # Eventos del Mouse
cv2.waitKey(0)                                  # Esperar a que la imagen se cierre


# COORDENADAS 1
pts1=np.float32([posNp[0],posNp[1],posNp[2]]) # Con los valores acumulados crear un vector de puntos
print (pts1)                                        # Imprimir valor de los primeros puntos

# IMAGEN 2
print('inserte el directorio de la segunda imagen:')    # Aviso para el usuario
path = input()                                          # Path insertado por el usuario
path_file = os.path.join(path)                          # Acceder al archivo
image2 = cv2.imread(path_file)                          # Leerla como imagen
print('seleccione SOLO 3 PUNTOS en la imagen y luego cierrela:')         # Aviso para el usuario

cv2.imshow('Image1', image2)                            # Mostrar imagen
cv2.namedWindow('Image1')                               # Nombre de la imagen
cv2.setMouseCallback('Image1', raton2)                  # Evento del raton para la imagen 2
cv2.waitKey(0)                                          # Esperar a que se cierre la imagen 2


# COORDENADAS 2
pts2=np.float32([posNp1[0],posNp1[1],posNp1[2]])  # Con los valores acumulados crear un vector de puntos
print (pts2)                                            # Imprimir valor de los segundos 3 puntos


M_affine =np.float32(cv2.getAffineTransform(pts1, pts2))    # Obtener la matriz de transformacion affine a partir de los puntos
image_affine = cv2.warpAffine(image, M_affine, image.shape[:2])  # Aplicarla a la imagen
cv2.imshow('Image affine', image_affine)                    # Mostrar imagen con transformacion Affine


# Calculo de matriz de similitud
sx=np.float32(np.sqrt((M_affine[0,0])**2+(M_affine[1,0])**2))  # De la Matriz Affine obtener el escalamiento en x
sy=np.float32(np.sqrt((M_affine[0,1])**2+(M_affine[1,1])**2))  # De la Matriz Affine obtener el escalamiento en y

theta_rad = np.float32(np.arctan2(M_affine[1,0],M_affine[0,0]))     # De la Matriz Affine obtener el angulo de rotacion
theta = np.float32(theta_rad * 180 / np.pi)                         # Angulo en Grados

tx = np.float32(((M_affine[0,2]*np.cos(theta_rad))-(M_affine[1,2]*np.sin(theta_rad)))/sx)  # De la Matriz Affine obtener la traslacion en x
ty = np.float32(((M_affine[0,2]*np.sin(theta_rad))+(M_affine[1,2]*np.cos(theta_rad)))/sy)  # De la Matriz Affine obtener la traslacion en y

M_sim = np.float32([[sx * np.cos(theta_rad), -np.sin(theta_rad), tx],
                    [np.sin(theta_rad), sy * np.cos(theta_rad), ty]])   # Crear matriz de similitud
image_similarity = cv2.warpAffine(image, M_sim, image.shape[:2])        # Aplicar la matriz de similitud a la imagen 1

# ERROR entre Norma de los pixeles

M_sim_T= np.append(M_sim,[[0,0,1]], axis= 0)
coor1 = pts1.transpose()                                # Primeras coordenadas transpuestas
Hom = np.append(coor1,[[1,1,1]],axis = 0)               # Añadir 1 a los vectores para poder multiplicar
T = np.matmul(M_sim,Hom)                                # Aplicar la matriz similitud a los puntos
P1 = T[:-1,:].transpose()                               # Volver a transponer para poder restar
Error = np.linalg.norm(P1-pts2, axis = 1)               # norma del error entre puntos 1 y 2
print(Error)                                            # Imprimir el Error para cada coordenada


cv2.imshow('Image similitud', image_similarity)         # Mostrar Imagen con transformacion de similitud
cv2.imshow('Lena warped', image2)                       # Mostrar Imagen 2
cv2.waitKey(0)                                          # Esperar a que se cierre

# ejemplo de path = C:\Users\Erick\Desktop\OCTAVO SEMESTRE\PROC DE IMAGENES\lena_warped.png