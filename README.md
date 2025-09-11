# NeuralNetworkTR
Red Neuronal para mi Trabajo de investigación. El trabajo que acompaña este código se puede encontrar en ¿Qué hay detrás de una red neuronal?.pdf 

## Índice: 
1. Dependencias
2. Uso de la red neuronal
    1. Entrenar red neuronal
    2. Comprobar red neuronal


## Dependencias

Para usar la red neuronal hacen falta 3 dependencias: 
- [Numpy](https://numpy.org/install/)
- [Matplotlib](https://matplotlib.org/stable/users/getting_started/)
- [mnist](https://pypi.org/project/mnist/)

Todas ellas se pueden instalar con `pip`.

## Uso de la red neuronal

### Entrenar red neuronal

Ejecutando el archivo `main.py`, empezará a entrenar la red neuronal. Por cada iteración, se imprime a la consola la iteración en la que está, el coste y las activaciones de esa iteración:

![Screenshot_20230826_133706](https://github.com/davoriols/NeuralNetworkTR/assets/35429501/4ccf3ea3-870e-4737-abe8-bb6c5cef6614)

Por cada iteración, se imprime la iteración en la que está, en este caso, entre 1097 y 1100, el coste, separado por la flecha, y debajo de todo una lista con las activaciones de la última capa de la red neuronal. 

Esta lista es el resultado de la red neuronal, donde el número más alto, es el que la red neuronal ha detectado. Esta lista empieza por el 0, de forma que en esta lista la red neuronal piensa que es un 5. 

Esto es esperado, porque aunque se entrene con todas las imágenes, el coste que se muestra en la terminal es para la primera imagen de los datos mnist, que es un 5: 

![Screenshot_20230826_135338](https://github.com/davoriols/NeuralNetworkTR/assets/35429501/71cfef9c-1d10-4328-834f-47e93458df6c)

Una vez la red neuronal es entrenada, se te pide escoger un número entre 0 y 9999. Este número se usa para mostrar una de las 10000 imágenes de prueba, que la red neuronal no ha visto antes, para verificar su entrenamiento. 

![Screenshot_20230826_135953](https://github.com/davoriols/NeuralNetworkTR/assets/35429501/d4984a83-29a9-4888-b7fd-8e6a954a2c7a)
![Screenshot_20230826_140048](https://github.com/davoriols/NeuralNetworkTR/assets/35429501/66b02622-ddf6-4e1e-b88e-b82fae3279b8)

Vemos que se nos muestra la etiqueta de la imagen, es decir, el número de la imagen, y el número que la red neuronal piensa que hay. En este caso son el mismo. 

---

### Comprobar la red neuronal

Podemos comprobar la precisión de nuestra red neuronal ejecutando el archivo `test.py`. Pero antes de ello, podemos configurarlo: 

Dentro del archivo, en las líneas 10 y 13, podemos cambiar las variables `useTrainData` y `showResults`. Podemos modificar `useTrainData` para usar pesos y sesgos de una red neuronal previamente entrenada.
Los datos de la red previamente entrenada están en la carpeta `trainedData`. Si esta variable se asigna a `False`, se usan los datos de la red entrenada con `main.py`, cuyos datos están en la carpeta `data`. 

Podemos cambiar `showResults` para ver en qué imágenes se ha equivocado la red neuronal: 

![Screenshot_20230826_141014](https://github.com/davoriols/NeuralNetworkTR/assets/35429501/395cad9c-ddec-491a-b0e3-82721f662881)

Si asignamos `showResults` a `False`, estas imágenes no se mostrarán y se seguirá con el programa, que una vez acabado, muestra el porcentaje de aciertos de la red neuronal 
y te vuelve a pedir un número entre 0-9999 para ver alguna imagen en específico.

![Screenshot_20230826_141552](https://github.com/davoriols/NeuralNetworkTR/assets/35429501/60e6bee4-ed97-423e-8198-24c0eefd379f)


