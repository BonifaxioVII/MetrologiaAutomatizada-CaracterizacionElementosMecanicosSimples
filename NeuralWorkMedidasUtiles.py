# Importar datos de Ternsorflow
import tensorflow as tf

import random
from math import pi, sqrt
import matplotlib.pyplot as plt
import numpy as np

def Generar_var(cant):
    """ Crea las variables de entrada y salida que requiere el modelo aleatoriamente
    cant: Cantidad de datos a realizar
    
    Retorna:
    Entrada: [[Dc, PerimetroInt, PerimetroExt], [], [], ...] donde P1 y P2 son  el centro de gravedad de los 2 circulos interiores
    Salida: [[Do, Ancho], [], [], ...] """
    #Parametros de entrada
    Entrada = []

    # Parametros de salida
    Salida = []

    while True:
        #Parametros de salida para la red neuronal
        Ro = (random.randint(1,8)) #Numero aleatorio para el radio interior
        Rf = (random.randint(Ro+1,15)) #Numero aleatorio para el ancho del eslabón
                
        #Parametros de entrada para la red neuronal
        Dc = (random.randint(Ro+Rf+1,30))
        AreaNodos = (pi*Ro**2)
        AreaExt = ((pi*Rf**2) + (Dc*Rf*2)) - (2*AreaNodos) 
        
        Entrada.append([Dc, AreaNodos, AreaExt])
        Salida.append([2*Ro, 2*Rf]) #Se multiplican por 2 pues queremos hallar los diametros
        if len(Entrada) == cant: break
    
    return np.array(Entrada, dtype=float), np.array(Salida, dtype=float)

num_entrenamiento = 10000
num_pruebas = 15
data_in_train, data_out_train = Generar_var(num_entrenamiento)
data_in_test, data_out_test = Generar_var(num_pruebas)

# Entrenar modelos
modelo = tf.keras.Sequential([

#Aprendizaje artificial
tf.keras.layers.Dense(units=120, input_shape=[3]),
tf.keras.layers.Dense(units=60, activation="relu"),
tf.keras.layers.Dense(units=30, activation="relu"),
tf.keras.layers.Dense(2)
])

#Compilar el modelo
modelo.compile(
    loss='mean_squared_error',
    optimizer="adam",
    metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=70, restore_best_weights=True)

historial = modelo.fit(data_in_train, data_out_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# evaluamos el modelo
scores = modelo.evaluate(data_in_test, data_out_test)
print("\n%s: %.2f%%" % (modelo.metrics_names[1], scores[1]*100))

print("Hagamos una predicción!")
resultado = modelo.predict(data_in_test)

for i in range(len(resultado)):
    print("Los valores referencia son  " + str(data_in_test[i]))
    print("El resultado es " + str(resultado[i]) + " vs " + str(data_out_test[i]))

plt.plot(historial.history["loss"])
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.title("Función de perdidas RN1")
plt.show()

# Visualizar la estructura de la red
tf.keras.utils.plot_model(modelo, to_file='RN1.png', show_shapes=True, show_layer_names=False)

modelo.save('ModeloMedidasUtiles.h5')