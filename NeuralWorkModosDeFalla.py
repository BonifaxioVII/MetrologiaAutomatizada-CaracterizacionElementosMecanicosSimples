# Importar datos de Ternsorflow
from tensorflow import keras
from keras import layers

from keras import Sequential
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras import backend as K

from random import uniform, choice, randint
from math import pi, sqrt, sin, cos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product

from PGradoFunciones import EslabonBinario, convert_unit


"""Creación de datos -----------------------------------------------"""
# Crea las variables de entrada al modelo aleatoriamente
def sambling():
    """ Crea las variables deseadas que requiere el modelo aleatoriamente
    Métodos posibles: Método de MonteCarlo, Latin Hypercube, Ramdon
    
    Retorna:
    Entrada (Dimensiones): [Do, Rf, t, Dp]
        Do: Diametro del agujero (cm)
        Df: Diametro exterior (cm)
        t: Espesor (cm)
        Dp: Diametro del pin (cm)
        
    Entrada (Propiedades): [Sut, Sus, SubL, SubP, E]
        Sut: Resistencia máxima a la tracción (MPa)
        Sus: Resistencia máxima al corte (MPa)
        SubL: Resistencia máxima del rodamiento, Lug (MPa)
        SubP: Resistencia máxima del rodamiento, Pin (MPa)
        E: Modulo de elasticidad (MPa)"""
    
    """Parametros de entrada para la red neuronal:
    Caracteristicas dimensionales"""
    #Parametros de salida para la red neuronal
    
    Do = uniform(2, 10) #Numero aleatorio para el diametro interior (entre 2 y 10 cm)
    Df = uniform(Do+1, 15) #Numero aleatorio para el diametro exterior (entre Do+0.5 y 20 cm)
    t = uniform(0.05, 2.5) #Espesor aleatorio entre 0.1 y 2 cm 
    Dp = uniform(Do-1, Do-0.1) #Numero aleatorio para el diametro del pin entre 0.25 y Do
    
    #Parametros de entrada
    Eslabon = EslabonBinario(37.7952755906)
    Dimensiones = {"Do": Do, "Ancho": Df, "t": t, "Dp": Dp}
    
    """Propiedades del material y caracteristicas generales
    Valores enteramente inventados"""
    materiales = ["Acero inoxidable 304", "Aluminio 6061", "Latón"]
    material = choice(materiales)

    #Fuerza de tensión aplicada al elemento binario
    F = randint(500, 200000) #Fuerza aplicada en Newtons
    
    #Agregar información al eslabón
    Eslabon.add_dimensions(Dimensiones)
    Eslabon.add_material(material)
    Eslabon.add_force(F)
    Eslabon.Find_extra_dimensions()
    return Eslabon
         
# Crea las variables deseadas del modelo 
def modos_falla(Obj, mostrar_data = False):
    """Modos de falla
    Crea las variables de salida que requiere el modelo aleatoriamente
    
    Salida: Los factores de seguridad por cada error
        El minimo es el error más posible"""
    
    # Parametros de entrada
    Do, Df, t, Dp =  tuple(Obj.dimensiones.values()) 
    Z, a, Ls = tuple(Obj.dim_extra.values())
    Sut, Sus, Sub, E = tuple(Obj.propiedades.values())
    F = Obj.F
    
    # Parametros de salida
    Salida = []
    
    #Tensión a través de la sección de la red
    At = t * (Df - Do) #Área neta de tracción (m^2)
    Put = At * Sut #Carga de tracción última (m^2 * pa = N)
    
    #Falla por doble corte
    At = 2 * Ls * t #Área neta de corte
    Pus = At * Sus #Carga de tracción última
    
    #Falla de rodamiento
    Ab = Dp * t #Area del rodamiento (m^2)
    Sub = min([Sub, 1.5*Sut]) 
    Pub = Sub*Ab #Carga del rodamiento (N)
    
    #Falla en un unico plano
    At = a * t #Area de tensión
    Ptu = Sut * At
    
    Salida = [int(Put), int(Pus), int(Pub), int(Ptu)]

    if mostrar_data == True: 
        print(f"Dimensiones (cm): {convert_unit(Obj.dimensiones, 100, True)}")
        print(f"Propiedades (MPa): {convert_unit(Obj.propiedades, 10**6, False)}")
        print(f"Fuerza: {Obj.F} N")
        print(f"\nFallas: \n- Falla por sección: {Put} \n- Falla por doble corte: {Pus} \n- Falla por rodamiento: {Pub}")
        print("\n\n")
        Obj.nueva_imagen(True)
    
    return Salida

# Agrupa una cierta cantidad de variables de entrada y salida realizadas aleaotoriamente
def Generar_var(cant, graficar = False):
    """ Crea las variables de entrada y salida que requiere el modelo aleatoriamente
    cant: Cantidad de datos a realizar
    
    Retorna:
    Entrada (Dimensiones): [Do, Df, t, Dp]
        Do: Diametro del agujero (cm)
        Df: Diametro exterior (cm)
        t: Espesor (cm)
        Dp: Diametro del pin (cm)
        
    Entrada (Propiedades): [Sut, Sus, SubL, SubP, E]
        Sut: Resistencia máxima a la tracción (MPa)
        Sus: Resistencia máxima al corte (MPa)
        SubL: Resistencia máxima del rodamiento, Lug (MPa)
        SubP: Resistencia máxima del rodamiento, Pin (MPa)
        E: Modulo de elasticidad (MPa) """ 

    def Graficar(Do, Df, t, Dp):
        x = np.linspace(0, cant, num=len(Do))
        
        plt.scatter(x, Do)
        plt.scatter(x, Df)
        plt.scatter(x, t)
        plt.scatter(x, Dp)
        plt.plot(x,Do, x,Df, x,t, x,Dp)
        plt.xlabel('Iteraciones')
        plt.ylabel('Variables creadas (cm)')
        plt.title("Modelo alaetorio por cada iteración")
        plt.legend(["Do", "Df", "t", "Dp"])
        plt.grid()
        plt.show()
    
    Data = pd.DataFrame()
    #Propiedades dimensionales
    Data_Do, Data_Df, Data_t, Data_Dp, Data_F = [], [], [], [], []
    #Propiedades material
    Data_Sut, Data_Sus, Data_Sub, Data_E = [], [], [], []
    #Modos de falla
    Data_Falla1, Data_Falla2, Data_Falla3, Data_Falla4 = [], [], [], []
    
    iteraciones = 0
    if graficar == True: Do_s, Df_s, t_s, Dp_s, F_s = [], [], [], [], []
    while True:
        iteraciones += 1
        Obj = sambling()
        Fallas = modos_falla(Obj, graficar)
        Do, Df, t, Dp =  tuple(Obj.dimensiones.values()) 
        Sut, Sus, Sub, E = tuple(Obj.propiedades.values())
        
        Data_Do.append(round(Do*1000, 5)) #mm
        Data_Df.append(round(Df*1000, 5)) #mm
        Data_t.append(round(t*1000, 5)) #mm 
        Data_Dp.append(round(Dp*1000, 5)) #mm
        Data_F.append(Obj.F) #N
        Data_Sut.append(Sut/(10**6)) #MPa
        Data_Sus.append(Sus/(10**6)) #MPa
        Data_Sub.append(Sub/(10**6)) #MPa
        Data_E.append(E/(10**9)) #GPa
        Data_Falla1.append(Fallas[0]/(10**3)) #kN
        Data_Falla2.append(Fallas[1]/(10**3)) #kN
        Data_Falla3.append(Fallas[2]/(10**3)) #kN
        Data_Falla4.append(Fallas[3]/(10**3)) #kN
        
        if graficar == True:
            Do, Df, t, Dp = Obj.dimensiones["Do"], Obj.dimensiones["Ancho"], Obj.dimensiones["t"], Obj.dimensiones["Dp"]
            Do_s.append(Do)
            Df_s.append(Df)
            t_s.append(t)
            Dp_s.append(Dp)
            F_s.append(Obj.F)         
        if iteraciones == cant: break
        
    if graficar == True: 
        Graficar(Do_s, Df_s, t_s, Dp_s)
    
    Data["Do"], Data["Df"], Data["t"], Data["Dp"], Data["F"] = Data_Do, Data_Df, Data_t, Data_Dp, Data_F
    Data["Sut"], Data["Sus"], Data["Sub"], Data["E"] = Data_Sut, Data_Sus, Data_Sub, Data_E
    Data["F1"], Data["F2"], Data["F3"], Data["F4"] = Data_Falla1, Data_Falla2, Data_Falla3, Data_Falla4
    return Data  
  

"""Análisis y arreglo de datos   -----------------------------------------------"""
# Ver las distribuciones de falla por combinación
def ver_dist_falla(data):
    promedio_fc = [data["F1"].mean(), data["F2"].mean(), data["F3"].mean()]
    promedio_fc = [promedio_fc[0]/(10**3), promedio_fc[1]/(10**3), promedio_fc[2]/(10**3)]

    plt.figure(figsize=(10, 6))
    plt.bar(["F1", "F2", "F3"], promedio_fc)  
    plt.xlabel("Modos de falla")
    plt.ylabel("Fuerza critica [kN]")
    plt.title("Promedio de fuerza critica por modo de falla")
    plt.grid()
    plt.show()  
    
# Convertir dataframe a lista
def data_lista(data):
    entrada, salida = [], []
    for i in range(len(data)):
        eslabon = data.iloc[i]        
        entrada.append([eslabon['Do'], eslabon['Df'], eslabon["t"], eslabon["Dp"], eslabon["Sut"], eslabon["Sus"], eslabon["Sub"]])
        salida.append([eslabon["F1"], eslabon["F2"], eslabon["F3"], eslabon["F4"]])
    
    return np.array(entrada), np.array(salida)
num_entrenamiento = 10000
num_pruebas = 25
Data_train = Generar_var(num_entrenamiento, graficar = False)
ver_dist_falla(Data_train)
entrada_train, salida_train = data_lista(Data_train)

Data_test = Generar_var(num_pruebas)
entrada_test, salida_test = data_lista(Data_test)


"""Modelo neuronal   -----------------------------------------------"""
modelo = Sequential([
Dense(units=200, input_shape=[7], activation="relu"),
Dense(units=150, activation = "relu"),
Dense(units=100, activation = "relu"),
Dense(units=50, activation = "relu"),
Dense(units=25, activation = "relu"),
Dense(units=4, activation = "linear")])

modelo.summary()
plot_model(modelo)

#Compilar el modelo
modelo.compile(
    loss='mean_squared_error',
    metrics=['accuracy'],
    optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

historial = modelo.fit(entrada_train, salida_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# evaluamos el modelo
scores = modelo.evaluate(entrada_test, salida_test)
print("\n%s: %.2f%%" % (modelo.metrics_names[1], scores[1]*100))

plt.plot(historial.history["loss"][5:])
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.title("Función de perdidas RN2")
plt.show()

modelo.save('ModeloModosFalla.h5')

# Visualizar la estructura de la red
plot_model(modelo, to_file='RN2.png', show_shapes=True, show_layer_names=False)

print("Hagamos una predicción!")
resultado = modelo.predict(entrada_test)

for i in range(len(resultado)):
    print("Los valores referencia son " + str(entrada_test[i]))
    print("El resultado es " + str(resultado[i]) + " vs " + str(salida_test[i]))
