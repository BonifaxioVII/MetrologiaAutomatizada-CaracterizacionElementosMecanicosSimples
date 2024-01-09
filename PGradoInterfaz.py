print("1. Descargando modulos...")
import PGradoFunciones as P

from os import system
from tensorflow import keras
from keras.models import load_model

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

print("2. Descargando redes neuronales... ")
"""------------------------------------ Descargar redes neuronales"""
modelo1 = load_model('ModeloMedidasUtiles.h5')
modelo2 = load_model('ModeloModosFalla.h5')


print("3. Cargando interfaz...")
"""------------------------------------ Funciones de interfaz"""
instrucciones = """
Seleccione una imagen para su análisis
Recomendaciones:
1. El objeto deberá estar sobre una hoja tamaño carta.
2. El objeto deberá estar sobre una superficie de alto contraste.
3. Tenga precaución con el brillo y las sombras.
"""

system("cls")
def crearVentana():
	#Crear ventana
	ventana = tk.Tk()

	#Dimensiones de la ventana
	ventana.geometry("500x650")

	#Etiqueta de la ventena
	ventana.title("Analizador de eslabones binarios")

	return ventana

def mostrar_imagen(imagen, V, tamaño):
    imagen.thumbnail(tamaño)  # Redimensiona la imagen para ajustarla en la ventana
    imagen = ImageTk.PhotoImage(imagen)
    imagen_x = tk.Label(V)
    imagen_x.config(image=imagen)
    imagen_x.image = imagen  # Mantén una referencia para evitar que la imagen sea eliminada por el recolector de basura
    return imagen_x

def seleccionar_imagen():
    global imagen_path
    file_path = filedialog.askopenfilename()
    if file_path:
        imagen_path = file_path
        img = Image.open(imagen_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        imagen_label.config(image=img)
        imagen_label.image = img
        analizar_boton.pack()  # Muestra el botón de análisis
     
def paso_a_paso(Fig):
    info_pasos = {1: ("Paso #1: Imagen original", "No se identificó correctamente \nla imagen original.", (1,0)),
                2: ("Paso #2: Resaltar contornos de hoja", "No fue posible aplicar \nlos arreglos a la imagen original.", (1,1)),
                3: ("Paso #3: Identificar contornos relevantes", "No se hallaron los bordes \nde la hoja carta.", (1,2)),
                4: ("Paso #4: Realizar transformación de perspectiva", "No se alineó correctamente \nlos bordes de la hoja tamaño carta.", (3,0)),
                5: ("Paso #5: Imagen blanco y negro con correcciones", "No se pudo binarizar la imagen.", (3,1)),
                6: ("Paso #6: Resaltar contornos del eslabon", "No se hallaron \nlos contornos del eslabon.", (3,2))}
              
    #Mostrar las imagenes 1 a 1
    def mostrar_paso_a_paso(): Fig.mostrar_paso_a_paso()
      
    # Interfaz y foto por falla
    def MostrarFalla(paso):
        info = info_pasos[paso]
        
        interfaz = tk.Label(V2_1, text=info[0], justify=tk.LEFT, font=("Arial", 7)) 
        interfaz.grid(row=info[-1][0], column=info[-1][1], padx=10, pady=10, sticky="nw")	
        
        try: 
            if paso == 1: img = Fig.img_original
            elif paso == 2: img = Fig.img_adapted
            elif paso == 3: img = Fig.img_original_silueta
            elif paso == 4: img = Fig.img_alineada
            elif paso == 5: img = Fig.img_alineada_silueta
            elif paso == 6: img = Fig.img_alineada_cnts
            
            img = Image.fromarray(img)
            imagen_x = mostrar_imagen(img, V2_1, (150, 150))
            imagen_x.grid(row=info[-1][0] + 1, column=info[-1][1], padx=10, pady=10, sticky="nw")	
            return True
            
        except:  
            error = tk.Label(V2_1, text=info_pasos[paso][1], justify=tk.LEFT, font=("Arial", 10), fg = "red") 
            error.grid(row=info[-1][0] + 1, column=info[-1][1], padx=10, pady=10, sticky="nw")	
            return False
           
    #Presentación de esta pestaña    
    V2_1 = tk.Toplevel(V1)
    V2_1.title("Paso a paso enumerado de las medidas útiles")

    interfaz = tk.Label(V2_1, text="Paso a paso: ", font=("Arial", 15)) 
    interfaz.grid(row=0, column=0, padx=10, pady=10, sticky="nw")	

    #Agregar información paso por paso
    for paso in range(1, 7):
        if MostrarFalla(paso) == False: return

    # 7. Paso    
    data = f"""Paso #7: Medidas útiles del eslabón:
    - Distancia entre centros: {round(float(Fig.MedidasUtiles["Dc"]),3)} cm
    - Diametro interno: {round(float(Fig.MedidasUtiles["Do"]),3)} cm
    - Diametro externo | Ancho: {round(float(Fig.MedidasUtiles["Ancho"]),3)} cm  """
    interfaz = tk.Label(V2_1, text=data, justify=tk.LEFT, font=("Arial", 10)) 
    interfaz.grid(row=5, column=0, padx=10, pady=10, sticky="nw")
    try: Fig.MedidasUtiles
    except: 
        error = tk.Label(V2_1, text="", justify=tk.LEFT, font=("Arial", 10), fg = "red") 
        error.grid(row=6, column=0, padx=10, pady=10, sticky="nw")
        return
    
    # Botón para ver todas las imagenes
    boton_continuar = tk.Button(V2_1, text="Presione aqui para ver todas las imagenes unas por una.", command = mostrar_paso_a_paso)
    boton_continuar.grid(row=7, column=0, padx=10, pady=10, sticky="nw") 
    
def analizar_fallas(Obj, t, F, Dp, material):    
    #Actualizar datos
    t, F, Dp, material = t.get(), F.get(), Dp.get(), material.get()
    
    #Opciones de error
    if t == "" or F == "" or Dp == "": 
        messagebox.showerror("No se han añadido todas las caracteristicas extra del eslabón.")
        return
    
    try: t, F, Dp = float(t), float(F), float(Dp)
    except: 
        messagebox.showwarning("Las caracteristicas no pueden ser strings")
        return
    
    #Caracterizar Obj
    Obj.add_dimensions({"Dp": float(Dp), "t": float(t)})
    Obj.add_force(float(F)/1000)
    Obj.add_material(material)
    Obj.Find_extra_dimensions()

    F = Obj.F
    W = Obj.dimensiones['Ancho']
    t = Obj.dimensiones['t']
    Dh = Obj.dimensiones['Do']
    Dp = Obj.dimensiones['Dp']
    DL = Obj.dimensiones['Ancho'] - Obj.dimensiones['Do']
    Dc = Obj.dimensiones['Ancho']

    # Mostrar los resultados en una nueva ventana
    V3 = tk.Toplevel(V1)
    #Dimensiones de la ventana
    V3.geometry("600x700")
 
    V3.title("Resultados del Análisis de Fallas")
        
    fuerzas_criticas = Obj.modos_falla(modelo2)
    fuerzas_criticas = fuerzas_criticas[0]
    
    # Crear un canvas para mostrar los resultados con scrollbar
    canvas = tk.Canvas(V3)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar = tk.Scrollbar(V3, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.config(yscrollcommand=scrollbar.set)
    
    # Crear un frame dentro del canvas para los resultados
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor=tk.NW)
        
    #Bienvenida al usuario
    Presentacion = tk.Label(frame, text="Analisis de fallas y fuerzas criticas", font=("Arial", 20)) 
    Presentacion.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
    interfaz = tk.Label(frame, text="El analisis de fuerzas se hizo mediante una red neuronal desarrollada de 1 dimensión. \nPara más detalles consulte la guía. \n\nEl eslabón seleccionado por uste tiene las siguientes caracteristicas: ", justify=tk.LEFT, font=("Arial", 10)) 
    interfaz.grid(row=1, column=0, padx=10, pady=10, sticky="nw")
    
    #Mostrar imagen de información 
    data = Image.open("ModosFalla\Data.png")
    imagen_x = mostrar_imagen(data, frame, (250, 250))
    imagen_x.grid(row=2, column=0, padx=10, pady=10, sticky="nw")	
    
    #Mostrar información general del eslabón
    resultados = f"""Caracteristicas dimensionales del eslabón: 
    - Fapp: {round(float(F*1000),3)} kN
    - W: {round(float(W*100), 3)} cm
    - t: {round(float(t*100), 3)} cm
    - D0: {round(float(Dh*100), 3)} cm
    - Dp: {round(float(Dp*100), 3)} cm
    - DL: {round(float(DL*100), 3)} cm
    - Dc: {round(float(Dc*100), 3)} cm
Otras dimensiones:
    - Z (Perdida de longitud en el plano de corte): {round(float(Obj.dim_extra['Z']*100), 3)} cm
    
Propiedades del material:  {Obj.material}
Supondremos que el pin es del mismo material que el eslabón.
    - Resistencia a la tracción (Sut): {float(Obj.propiedades["Sut"]/(10**6))} MPa
    - Resistencia al corte (Ssu): {float(Obj.propiedades["Sus"]/(10**6))} MPa: 
    - Resistencia al rodamiento del eslabón (Sbru): {float(Obj.propiedades["Sub"]/(10**6))} MPa
    - Modulo de elasticidad (E): {float(Obj.propiedades["E"]/(10**9))} GPa
    
Para cada modo de falla se encontraron los siguientes resultados: """
    
    resultado_label = tk.Label(frame, text=resultados, justify=tk.LEFT, font=("Arial", 7))
    resultado_label.grid(row=3, column=0, padx=10, pady=10, sticky="nw")
    
    # Mostrar los modos de falla 
    valores_falla = {1: ["Falla de tensión en la sección neta", "ModosFalla\F1.png", fuerzas_criticas[0], round(float(fuerzas_criticas[0]/(Obj.F * 1000)), 3), (5, 0)],
                    2: ["Falla por doble corte", "ModosFalla\F2.png", fuerzas_criticas[1], round(float(fuerzas_criticas[1]/(Obj.F * 1000)), 3), (5, 5)],
                    3: ["Falla de cizallamiento", "ModosFalla\F3.png", fuerzas_criticas[2], round(float(fuerzas_criticas[2]/(Obj.F * 1000)), 3), (10, 0)],
                    4: ["Falla por hendedura", "ModosFalla\F4.png", fuerzas_criticas[3], round(float(fuerzas_criticas[3]/(Obj.F * 1000)), 3), (10, 5)]}
    
    for falla in range(1, 5):
        F = valores_falla[falla]
        Data = f""" {F[0]}
        Fuerza critica (Fc): {F[2]} kN
        Factor de seguridad (FS): {F[3]} """
        interfaz = tk.Label(frame, text=Data, justify=tk.LEFT, font=("Arial", 7)) 
        interfaz.grid(row=F[-1][0], column=F[-1][1], padx=10, pady=10, sticky="nw")
        
        F1 = Image.open(F[1])
        imagen_F1 = mostrar_imagen(F1, frame, (200, 200))
        imagen_F1.grid(row=F[-1][0] + 1, column=F[-1][1], padx=10, pady=10, sticky="nw")
    
    # Configurar el tamaño del canvas
    frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    if DL > Dh: messagebox.showwarning("Un diametro del pasador superior al diametro interior revelará resultados erroneos.")
def analizar_imagen():  
    if imagen_path: pass
    else: messagebox.showwarning("No ha seleccionado imagen. Intentelo de nuevo")
    
    # Mostrar los resultados en una nueva ventana
    V2 = tk.Toplevel(V1)
    V2.title("Resultados del Análisis Dimensional")

    Fig = P.AnalisisFigura(imagen_path, 2, modelo1)
    
    #Bienvenida al usuario
    Presentacion = tk.Label(V2, text="Analisis de eslabones", font=("Arial", 20)) 
    Presentacion.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
    interfaz = tk.Label(V2, text="El analisis de eslabones se hizo mediante arreglos morfologicos a la imagen original. \nPara más detalles consulte la guía.", justify=tk.LEFT, font=("Arial", 10)) 
    interfaz.grid(row=1, column=0, padx=10, pady=10, sticky="nw")
    
    try: Fig.proceso(conf = True)
    except: 
        error = tk.Label(V2, text="Hubo un error durante el procedimiento.", font=("Arial", 15), fg = "red") 
        error.grid(row=5, column=0, padx=10, pady=10, sticky="nw")
        
        boton_salir = tk.Button(V2, text="Presione para mirar el paso a paso del problema", command=lambda: paso_a_paso(Fig))
        boton_salir.grid(row=10, column=0, padx=10, pady=10, sticky="nw")
        
        V1.mainloop()
            
    resultados = "Medidas útiles del eslabón: " 
    resultados += f"\nDc: {round(float(Fig.MedidasUtiles['Dc']),5)} cm"
    resultados += f"\nDo: {round(float(Fig.MedidasUtiles['Do']),5)} cm"
    resultados += f"\nAncho: {round(float(Fig.MedidasUtiles['Ancho']),5)} cm"
    
    resultado_label = tk.Label(V2, text=resultados, justify=tk.LEFT)
    resultado_label.grid(row=5, column=0, padx=10, pady=10, sticky="nw")

    #Crear eslabon como clase
    eslabon = P.EslabonBinario(Fig.dim_img)
    eslabon.add_dimensions(Fig.MedidasUtiles)
    eslabon.nueva_imagen(False)

    #Mostrar bosquejo de eslabón
    img = Image.fromarray(eslabon.img2D)
    imagen_x = mostrar_imagen(img, V2, (300, 300))
    imagen_x.grid(row=6, column=0, padx=5, pady=5, sticky="nw")
    
    # Botón para ver el proceso de la imagen
    boton_seleccionar = tk.Button(V2, text="Presione aquí si desea conocer el paso a paso de estos calculos.", command=lambda: paso_a_paso(Fig))
    boton_seleccionar.grid(row=0, column=1, padx=10, pady=10, sticky="nw")  
        
    # Elementos extra
    # Espesor
    valor_T_label = tk.Label(V2, text="Espesor: Ingrese el valor del " + "espesor" + " del eslabón binario en (cm):")
    valor_T_label.grid(row=10, column=0, padx=10, pady=20, sticky="nw")  
    valor_T_entry = tk.Entry(V2)
    valor_T_entry.grid(row=11, column=0, padx=10, pady=10, sticky="nw")  
    
    # Diametro del pin
    valor_Dp_label = tk.Label(V2, text="Diametro del pin: Ingrese el valor del " + "diametro del pin" + " en (cm):")
    valor_Dp_label.grid(row=10, column=1, padx=10, pady=20, sticky="nw")  
    valor_Dp_entry = tk.Entry(V2)
    valor_Dp_entry.grid(row=11, column=1, padx=10, pady=10, sticky="nw")  
    
    # Fuerza
    valor_F_label = tk.Label(V2, text="Fuerza: Ingrese el valor de la " + "fuerza" + " del eslabón binario en (kN):")
    valor_F_label.grid(row=15, column=0, padx=10, pady=20, sticky="nw")  
    valor_F_entry = tk.Entry(V2)
    valor_F_entry.grid(row=16, column=0, padx=10, pady=10, sticky="nw") 
    
    # Material
    opcion_label = tk.Label(V2, text="Seleccione el material del eslabón:")
    opcion_label.grid(row=15, column=1, padx=10, pady=20, sticky="nw")  
    opciones = ["Acero inoxidable 304", "Aluminio 6061", "Latón"]
    opcion_var = tk.StringVar(V2)
    opcion_var.set(opciones[0])  # Valor por defecto
    opcion_menu = tk.OptionMenu(V2, opcion_var, *opciones)
    opcion_menu.grid(row=16, column=1, padx=10, pady=10, sticky="nw")  

    # Botón para continuar al analisis de falla
    boton_continuar = tk.Button(V2, text="Presione aqui para continuar con el analisis de fallas.", command = lambda: analizar_fallas(eslabon, valor_T_entry, valor_F_entry, valor_Dp_entry, opcion_var))
    boton_continuar.grid(row=1, column=1, padx=10, pady=10, sticky="nw") 
    
    
    
    
"""------------------------------------ Interfaz principal."""
V1 = crearVentana()
imagen_path = None

#Bienvenida al usuario
Presentacion = tk.Label(V1, text="¡Bienvenido al Analizador de Figuras!", font=("Arial", 20)) 
Presentacion.pack(side=tk.TOP, padx=20, pady=20)
interfaz = tk.Label(V1, text=instrucciones, justify=tk.LEFT, font=("Arial", 10)) 
interfaz.pack(side=tk.TOP, padx=20, pady=10)

#Imagen de muestra
imagen_label = tk.Label(V1)
imagen_label.pack(side=tk.TOP, padx=20, pady=10)

# Botón para seleccionar la imagen
boton_seleccionar = tk.Button(V1, text="Seleccionar Imagen", command=seleccionar_imagen)
boton_seleccionar.pack(side=tk.TOP, padx=20, pady=10)	

# Botón para analizar la imagen (inicialmente oculto)
analizar_boton = tk.Button(V1, text="Presione aquí para analizar la imagen", command=analizar_imagen)
analizar_boton.pack(side=tk.TOP, padx=20, pady=10)	
analizar_boton.pack_forget()  # Oculta el botón al principio

#Mostrar ventana
V1.mainloop()





