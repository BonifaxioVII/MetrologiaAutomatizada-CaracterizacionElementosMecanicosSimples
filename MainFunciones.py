import cv2
import numpy as np
from math import sqrt, pi

#Data para la interfaz
from os import system


"""------------------------------------ Funciones Secundarias dependientes."""
# Hace una conversión entre la imagen real y un hoja blanca A4
def ReglaDeTres(self, valor, inverso=False, potencia = 1):
	"""valor: valor del pixel al que se convertirá en real
	inverso=False: Si inverso es verdadero, valor es el valor real que se convertirá en pixeles
 	"""
	#result (cm) = valor (pix) / self.dim_img (pix/cm) 
	#potencia == 2: result (cm^2) = valor (pix^2) / self.dim_img**2 (pix^2/cm^2) = pix^2*cm^2/pix^2 = cm^2 
 
	if inverso == True: return valor*(self.dim_img**potencia)
	else: return valor/(self.dim_img**potencia)

#Convertir una caracteristica de orden de unidades
def convert_unit(dictionary, unit, multiplicar = True):
	"""unit: hace referencia a la constante que se va a multiplicar/dividir por los valores del diccionario
	multiplicar: True (multiplica el valor del dict por unit. De lo contrario, lo dividirá)"""
	dict_extra = {}
	for key, value in dictionary.items():
		dict_extra[key] = value * unit if multiplicar == True else value / unit

	return dict_extra


"""------------------------------------ Clase Eslabon."""
class EslabonBinario():
	def __init__(self, dim_img):
		"""Crea un objeto propio al eslabon """
		self.dim_img = dim_img	
		self.dim = [int(21.59*dim_img), int(27.94*dim_img)] #[27.94, 21.59] Hoja tamaño carta
		self.dimensiones = {}
		self.dim_extra = {}
		self.propiedades = {}

	#Agregar dimensiones
	def add_dimensions(self, Dimensiones:dict):
		"""Añade la longitud de las dimensiones del eslabon [Dc, Do, Ancho, Dp]
  		Entrada (Dimensiones): [Do, Ancho, t, Dp]
        Do: Diametro del agujero (cm)
        Ancho: Diametro exterior (cm)
        t: Espesor (cm)
        Dp: Diametro del pin (cm)"""
        
		Dim_param = ["Dc", "Do", "Ancho", "Dp", "t"]
  
		for param in Dim_param:
			try:
				parametro_convert = Dimensiones[param]/100 #m
				self.dimensiones[param] = parametro_convert
			except: pass
   
	#Agregar fuerza
	def add_force(self, F):
		self.F = F

	#Agregar material
	def add_material(self, material):
		"""Añade las propiedades de material del eslabon
  		material: Elije entre 3 posibles materiales"""
		self.material = material
        
		if material == "Acero inoxidable 304": 
			Sut = 515 #Resistencia a la tracción (MPa)
			Sus = 590 #Resistencia al corte MPa (Mpa)
			Sub = 300 #Resistencia al rodamiento (MPa) Lug
			E = 193 #Modulo de elasticidad (Gpa)

		elif material == "Aluminio 6061":
			Sut = 310 #Resistencia a la tracción (MPa)
			Sus = 276 #Resistencia al corte MPa (Mpa)
			Sub = 70 #Resistencia al rodamiento (MPa) Lug
			E = 68 #Modulo de elasticidad (Gpa)

		else:
			Sut = 425 #Resistencia a la tracción (MPa)
			Sus = 330 #Resistencia al corte MPa (Mpa)
			Sub = 140 #Resistencia al rodamiento (MPa) Lug
			E = 96 #Modulo de elasticidad (Gpa)
   
		Sut, Sus, Sub, E = Sut*10**6, Sus*10**6, Sub*10**6, E*10**9
		self.propiedades = {"Sut": Sut, "Sus": Sus, "Sub": Sub, "E": E} #Pa

	#Encontrar dimensiones extra
	def Find_extra_dimensions(self):
		#Extra dimensiones
		Ro, Df, t, Dp = self.dimensiones["Do"]/2, self.dimensiones["Ancho"], self.dimensiones["t"], self.dimensiones["Dp"]

		Z = Df - np.sqrt((Df)**2 - ((Dp/2)*np.sin(2*pi/9))**2) #Perdida de longitud en el plano de corte
		a = Df - Ro #Distancia desde el borde del eslabon hasta el borde del agujero
		Ls = a + (Dp/2) * (1 - np.cos(2*pi/9)) - Z #largo neto del plano de corte

		self.dim_extra["Z"] = round(Z,3)
		self.dim_extra["a"] = round(a,3)
		self.dim_extra["Ls"] = round(Ls,3)

	#Encontrar modos de falla
	def modos_falla(self, modelo):
		Do, Ancho = self.dimensiones["Do"]*1000, self.dimensiones["Ancho"]*1000
		t, Dp = self.dimensiones["t"]*1000, self.dimensiones["Dp"]*1000
		Sut, Sus, Sub = self.propiedades["Sut"]/(10**6), self.propiedades["Sus"]/(10**6), self.propiedades["Sub"]/(10**6)
     
		historial = np.array([Do, Ancho, t, Dp, Sut, Sus, Sub], dtype=float)
		resultado = modelo.predict(np.array([historial]))
   
		self.fuerzas_criticas = resultado
		return resultado

	#Crear imagen nueva
	def nueva_imagen(self, mostrar = False, bicolor = False):     
		#Crea la imagen
		img_eslabon = np.ones((self.dim[0],self.dim[1], 3),np.uint8)
		img_eslabon[:,:,:] = (255,255,255)

		#Valores utiles en pixeles
		x, y = self.dim[1]/2, self.dim[0]/2
		try: w, h = ReglaDeTres(self, self.dimensiones["Dc"]*100/2, True, 1), ReglaDeTres(self, self.dimensiones["Ancho"]*100/2, True, 1)
		except: w, h = 0, ReglaDeTres(self, self.dimensiones["Ancho"]*100/2, True, 1)

		r = ReglaDeTres(self, self.dimensiones["Do"]*100, True, 1)
		x, y, w, h, r = int(x), int(y), int(w), int(h), int(r/2)
		
		#Dibuja el rectangulo principal del eslabon
		if w != 0: cv2.rectangle(img_eslabon, (x-w, y-h), (x+w, y+h), 0, -1)
		else: cv2.rectangle(img_eslabon, (0, y-h), (x, y+h), 0, -1)
  
		#Dibuja los circulos exteriores
		cv2.circle(img_eslabon, (x-w, y), h, (0,0,0), -1)
		cv2.circle(img_eslabon, (x+w, y), h, (0,0,0), -1)
  
		#Dibuja los circulos interiores
		cv2.circle(img_eslabon, (x-w, y), r, (255,255,255), -1)
		cv2.circle(img_eslabon, (x+w, y), r, (255,255,255), -1)

		#Producir imagenes de entrenamiento
		if bicolor == False:
			#Dibujar escala
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.line(img_eslabon, (0,0), (int(ReglaDeTres(self, 5, True)), 0), (199, 0, 57), 5)
			cv2.putText(img_eslabon, f"5 centimetros", (20, 20), font, 0.3, (199, 0, 57), 1, cv2.LINE_AA)

			try: #Dibujar fuerza
				max_arrow = 50
				cv2.arrowedLine(img_eslabon, (x+w+h, y), (x+w+h+(max_arrow), y), (120, 40, 140), 2, cv2.LINE_4)
				if w != 0: cv2.arrowedLine(img_eslabon, (x-w-h, y), (x-w-h-(max_arrow), y), (120, 40, 140), 2, cv2.LINE_4)

				cv2.putText(img_eslabon, f"F = {self.F/1000} kN", (x+w+h+(max_arrow), y+int(max_arrow/2)), font, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
				if w != 0: cv2.putText(img_eslabon, f"F = {self.F/1000} kN", (x-w|-h-(max_arrow), y-(max_arrow)), font, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
			except: pass

		else: 
			a = 4

		try: #Dibujar pin
			Dp = int(ReglaDeTres(self, self.dimensiones["Dp"]*100/2, True))
			cv2.circle(img_eslabon, (x-w, y), Dp, (0,255,255), -1)
			cv2.circle(img_eslabon, (x-w, y), Dp, (0,255,255), -1)
		except: pass
  
		self.img2D = img_eslabon
		if mostrar == True:
			# 9. Paso
			cv2.imshow("Imagen reformada", self.img2D)
			cv2.waitKey(0) 
  
  
"""------------------------------------ Clase para analizar figura."""		
class AnalisisFigura():
	#Quita sombras y recalca fronteras
	def ArreglosImagen(self, img):
		rgb_planes = cv2.split(img)
		result_norm_planes = []
		for plane in rgb_planes:
			# Dilatación para realzar detalles
			dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8))

			# Creación de un fondo suavizado
			bg_img = cv2.GaussianBlur(dilated_img, (5,5), 0)

			# Diferencia entre el plano original y el fondo suavizado
			diff_img = 255 - cv2.absdiff(plane, bg_img)

			# Normalización para mejorar la visualización
			norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
			result_norm_planes.append(norm_img)
		
		# Combinación de los planos para obtener la imagen final
		return cv2.merge(result_norm_planes)
	
	def mostrar_paso_a_paso(self):
		try:
			print("Paso a paso enumerado:")
			
			# 1. Paso
			cv2.imshow("1. Imagen original", self.img_original)
			print("\n1.(init) Lee la imagen original")
			cv2.waitKey(0) 

			# 2. Paso
			cv2.imshow("2. Imagen original con arreglos", self.img_adapted)
			print("\n2.(ArreglosImagen) Le hace arreglos de imagen a la imagen original")
			cv2.waitKey(0) 

			# 3. Paso
			cv2.imshow("3.Silueta de la imagen", self.img_original_silueta)
			print("\n3.(ArreglosImagen) Mediante Canny (función con la cual se hallan los bordes de la imagen) se identifican los posibles contornos de la imagen ")
			cv2.waitKey(0) 

			# 4. Paso
			cv2.imshow("4. Imagen alineada", self.img_alineada)
			print("\n4.(Trans_Perspectiva)  Transforma la imagen de acuerdo con los bordes de la hoja de papel")
			cv2.waitKey(0) 

			# 5. Paso
			cv2.imshow("5. Imagen alineada con arreglos de imagen", self.img_alineada_filtro)
			print("\n5.(manipular_img) Mediante GaussianBlur se hacen arreglos a la imagen para evitar desperfectos")
			cv2.waitKey(0)
		
			# 6. Paso
			cv2.imshow("6. Imagen alineada blanco y negro con correcciones", self.img_alineada_silueta)
			print("\n6.(manipular_img) Transforma la imagen mediante una serie de correcciones de color y brillo")
			cv2.waitKey(0) 
		
			# 6.5 Paso
			try:
				cv2.imshow("6.5 Imagen alineada blanco y negro con correcciones y bit_not", self.img_alineada_silueta_inv)
				print("\n6.5 (manipular_img) Convierte el fondo en negro")
				cv2.waitKey(0) 
			except: pass 

			# 7. Paso
			cv2.imshow("7. Contornos de la imagen", self.img_alineada_cnts)
			print("\n7.(momentos_geometricos) Muestra los contornos de imagen identificados de acuerdo con la jerarquia de la imagen. Además agrega sus respectivos centroides encontrados con los momentos de imagen")
			cv2.waitKey(0) 

			#Ejecución de salida
			input("\nPresiona cualquier tecla para continuar:")
			cv2.destroyAllWindows()
			return

		except: print("Hubo un error en este paso") 

	#Funcion de inicio
	def __init__(self, ruta, escala, modelo) -> None:
		"""ruta: Ruta de la imagen a analizar
  		escala: El tamaño de la imagen a mostrar en relación con la imagen original"""
		#Importar imagen  
		imagen = cv2.imread(ruta) 
		imagen = cv2.resize(imagen, dsize = (int(imagen.shape[1]/2), int(imagen.shape[0]/2)))
		self.img_original = imagen.copy()
		dim_img = (1/escala)*37.7952755906 
  
		self.dim = [int(21.59*dim_img), int(27.94*dim_img)] #[27.94, 21.59] Hoja tamaño carta ; [x, y] pixeles = xCarta (cm) * Fs (pix/cm)
		self.dim_img = dim_img #pix/cm
		self.modelo_medidas = modelo
  
		self.img_adapted = self.ArreglosImagen(self.img_original)
		canny = cv2.Canny(self.img_adapted,10,250)
		kermel = np.ones((3,3),np.uint8)
		self.img_original_silueta = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kermel, iterations=3)
  
	#Realiza los analisis principales de la imagen
	def proceso(self, conf=False):
		"""show_countours_trans_perspective: False -> (True) Muestra todos los contornos q se hayan en la imagen original
  		conf: False -> (True) Tiene confianza en la confiabilidad del proceso y el programa nunca acudirá a usted
		show_countours_momentos: False -> (True) Muestra todos los contornos q se hayan en la imagen alineada
  		mostrar_data: False -> (True) Muestra la información que retorna los momentos de imagen"""
  
		self.Trans_Perspectiva(conf)
		self.manipular_img()
		Dc_int, area_cnts = self.momentos_geometricos()
		self.encontrar_medidas_utiles(Dc_int, area_cnts)
		return
  
	# Transformación de perspectivas
	def Trans_Perspectiva(self, confianza=False):
		"""Convierte la imagen en una alineada a un plano 2D
  		show_found_countors: True/False"""
		img = self.img_original #Sobre esta imagen se dibujará
		img_original = img.copy() #Nos ayudará a limpiar puntos realizados con el mouse	
  
		#Copilado de coordenadas de los vertices
		vertices = []
		
		#Diccionario con las intrucciones al usuario
		intrucciones = {0: "1. Haz click en la esquina \033[1m superior derecha \033[0m de la imagen",
						1: "2. Haz click en la esquina \033[1m superior izquierda \033[0m de la imagen", 
						2: "3. Haz click en la esquina \033[1m inferior derecha \033[0m de la imagen", 
						3: "4. Haz click en la esquina \033[1m inferior izquierda \033[0m de la imagen"}
	
		#Muestra y recopila los eventos del mouse
		def puntos(event,x,y,flags,param):     	 
			if event == cv2.EVENT_LBUTTONDOWN:
				cv2.circle(img,(x,y),5,(0,255,0),2)
				vertices.append([x,y])

		#Dibuja las fronteras de la imagen de acuerdo con los vertices establecidos
		def frontera_img(vertices):
			try:
				cv2.line(img,tuple(vertices[0]),tuple(vertices[1]),(255,0,0),1)
				cv2.line(img,tuple(vertices[0]),tuple(vertices[2]),(255,0,0),1)
				cv2.line(img,tuple(vertices[2]),tuple(vertices[3]),(255,0,0),1)
				cv2.line(img,tuple(vertices[1]),tuple(vertices[3]),(255,0,0),1)
	
			except: pass
				
		#Muestra la imagen a la espera de un click
		def Mostrar_img(img, titulo, timeKey):
			"""Devuelve:
   			True: Si no hubo problemas y se puede continuar
	  		False: Si no se puede continuar el proceso y se debe repetir"""
			len_0 = len(vertices)
			def protocolo():
				#Muestra la imagen
				cv2.imshow(titulo, img) 
				
				#Esperar confirmación
				k = cv2.waitKey(timeKey) & 0xFF
				if k == ord('r'): return False
				elif k == 27: quit()
				else: return True

			if timeKey == 0: return protocolo()
			while True:		
				try: 
					if len(vertices) > len_0: break
				except: pass 
				return protocolo()

		#Eventos encontrar hoja en automatico
		def auto(img):
			"""Devuelve Falso si el proceso automatico no fue posible, de lo contrario devolverá True"""
			#Organizar los vertices como es debido (sup der - sup izq - inf der - inf izq)
			def ordenar_vertices(vertices):
				n_vertices = np.concatenate([vertices[0], vertices[1], vertices[2], vertices[3]]).tolist()
				y_order = sorted(n_vertices, key=lambda n_vertices: n_vertices[1])

				x1_order = y_order[:2]
				x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])

				x2_order = y_order[2:4]
				x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])
				
				return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]
			canny = self.img_original_silueta
			cnts = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
			cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:]

			# Iterar por contorno encontrado
			for c in cnts:
				# Prueba para identificar si se trata de un poligono de 4 lados
				epsilon = 0.01*cv2.arcLength(c,True)
				approx = cv2.approxPolyDP(c,epsilon,True)
				area = cv2.contourArea(c)
	  
				if len(approx)==4 and area > 10000: #Si se trata de un cuadrado
					cv2.drawContours(img, [approx], 0, (0,255,255),2)
					vertices = ordenar_vertices(approx)
	 
					pts1 = np.float32(vertices)
					# La imagen debe estar con la parte más ancha en el eje x y la parte más corta en el y
					if vertices[1][0] - vertices[0][0] > vertices[2][1] - vertices[0][1]: self.dim.reverse()
					pts2 = np.float32([[0,0],[self.dim[0],0],[0,self.dim[1]],[self.dim[0],self.dim[1]]])
	 
					#Transformación de perspectiva
					M = cv2.getPerspectiveTransform(pts1,pts2)
	 
					#Aplica la matriz de transformación a la imagen
					img_alineada = cv2.warpPerspective(img_original, M, (self.dim[0],self.dim[1]))
	 
					#Comprobar imagen
					if confianza == False:
						#Esperar confirmación
						system("cls")
						print("Proceso automatico: \n")
						print("Presione 'R' para hacer el proceso manualmente")
						print("Presione cualquier otra letra para continuar")
	 
						cv2.imshow("imagen original", img) 
						if Mostrar_img(img_alineada, "Imagen alineada", 0) == False: return False
						else: 
							self.img_alineada = img_alineada
							cv2.destroyAllWindows()
							system("cls")
							return True
					else: 
						self.img_alineada = img_alineada
						return True
	
		try: 
			if auto(img)  == True: return
			else: 
				if confianza == True: return
				else: img = img_original.copy()
		except: 
			if confianza == True: return
			else: img = img_original.copy()

		#Eventos encontrar hoja manual
		#Pedir al usuario que interactue con la imagen
		cv2.destroyAllWindows()
		cv2.namedWindow('img')
		cv2.setMouseCallback('img',puntos)

		#Proceso del programa
		while True:
			#Instrucciones
			system("cls")
			print("Proceso manual")	
			print("Hubo un problema en la identifiación de la imagen. \nPara la proxima trata de tomar una foto nitida de unicamente la hoja A4 \n")
			print("Presione 'R' para repetir el proceso")
			print(f"{intrucciones[len(vertices)]}")

			#Muestra la imagen
			if Mostrar_img(img, "img", 1) == False: vertices, img = [], img_original.copy()

			#Actualiza la frontera de la imagen
			frontera_img(vertices)
  
			if len(vertices) == 4:
				#Puntos originales vs requeridos
				pts1 = np.float32([vertices])
				# La imagen debe estar con la parte más ancha en el eje x y la parte más corta en el y
				if vertices[1][0] - vertices[0][0] > vertices[2][1] - vertices[0][1]: self.dim.reverse()
				pts2 = np.float32([[0,0], [self.dim[0],0], [0,self.dim[1]], [self.dim[0],self.dim[1]]])

				#Matriz de transformación
				M = cv2.getPerspectiveTransform(pts1,pts2)

				#Aplica la matriz de transformación a la imagen
				img_alineada = cv2.warpPerspective(img_original, M, (self.dim[0],self.dim[1]))
				
				#Esperar confirmación
				system("cls")
				print("Proceso manual: \n")
				print("Presione 'R' para repetir el proceso")
				print("Presione cualquier otra letra para continuar")
				if Mostrar_img(img_alineada, "Imagen alineada", 0)  == False: 
					cv2.destroyAllWindows()
					vertices, img = [], img_original.copy()
				else:
					self.img_alineada = img_alineada
					cv2.destroyAllWindows()
					system("cls")
					break	
 
	# Manupular las imagenes alineadas para identificar contornos
	def manipular_img(self):
		"""Convierte la imagen alineada en un complejo blanco y negro"""
		img = self.img_alineada.copy()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.GaussianBlur(img, (5,5), 0, 0)
		self.img_alineada_filtro = img

		#th = cv2.Canny(img,100,150)
		th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		kermel = np.ones((3,3),np.uint8)
		th_plus = cv2.morphologyEx(th, cv2.MORPH_OPEN, kermel, iterations=2)
		self.img_alineada_silueta = th_plus

		if th_plus[0][0] == 255: self.img_alineada_silueta_inv = cv2.bitwise_not(th_plus)
		
	# Encuentra las caracteristicas útiles de la imagen para analizarlas en la red neuronal
	def momentos_geometricos(self):
		"""Obtiene el área, centro de gravedad y area del objeto
		show_found_countors: True/False"""
		img = self.img_alineada.copy()
		try: gray = self.img_alineada_silueta_inv
		except: gray = self.img_alineada_silueta
  
		cnts, jerarquia = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		#cnts = sorted(cnts, key=cv2.countourArea, reverse=True)[:2]
  
		area_cnts = []
		Dc_int = []
		posible_circles = []
  
		#Identificar todos los contornos
		for i in range(len(cnts)):
			jquia_ctns = jerarquia[0][i]

			# Contorno exterior: No padres si hijo
			if jquia_ctns[3] == -1 and jquia_ctns[2] != -1: cnts_ext = cnts[i]

			# Circulos internos: No hijo si padres
			elif jquia_ctns[2] == -1 and jquia_ctns[3] != -1: posible_circles.append(cnts[i])				

			else: continue   
			
		#Si hay más de 2 circulos encontrados
		if len(posible_circles) != 2: 
			choose_circle = {}
			#Encuentra los posibles mejores circulos
			for c in posible_circles:
				epsilon = 0.0001*cv2.arcLength(c,True)
				approx = cv2.approxPolyDP(c,epsilon,True)
				choose_circle[len(approx)] = c 
			
			#Organiza los circulos por la cantidad de vertices encontrados 
			# (Entre más vertices, más posible es que se trate de uno de los circulos interiores)
			choose_circle = sorted(choose_circle.items(), reverse=True)[:2]

			#Agrega los mejores circulos
			posible_circles = []
			for circle in choose_circle: posible_circles.append(circle[1])
    
		posible_circles.append(cnts_ext)	
  		#Llena de información el programa
		for c in posible_circles:
			M = cv2.moments(c)
			cX = int(M["m10"]/M["m00"])
			cY = int(M["m01"]/M["m00"])

			Dc_int.append([cX, cY]) #Lugar del centroide de la figura (contorno)
			area_cnts.append(M["m00"]) #area de la figura (contorno)
			
			cv2.drawContours(img, [c], 0, (0,255,0), 2) 
			cv2.circle(img, (cX, cY), 5, (255,0,0), -1)

		self.img_alineada_cnts = img
		return Dc_int, area_cnts

	#Encuentra distancia entre centros, radio interior y radio exterior
	def encontrar_medidas_utiles(self, Dc_int, area_cnts):
		#Variables de entrenamiento
		area_int_pix = (area_cnts[0] + area_cnts[1])/2
  
		area_int = ReglaDeTres(self, area_int_pix, potencia=2)
		area_ext = ReglaDeTres(self, area_cnts[-1] - 2*area_int_pix, potencia=2)
		Dc = ReglaDeTres(self, sqrt(abs(Dc_int[0][0] - Dc_int[1][0])**2 + abs(Dc_int[0][1] - Dc_int[1][1])**2))

		#Analisis de modelo
		historial = np.array([Dc, area_int, area_ext], dtype=float)
		resultado = self.modelo_medidas.predict(np.array([historial]))

		Do, Ancho = resultado[0][0], resultado[0][1]
		self.MedidasUtiles = {"Dc": Dc, "Do": Do, "Ancho": Ancho}

	
