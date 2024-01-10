# Metrología Automatiza para elementos mecánicos simples.
Este programa hace parte de mi proyecto individual desarrollado en la Universidad de los Andes. 
Consiste en identificar, caracterizar y analizar un elemento mecánico simple como un eslabón, a través de un análisis de visión por computadora y redes neuronales. Toda la información al respecto de los objetivos, funcionamiento, metodología y resultados del proyecto pueden ser vistos en el archivo InformePublico.pdf

Para ejecutarlo deberá correr el script MainInterfaz.py, y tener instaladas las librerías Tkinder, OpenCv y TensorFlow.
La estructura del programa consta de un archivo MainFunciones.py en el cual se realizan tanto los arreglos morfologicos como el análisis de la figura. 
Dicho script es luego ejecutado en MainInterfaz.py, que luego agrupa la información en una interfaz simple y amigable con el usuario que usa la carpeta ModosFalla para mostrar la información. 

La carpeta img, son imagenes propuestas para la buena ejecución del programa. No obstante, estas no son las unicas analizables, son solo un grupo de ejemplos del tipo de imagenes a las cuales se hace referencia.

Los 2 scripts restantes son los que se usaron para la creación de las redes neuronales que luego fueron guardadas con los nombres ModeloMedidasUtiles.h5 y ModeloModosFalla.h5.


Finalmente, cabe aclarar que este es un proyecto en desarrollo y se invita a quien le interese a proponer posibles mejoras o sugerencias. 
