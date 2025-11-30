source proy_dpd/bin/activate

# Problema
Crear un plataforma que impulsada por computer vision, pueda detectar y contar de forma automática los diferentes objetos (productos) en una imagen

# Aplicación
En el manejo de inventario en supermercados, el objetivo es mantener un surtido adecuado en las góndolas
Para ello se captura una foto e instantáneamente se realiza el conteo
Con ello determinamos el grado de completiutud y la falta de surtido

Como caso base, nos enfocaremo solo en el conteo, es decir no separaremos por categoría (lacteos, yogures, etc)

# Yolo
Algoritmo nacido para detección de objetos
El bounding box en la caja que marca la localización del objeto

Manejo de inventario 
Monitoreo de góndolas

# Dataset
- Base: https://universe.roboflow.com/new-workspace-l2a1a/bbox-retail

- Otros:
https://universe.roboflow.com/hardik-srivastava/shelf-auditing-for-retail-v2.0
https://universe.roboflow.com/sku110kre/image_classification-rep35
https://universe.roboflow.com/dataconversion/sku10000



# Modelo
Modelo pre entrenado para conteo de objetos en góndolas


# Antecedentes

- Es una investigación de visión por computadora que busca automatizar la monitorización de productos en supermercados para resolver problemas como la falta de existencias o el incumplimiento de los planogramas.Después de revisar algoritmos populares como R-CNN, Fast R-CNN y Faster R-CNN, el estudio concluye que YOLOv2 (You Only Look Once, Versión 2) es el más apropiado para este desafío debido a su equilibrio superior entre velocidad y precisión.
    - https://www.researchgate.net/publication/335202398_Object_Detection_in_Shelf_Images_with_YOLO

-  Neurolabs se especializa en visión por computadora sintética para el sector minorista (retail), específicamente para la auditoría de estanterías y la verificación de la ejecución de planogramas.
    - https://www.neurolabs.ai/post/what-does-the-future-of-retail-shelf-auditing-look-like-with-synthetic-computer-vision

- Un guest post que detalla la aplicación de YOLOv5 para detectar artículos en estanterías de tiendas, cubriendo el preprocesamiento de datos y el entrenamiento del modelo.
    - https://blog.roboflow.com/retail-store-item-detection-using-yolov5/


# Equipo
- **Johar :** Data Scientis & Business Specialist
- **Diego :** Data Engineer
- **Fabricio :** Data Analyst