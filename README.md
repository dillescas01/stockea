source proy_dpd/bin/activate
# 🛒 Smart Retail: Sistema Integral de Gestión de Góndolas
> **Detección de Objetos (OSA) | Predicción de Demanda | Ruteo Inteligente**

Este proyecto implementa una solución para optimizar la gestión de inventario en tiendas minoristas (caso de uso: Tiendas Tambo). El sistema automatiza la auditoría de estanterías mediante visión por computadora, predice quiebres de stock futuros y genera rutas de visita optimizadas para los gestores de campo.

---

## 👥 Equipo del Proyecto
* **Johar:** Data Scientist & Business Specialist
* **Diego:** Data Engineer
* **Fabricio:** Data Analyst

---

## 🎯 Definición del Problema y Antecedentes

### Problema
Crear una plataforma que, impulsada por *computer vision*, pueda detectar y contar de forma automática los diferentes objetos (productos) en una imagen.

### Aplicación
En el manejo de inventario en supermercados, el objetivo es mantener un surtido adecuado en las góndolas. Para ello se captura una foto e instantáneamente se realiza el conteo, determinando el grado de completitud y la falta de surtido.
* **Alcance:** Como caso base, nos enfocaremos solo en el conteo general ("clase única"), es decir, no separaremos por categoría específica (lácteos, yogures, etc.), sino que evaluaremos la disponibilidad general frente al planograma.

### Antecedentes e Investigación
El proyecto se fundamenta en investigaciones previas sobre automatización en retail:
* **Algoritmos (YOLO):** Estudios concluyen que YOLO (You Only Look Once) es el más apropiado por su equilibrio velocidad-precisión para resolver problemas de falta de existencias. [ResearchGate: Object Detection in Shelf Images](https://www.researchgate.net/publication/335202398_Object_Detection_in_Shelf_Images_with_YOLO).
* **Referentes de Industria:** Empresas como **Neurolabs** utilizan visión sintética para la auditoría de estanterías y verificación de planogramas. [Neurolabs Blog](https://www.neurolabs.ai/post/what-does-the-future-of-retail-shelf-auditing-look-like-with-synthetic-computer-vision).
* **Metodología:** Se siguieron guías de Roboflow para el preprocesamiento y entrenamiento de modelos de detección en retail. [Roboflow Blog](https://blog.roboflow.com/retail-store-item-detection-using-yolov5/).

---

## 🛠️ Tech Stack & Herramientas

| Dominio | Tecnologías / Librerías |
| :--- | :--- |
| **Lenguaje Core** | `Python 3.10+` |
| **Computer Vision** | `Ultralytics YOLO (v11/v5)`, `OpenCV`, `Pillow` |
| **Data Wrangling** | `Pandas`, `NumPy`, `OpenPyXL` |
| **Machine Learning** | `Scikit-learn` (ExtraTreesRegressor, MultiOutputRegressor) |
| **Optimización** | `Pyomo` (Optimization Modeling), `Haversine` (Geo-cálculos) |
| **Visualización & UI** | `Streamlit` (Framework Web), `PyDeck` (Mapas 3D), `Matplotlib` |
| **Control de Versiones** | `Git`, `GitHub` |

---

##  Arquitectura y Flujo del Proyecto

La arquitectura sigue una estrategia para garantizar la calidad del dato desde la captura hasta la toma de decisión.

![Diagrama de Flujo del Proyecto](image-1.png)
*Figura 1: Pipeline de datos desde la captura visual hasta la optimización logística.*

1.  **Input Visual:** Captura de imagen de la góndola.
2.  **Procesamiento:** Detección de productos (YOLO).
3.  **Almacenamiento:** Ingesta estructurada.
4.  **Predicción:** Estimación de disponibilidad futura.
5.  **Salida:** Ruteo óptimo.

---

## 📝 Documentation & Report: Process Details

A continuación se detalla el proceso técnico completo (Data Wrangling, Modeling, Prototyping) implementado en el código fuente.

### 1. Data Wrangling (Ingeniería de Datos)
* **Generación y Simulación (Capa Bronze):**
    * Se estructuró un diccionario maestro `stores_meta.py` que actúa como fuente de verdad para IDs, coordenadas y capacidades de planograma.
    * Ante la falta de históricos reales extensos, el script `generar_hist_osa_sintetica_clean.py` genera series de tiempo diarias simuladas desde enero 2024, aplicando factores de estacionalidad semanal (`DOW_MULT`) para emular el comportamiento real de compra.

* **Enriquecimiento (Capa Silver):**
    * En `forecast_utec.py`, se transforman los datos crudos mediante *Feature Engineering*:
        * **Lags:** Valores pasados (t-1, t-7, t-14).
        * **Rolling Statistics:** Medias móviles de 7 días.
        * **Encoding Temporal:** Transformación cíclica (Seno/Coseno) del día de la semana.

* **Consolidación (Capa Gold):**
    * El script `genera_data_dummy.py` unifica el histórico real con las predicciones del modelo. Se integra la segmentación estratégica (Estratos A, B, C, D) para alimentar el algoritmo de prioridad.

### 2. Modeling (Modelado y Algoritmos)
* **Visión Computacional (YOLO):**
    * Modelo entrenado para conteo agnóstico de objetos (`nc: 1`) utilizando el dataset `bbox-retail`. Se filtra por umbral de confianza para reducir falsos positivos.
    * **Métrica OSA:** $OSA \% = (\text{Productos Detectados} / \text{Capacidad Planograma}) \times 100$.

* **Forecasting (Predicción):**
    * Modelo: `ExtraTreesRegressor` con estrategia `MultiOutputRegressor` para predecir 7 días simultáneos.
    * Restricciones: Se aplica *clipping* para que la predicción no supere la capacidad física de la góndola.

* **Ruteo Inteligente (Optimización):**
    * **Función de Prioridad:** $Prioridad = 0.6(1 - OSA) + 0.3(Estrato) + 0.1(Gap)$.
    * **Algoritmo:** Híbrido. Intenta una solución exacta con `Pyomo` (MTZ formulation) y hace fallback a una heurística *Greedy + 2-opt* si no hay solver disponible.

### 3. Prototyping (Aplicación Web)
La solución se materializa en una interfaz unificada desarrollada con **Streamlit** (`app.py`), dividida en módulos funcionales.

| Módulo de Ruteo Geoespacial | Análisis de Métricas y Forecast |
| :---: | :---: |
| ![Dashboard Ruteo](image-2.png) | ![Metricas Forecast](image-3.png) |
| *Figura 2: Mapa interactivo con semáforo de prioridades.* | *Figura 3: Proyección de stock y KPIs.* |

---

## 📊 Especificaciones de Datos y Resultados Experimentales

### 4. Diccionario de Datos (Data Dictionary)

####  Capa Bronze (Ingesta)
**Archivo:** `osa_hist_Tambo_UTEC.xlsx` / `osa_resultados.xlsx`

| Columna | Tipo de Dato | Descripción | Ejemplo |
| :--- | :--- | :--- | :--- |
| `id` | String | Identificador único de la tienda. | `TUB0001` |
| `local` | String | Nombre comercial. | "Tambo UTEC" |
| `distrito` | String | Ubicación geográfica. | "Barranco" |
| `productos disponibles`| Integer | **Output YOLO:** Objetos detectados. | `22` |
| `productos esperados` | Integer | Capacidad del planograma. | `35` |
| `osa` | Float | KPI de Disponibilidad (%). | `61.11` |

####  Capa Gold (Priorización)
**Archivo:** `gold_tiendas_7d.xlsx`

| Columna | Descripción | Regla de Negocio |
| :--- | :--- | :--- |
| `estrato` | Char (A/B/C/D) | Nivel Socioeconómico (Peso: 30%). |
| `osa` | Float | Mínimo OSA predicho a 7 días (Peso: 60%). |

### 5. Especificaciones del Dataset (YOLO)
Se utilizó el dataset **`bbox-retail` (v4 tiled)** de Roboflow, optimizado para entornos de retail.
* **Volumen:** 21,492 imágenes.
* **Pre-procesamiento:** Auto-orientación, Redimensionamiento (416x416), Ecualización de contraste.
* **Augmentation:** Flip vertical (50%), Rotación (±10°), Exposición (±25%).

### 6. Análisis de Resultados (Caso de Estudio)

![Resultados Generales del Sistema](image-4.png)
*Figura 4: Panel de resultados consolidado mostrando el estado de la red de tiendas.*

#### Validación del Ruteo Inteligente
Se ejecutó el algoritmo de optimización con datos reales (`ruta_sugerida.csv`). El objetivo fue minimizar la distancia ponderada por la urgencia.

| Orden | ID Tienda | Distrito | Estrato | OSA (%) | Prioridad | Acción Logística |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `TCLV0001` | La Victoria | **C** | 68.75% | **0.368** | **Visita Inmediata** |
| **2** | `TUB0665` | Barranco | B | 70.97% | 0.316 | Ruta Eficiente |
| **3** | `TAM0001` | Miraflores | A | 78.12% | 0.225 | Baja Prioridad |
| ... | ... | ... | ... | ... | ... | ... |
| **5** | `TMEA0001` | El Agustino | **D** | **63.33%**| **0.440** | **Máxima Criticidad** |

**Interpretación:** El sistema asignó correctamente la **mayor prioridad (0.440)** a la tienda en "El Agustino" (Estrato D, OSA crítico 63%). Sin embargo, el algoritmo de ruteo la colocó al final del itinerario (posición 5) debido a su ubicación lejana, demostrando un balance inteligente entre **Urgencia vs. Eficiencia de Recorrido**.

#### Desempeño del Forecast
El modelo `ExtraTreesRegressor` demostró capacidad para capturar la tendencia semanal, utilizando los *lags* de $t-7$ para anticipar correctamente los picos de demanda cíclicos (fines de semana) característicos del negocio.

---

## 7. Conclusiones
1.  **Automatización Efectiva:** La integración de YOLO permite reducir el tiempo de auditoría de minutos a segundos, eliminando el error humano.
2.  **Gestión Proactiva:** El módulo de *Forecasting* transforma la operación de reactiva a proactiva, anticipando quiebres de stock.
3.  **Eficiencia Logística:** El algoritmo prioriza tiendas vulnerables (Estratos C/D con bajo stock) sin sacrificar la eficiencia operativa de la flota.

---



