# Aplicación del Algoritmo de Árboles de Decisión para la Clasificación de Correo SPAM

Este proyecto presenta la aplicación del algoritmo de árboles de decisión para la clasificación de correos SPAM, con el objetivo de identificar y analizar las características o features más influyentes que permiten su identificación. Se normalizó el conjunto de datos con ayuda del z-score, y finalmente se evaluó con las métricas de F1 Score, Accuracy Score y matrices de confusión. Los Árboles de Decisión son algoritmos de machine learning que funcionan como un diagrama de flujo jerárquico para tomar decisiones de clasificación.

## Características Principales del Proyecto  

El algoritmo construye un árbol donde cada nodo interno representa una pregunta sobre una característica o feature del correo. Cada rama representa una respuesta posible (true/false), y cada hoja contiene la clasificación final (spam o ham).

### Proceso de Construcción

#### Selección de características 
El algoritmo evalúa todas las características disponibles, para el dataset se escogieron las relacionadas a continuación.

- Cantidad de signos de exclamación `(cantidad_exclamaciones)`	
- Cantidad de signos de exclamación `cantidad_interrogaciones`
- Cantidad de URLs `cantidad_urls`
- Detección de JavaScript Embebido `javascript_embebido`
- Dominio del remitente `dominio_remitente`
- Domimio de replay-to `dominio_respuesta`
- Cantidad total de dominios diferentes en URLs `cantidad_dominios_urls`
- Presencia de direcciones IP en URL `ip_en_url`
- Presencia de adjuntos ejecutables `adjuntos_ejecutables`
- Adjuntos con patrones sospechosos `adjuntos_sospechosos`
- Cantidad de destinatarios `cantidad_destinatarios`
- Idioma del correo vs del usuario `idioma_diferente_usuario`
- Uso de lenguaje imperativo `lenguaje_imperativo`
- Uso de acortadores de URL `uso_acortadores`

#### División óptima
Utiliza métricas como la ganancia de información o el índice de Gini para determinar qué característica divide mejor los datos en cada nodo  

#### Recursividad
Repite el proceso en cada subconjunto hasta alcanzar un criterio de parada (pureza de clase, profundidad máxima, número mínimo de muestras)

#### Clasificación de Nuevos Correos
Para clasificar un nuevo correo, el algoritmo sigue el camino desde la raíz hasta una hoja, respondiendo las preguntas en cada nodo según las características del mensaje. La hoja final determina si es spam o ham.


## Librerías Utilizadas

| Categoría | Librería / Módulo | Descripción | Uso en el código |
|-----------|-------------------|-------------|------------------|
| **Manipulación de Datos** | **pandas (pd)** | Librería para análisis y manipulación de datos estructurados (tablas, series, etc.). | Manejo de datasets en estructuras `DataFrame` y `Series`. |
| | **numpy (np)** | Librería para operaciones numéricas y manejo de arreglos multidimensionales. | Operaciones matemáticas y soporte para estructuras de datos. |
| **Visualización** | **matplotlib.pyplot (plt)** | Librería para visualización y creación de gráficos estáticos. | Gráficos como histogramas, curvas y visualizaciones de datos. |
| | **seaborn (sns)** | Librería basada en matplotlib para visualización estadística avanzada. | Creación de gráficos con mejor estética, como mapas de calor y distribuciones. |
| **Modelado y Validación** | **sklearn.model_selection.train_test_split** | Función para dividir el dataset en conjuntos de entrenamiento y prueba. | Separación de datos en `train` y `test`. |
| | **sklearn.model_selection.StratifiedKFold** | Método de validación cruzada que mantiene la proporción de clases. | Partición de datos balanceada en entrenamiento y validación. |
| | **sklearn.model_selection.cross_val_score** | Función para evaluar modelos mediante validación cruzada. | Evaluación del desempeño del modelo en distintas particiones. |
| | **sklearn.tree.DecisionTreeClassifier** | Algoritmo de clasificación basado en árboles de decisión. | Creación y entrenamiento del modelo de clasificación. |
| | **sklearn.tree.plot_tree** | Función para visualizar árboles de decisión entrenados. | Visualización gráfica del árbol entrenado. |
| **Métricas de Evaluación** | **sklearn.metrics.accuracy_score** | Métrica que mide la proporción de predicciones correctas. | Evaluación de precisión del modelo. |
| | **sklearn.metrics.f1_score** | Métrica que combina precisión y recall en un solo valor. | Evaluación balanceada del modelo. |
| | **sklearn.metrics.precision_score** | Métrica que mide la proporción de verdaderos positivos entre las predicciones positivas. | Evaluación de precisión de las predicciones positivas. |
| | **sklearn.metrics.recall_score** | Métrica que mide la proporción de verdaderos positivos detectados. | Evaluación de la sensibilidad del modelo. |
| | **sklearn.metrics.confusion_matrix** | Matriz que muestra la relación entre predicciones y valores reales. | Análisis de errores de clasificación. |
| | **sklearn.metrics.classification_report** | Reporte con métricas principales (precision, recall, f1, support). | Resumen detallado del desempeño del modelo. |
| | **sklearn.metrics.roc_auc_score** | Métrica basada en el área bajo la curva ROC. | Evaluación del rendimiento del modelo en clasificación binaria. |
| **Preprocesamiento** | **sklearn.preprocessing.StandardScaler** | Escalador para normalizar características con media 0 y varianza 1. | Normalización de datos antes de entrenar modelos. |
| | **sklearn.pipeline.make_pipeline** | Función para encadenar pasos de preprocesamiento y modelado. | Creación de pipelines con escalado y modelo de clasificación. |
| **Aleatoriedad y Simulación** | **random** | Módulo estándar de Python para generar números pseudoaleatorios. | Selección aleatoria, generación de números aleatorios y control de experimentos reproducibles. |


## Estructura del Proyecto
* **`data/`**: Carpeta que contiene el conjunto de datos (`.csv`) utilizado.
* **`Arbol_Decision_SPAM-HAM.ipynb`**: Notebook de Google Colab con todo el código fuente, desde la carga de datos hasta la evaluación final.
* **`requirements.txt`**: Archivo que lista todas las dependencias de Python para una fácil instalación del entorno.
* **`README.md`**: Documentación del proyecto.

---

## Instalación y uso

Para replicar este proyecto en su entorno local, siga estos pasos:  

1.  **Clone el repositorio:**
    ```bash
    git clone https://github.com/vans13/Machine_learning/tree/42da87acb42872a93a37eeba48a231b610e9ae80/Tree_decision_HAM-SPAM
    cd Tree_decision_HAM-SPAM
    ```

2.  **Cree un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instale las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **O ejecute el Notebook:**
    Abre el archivo `Arbol_Decision_SPAM-HAM.ipynb` en Jupyter Notebook, JupyterLab o Google Colab para ver y ejecutar el análisis.

---












