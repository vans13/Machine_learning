# Aplicación del Algoritmo de Árboles de Decisión para la Clasificación de Correo SPAM

Este proyecto presenta la aplicación del algoritmo de árboles de decisión para la clasificación de correos SPAM, con el objetivo de identificar y analizar las características o features más influyentes que permiten su identificación. Se normalizó el conjunto de datos con ayuda del z-score, y finalmente se evaluó con las métricas de F1 Score, Accuracy Score y matrices de confusión. Los Árboles de Decisión son algoritmos de machine learning que funcionan como un diagrama de flujo jerárquico para tomar decisiones de clasificación.

## Características Principales del Proyecto

### Árboles de Decisión
Un árbol de decisión es un algoritmo de aprendizaje supervisado no paramétrico, que se utiliza tanto para tareas de clasificación como de regresión. Tiene una estructura jerárquica de árbol, que consta de un nodo raíz, ramas, nodos internos y nodos hoja.  

<img width="1024" height="576" alt="image" src="https://github.com/user-attachments/assets/f0d4067d-634f-4297-9f47-119bfb800f4d" />

El aprendizaje de árboles de decisión emplea una estrategia de divide y vencerás realizando una búsqueda codiciosa para identificar los puntos de división óptimos dentro de un árbol. Este proceso de división se repite de forma descendente y recursiva hasta que todos o la mayoría de los registros se hayan clasificado con etiquetas de clase específicas.

El sobreajuste en árboles de decisión está relacionado con su tamaño y estructura. Los árboles simples pueden lograr fácilmente nodos finales con datos de una única clase, es decir, completamente puros. No obstante, cuando el árbol se vuelve más complejo y profundo, resulta progresivamente más complicado preservar esa homogeneidad, lo que provoca que cada rama terminal contenga muy pocas observaciones. Esta dispersión excesiva de los datos, conocida como fragmentación, frecuentemente desencadena problemas de sobreajuste en el modelo.

Entiéndase que la homogeneidad hace referencia a la pureza de la clase, es decir, un nodo es completamente homogéneo o puro cuando todas las observaciones que contiene pertenecen a la misma clase. 

### Uso del algoritmo de Árbol de Decisión para la clasificación de correo SPAM/HAM

El algoritmo construye un árbol donde cada nodo interno representa una pregunta sobre una característica o feature del correo. Cada rama representa una respuesta posible (true/false), y cada hoja contiene la clasificación final (spam o ham). Para la selección del mejor atributo en cada nodo, el árbol de decisión tiene dos métodos usados como criterios de división: entropía (ganancia de información) y la impureza de Gini. 

Para este caso de clasificación, el algoritmo usado de `DecisionTreeClassifier` de la librería de `scikit-learn` usa la impureza de Gini por defecto como criterio de división. Esta impureza representa la medición del grado de heterogeneidad de las clases en un nodo de un árbol de decisión. Su definición precisa se basa en la probabilidad de clasificación incorrecta: es la probabilidad de que un elemento seleccionado al azar del nodo sea clasificado incorrectamente si se le asignara una etiqueta de clase de forma aleatoria, siguiendo la distribución de etiquetas en ese mismo nodo.


#### Impureza de Gini

La fórmula para calcular la Impureza de Gini es:

$$Gini = 1 - \sum_{i=1}^{c} (p_i)^2$$

#### Donde:

- **c**: Es el número total de clases o categorías (que en este caso serían 2 clases `SPAM` y `HAM`
- **p<sub>i</sub>**: Es la probabilidad de que un elemento seleccionado al azar pertenezca a la clase *i*.


Un conjunto de datos se considera puro si todas sus muestras pertenecen a una sola clase (máxima concentración de una etiqueta). Por el contrario, un conjunto de datos es impuro si las etiquetas de clase están mezcladas o distribuidas uniformemente entre las muestras (mínima concentración). Entre más pequeña es esta impureza de Gini, se dividirá mejor las características de los datos en cagtegorías distintas. 

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
El algoritmo de árboles de decisión utiliza métricas como la ganancia de información o el índice de Gini para determinar qué característica divide mejor los datos en cada nodo.

#### Recursividad
Además, este algoritmo repite el proceso en cada subconjunto hasta alcanzar un criterio de parada (pureza de clase, profundidad máxima, número mínimo de muestras)

#### Clasificación de Nuevos Correos
Para clasificar un nuevo correo, el algoritmo sigue el camino desde la raíz hasta una hoja, respondiendo las preguntas en cada nodo según las características del mensaje. La hoja final determina si es spam o ham. 


## Librerías Utilizadas

| Categoría | Librería / Módulo | Descripción | Uso en el código |
|-----------|-------------------|-------------|------------------|
| **Manipulación de Datos** | **pandas (pd)** | Librería para análisis y manipulación de datos estructurados (tablas, series, etc.). | Manejo de datasets en estructuras `DataFrame` y `Series`. |
| | **numpy (np)** | Librería para operaciones numéricas y manejo de arreglos multidimensionales. | Operaciones matemáticas y soporte para estructuras de datos. |
| **Visualización** | **matplotlib.pyplot (plt)** | Librería para visualización y creación de gráficos estáticos. | Gráficos como histogramas, curvas y visualizaciones de datos. Esta librería ayuda en el control de la presentación de los gráficos (títulos, leyendas, grids, límites de ejes y configuración de la figura) además de su renderización. |
| | **seaborn (sns)** | Librería basada en matplotlib para visualización estadística avanzada, su uso es recomendado con la librería matplotlib es sugerida en la documentación oficial de seaborn y múltiples tutoriales de Scikit-learn. | Creación de gráficos con mejor estética, como mapas de calor y distribuciones. Se usa esta librería en conjunto con matplotlib ya que aplica automáticamente temas visuales modernos, provee colores, fuentes y estilos más atractivos sin configuración adicional ya que Matplotlib requiere de más código para lograr esto mismo. No se está reemplanzando seaborn por matplotlib, simplemente se está generandop objetos de matplotlib internamente. Seaborn se encarga de cálculos estadísticos más complejos (boxplots).|
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

## Flujo de Trabajo 

La aplicación del modelo de  árbol de decisión en este caso se realizó con el siguiente flujo de trabajo, con el objetivo de realizar un análisis profundo de su implementación, la selección de las características y la evaluación de la estabilidad del modelo con una división de datos aleatoria. 

### Carga y preprocesamiento de datos.

Cabe aclarar que el dataset usado es el mismo  generado para la primera entrega de esta asignatura (features para la calcificación de SPAM/HAM), a partir de este se realizó: 

Lectura de datos: cargue del archivo dataset en formato .csv. 
Limpieza de datos y preparación: Las columnas `dominio_remitente` y `dominio_respuesta` se consolidaron en una única característica booleana (`dominio_coincide`). Esta nueva característica captura si los dominios coinciden (valor 1) o no (valor 0), eliminando las columnas originales para evitar la redundancia.
Separación de Variable: se realiza una división del conjunto de datos en una matriz de características X (features) y un vector objetivo y (target).

### Estandarización de los datos

En este apartado se tendrá en cuenta una situación respecto a la estandarización, si bien esta es necesaria para el entrenamiento efectivo de un modelo de ML, existe una particularidad con los árboles de decisión. Los modelos que aplican árboles de decisión son “inmunes”  a la escala de las características. Para corroborar esta situación se realizó una división de los datos para entrenar: un conjunto de datos sin estandarizar (originales) y un conjunto de datos estandarizados (StandardScaler). 

Al comparar sus métricas (específicamente el F1-Score), el código demuestra que la diferencia de rendimiento es insignificante. Esta justificación permite omitir el paso de optimizando así el uso de recursos computacionales sin dejar de lado el rendimiento.

### Agrupamiento de features y entrenamiento

Para el entrenamiento del modelo se tuvo en cuenta la siguiente distribución de features:


| Grupo      | Descripción        |
 |-----------------|--------------------|
 | all | Utiliza todas las características              disponibles como línea base para el rendimiento.  |
 | high_correlation | Incluye únicamente las características con la correlación más fuerte con la variable objetivo. |
 | without_noise | Excluye las características con una correlación muy baja, consideradas como "ruido". |
 | balanced | Un conjunto curado que combina características con correlación moderada y alta para un rendimiento óptimo. |

Para cada uno de estos grupos, se ejecutan dos modos de evaluación con 125 iteraciones cada uno, sumando un total de **1000** entrenamientos:

**Con semillas aleatorias:** Se realizaron 500 entrenamientos (125 por cada grupo de características). En cada una de estas iteraciones, los datos se dividen en conjuntos de entrenamiento y prueba usando una semilla (random_state) completamente aleatoria. Esto busca medir la estabilidad y la distribución del rendimiento del modelo para cada grupo de características cuando se enfrenta a diferentes subdivisiones de datos. Esto revela qué tan sensible es cada configuración a la variabilidad del muestreo.

**Con semillas fijas:** Se realizaron otros 500 entrenamientos. Sin embargo, en este modo, se genera una lista predefinida de 125 semillas. La iteración i del grupo all utiliza la misma semilla (y por lo tanto, la misma división de datos) que la iteración i de los otros tres grupos. Al eliminar la variabilidad de la división de datos entre los grupos, se puede realizar una comparación directa y justa para determinar qué conjunto de características tiene mejor rendimiento.

Finalmente se realiza una última prueba modificando los porcentajes de distribución del dataset entre entrenamiento y pruebas.

### Validación Cruzada K-fold-cross-validation

Después del entrenamiento, se implementó una evaluación final utilizando la Validación Cruzada Estratificada de K-Folds, con un valor de K=5. Esta validación actúa como una medida de referencia (benchmark) sólida y computacionalmente eficiente del rendimiento esperado del modelo. 

Esta validación permite comparar los resultados promedio del experimento iterativo (los 1000 entrenamientos) con un método de validación estándar de la industria. Si los resultados de ambos son consistentes, se refuerza la confianza en las conclusiones obtenidas.

Cabe aclarar que K-Fold Cross-Validation  no es el método principal de entrenamiento, sino una herramienta de validación complementaria y de confirmación. Su propósito es ofrecer una evaluación del rendimiento del modelo que es a la vez robusta, eficiente y menos susceptible a la aleatoriedad de una única división de datos.

A diferencia del método de división simple (hold-out) que se usa en las 1000 iteraciones, la validación cruzada utiliza cada punto de dato tanto para entrenamiento como para prueba, lo que conduce a una estimación del rendimiento más fiable.

El resultado final es un promedio de K evaluaciones, lo que lo hace mucho más estable y menos dependiente de la "suerte" de una división de datos particular.

Además, sirve como un punto de referencia sólido. Si el F1-Score promedio del modelo de 1000 ejecuciones para un grupo es de un valor cercano al 0.95, y el resultado de validación cruzada para ese mismo grupo es cercano a 0.95; es más seguro que ese es el verdadero rendimiento esperado del modelo.

### Visualización de resultados

Para cada grupo de características y cada modo, se calcularon estadísticas (media, desviación estándar, mínimo, máximo) para las métricas clave (F1-Score, Accuracy, etc.). Posterior a esto, se generaron diagramas de caja (boxplot) para comparar la distribución del rendimiento entre los grupos y gráficos de líneas (lineplot) para observar la progresión del rendimiento a lo largo de las 125 ejecuciones.

Por último se utilizó la matriz de confusión para evaluar el  rendimiento en predicciones específicas y la estructura del árbol de decisión (limitada a 5 niveles de profundidad) para interpretar su lógica interna.


## Análisis de Resultados

Ahora bien, después de ejecutado el script con las pruebas, se presenta el análisis que ofrece una evaluación del rendimiento del clasificador de Árbol de Decisión. La evaluación se fundamenta en los resultados obtenidos de los 1000 entrenamientos experimentales de entrenamiento basado en semilla y sin ella; además del entrenamiento por diferentes porcentajes de división del dataset. Este procedimiento permite explicar la factibilidad del modelo en cuanto a las modificaciones de su dataset y de su comportamiento en control base y sin este. A su vez, permite comprender el funcionamiento del modelo y el comportamiento de los Árboles de decisión en cuanto a su profundidad, reglas de decisión y complejidad cuando no se limita su aprendizaje.

Inicialmente se realizó la ejecución de dos series de 500 pruebas, cada una (modo aleatorio y reproducible) para los cuatro conjuntos de características definidos. Se utilizó las métricas de accuracy, F1-score, precision y recall. 



### Resultados del entrenamiento sin usar semilla (random):

<img width="1589" height="1145" alt="image" src="https://github.com/user-attachments/assets/0232f1f7-9bad-4d8b-9886-f3ada466c21f" />  

Para la visualización de los resultados de entrenamiento se optó por un gráfico de cajas y bigotes, ya que este permite contrastar en gran medida en qué rango se concentran los resultados de los entrenamientos además de proveer información visual para la identificación de outliers. Sin una semilla de reproducibilidad para los entrenamientos, se usará una fuente de aleatoriedad diferente, esto ayudará a determinar qué tan estable es el modelo frente al división de los datos.  Cabe resaltar que para la ejecución del entrenamiento se contemplaron 4 grupos de features, distribuidos de la siguiente manera: 

| Categoría        | Características                                                                 |
|------------------|---------------------------------------------------------------------------------|
| high_correlation | cantidad_exclamaciones, cantidad_urls, javascript_embebido, adjuntos_ejecutables, adjuntos_sospechosos, lenguaje_imperativo |
| without_noise    | cantidad_interrogaciones, cantidad_dominios_urls, dominio_coincide              |
| balanced         | cantidad_urls, lenguaje_imperativo, adjuntos_sospechosos, javascript_embebido, adjuntos_ejecutables, cantidad_exclamaciones, cantidad_destinatarios, ip_en_url, idioma_diferente_usuario |
| all              | cantidad_exclamaciones, cantidad_urls, javascript_embebido, adjuntos_ejecutables, adjuntos_sospechosos, lenguaje_imperativo, cantidad_interrogaciones, cantidad_dominios_urls, dominio_coincide, cantidad_destinatarios, ip_en_url, idioma_diferente_usuario |

Al comparar el rendimiento del modelo, se observa como este es sensible a la selección de características. El grupo de características sin ruido (without_noise) evidencia mejores y más estables resultados con muy poca variación, caso contrario como se muestra en el grupo de alta correlación (high_correlation), donde la concentración de los datos se muestra muy por debajo de los demás grupos (0.90) con una alta variación en los resultados en cada una de las 125 ejecuciones. 

Los grupos con todas las características y el grupo balanceado muestran un buen desempeño en general con distribuciones pequeñas en la mayoría de las métricas, juntos comparten una estabilidad similar, pero aún así se mantienen por debajo del desempeño del grupo sin ruido. 

<img width="1246" height="630" alt="image" src="https://github.com/user-attachments/assets/a0b03a2d-a71e-4518-98cb-4a4a5b057f81" />  

En esta gráfica se observa la evaluación de F1-Score del modelo por cada uno de los entrenamientos. Para el grupo de features sin ruido, se puede evidenciar el mejor rendimiento promedio a lo largo de los entrenamientos y menor fluctuación con métricas que se mantienen entre el 0.96 y 0.99, a diferencia del grupo de features con alta correlación, donde se obtuvo puntuaciones por debajo del 0.92 en comparación a los demás grupos, con picos y valles muy pronunciados, mostrando una alta inestabilidad signo de un rendimiento bajo por cada ejecución. 

<img width="1246" height="630" alt="image" src="https://github.com/user-attachments/assets/ef9bea80-2646-4786-ade3-bbb58cd18842" />  

Para la evaluación de la exactitud del modelo a lo largo de cada entrenamiento, también se denota como el grupo de alta correlación tiene una exactitud significativamente más baja e inestable a comparación de los demás grupos, caso contrario como se muestra en el grupo si ruido, donde su exactitud es consistente a lo largo de los entrenamientos. 

En cuanto al grupo con todas las características y el grupo balanceado, si bien se asemejan en su exactitud, se mantienen por debajo del grupo sin ruido de manera pronunciada. 

En este caso, el grupo sin ruido se consolida como el mejor y más estable grupo para el entrenamiento de este modelo de clasificación de SPAM, evidenciando en cada una de las métricas, valores consistentes que se mantienen en un rango de valores más estrecho con presencia muy baja de valores atípicos extremos. 


### Resultados del entrenamiento usando semilla

<img width="1589" height="1145" alt="image" src="https://github.com/user-attachments/assets/ed321cb0-ba49-4faa-b039-156b4c2d5177" />  

A comparación de las métricas anteriormente evaluadas sin semilla, se puede evidenciar cambios muy pequeños en cuanto a los valores presentados. Sin embargo, hay una pequeña connotación con el grupo de alta correlación, donde se puede evidenciar la presencia de más valores atípicos (outliers) a comparación de la prueba anterior con entrenamientos sin semilla. 

Esta particularidad demuestra claramente la inestabilidad y falta de robustez del grupo con alta correlación, con una alta variabilidad en sus resultados, indicando una posible redundancia en la información que se le presenta, dificultando el proceso de decisión del modelo. 

<img width="1246" height="630" alt="image" src="https://github.com/user-attachments/assets/0f59289d-677f-4a0b-9d31-9ed53237b1b0" />  

El análisis del F1-Score a lo largo de las ejecuciones de entrenamiento revela una notable convergencia en el rendimiento para tres de los cuatro grupos de características evaluados: ‘all’, ‘without_noise’ y ‘balanced’. Estos grupos demuestran una alta estabilidad, manteniendo un F1-Score consistentemente superior a 0.94, lo que sugiere una alta sensibilidad del modelo a la calidad y selección de las características.
Un fenómeno destacable ocurre entre las ejecuciones 80 y 100, donde estos tres grupos experimentan una caída simultánea en su rendimiento. En agudo contraste, el grupo ‘high_correlation’ alcanza uno de sus picos de rendimiento en este mismo intervalo, evidenciando un comportamiento inverso. Esta dinámica podría indicar que las características de baja correlación y el "ruido" (excluidas en el grupo high_correlation) juegan un papel crucial en la regularización del modelo, aportando robustez y mitigando la volatilidad ante variaciones en los datos de entrenamiento.

<img width="1246" height="630" alt="image" src="https://github.com/user-attachments/assets/9a415ad6-f092-41da-95b2-43b280c7a81c" />  

El modelo alcanza una Exactitud (Accuracy) general del 97%, lo que indica que clasificó correctamente la gran mayoría de los correos en el conjunto de prueba. Sin embargo, esta métrica por sí sola puede ser engañosa, especialmente si las clases no están balanceadas. Al analizar el F1-Score, observamos una visión más detallada: mientras la clase HAM obtiene un robusto 0.98, la clase SPAM alcanza 0.93. Este valor refleja el balance que hace el modelo entre no marcar correos legítimos como spam (Precisión) y su capacidad para detectar el spam real (Recall), ofreciendo una evaluación más completa de su rendimiento.

Es fundamental destacar que, aunque cada ejecución utiliza una semilla aleatoria (random_state) para la división de datos, esta semilla es consistente a través de los cuatro grupos de características en cada iteración. Esta metodología garantiza una comparación directa y equitativa del rendimiento de los modelos, aislando el efecto de la selección de características como única variable.

## Visualizaciones de los modelos más representativos
A continuación, se presenta un análisis comparativo basado en la matriz de confusión para ilustrar la diferencia de rendimiento. Para ello, se examinan los dos casos más extremos: la mejor ejecución del grupo con mayor desempeño (without_noise) y la mejor ejecución del grupo con el rendimiento más bajo (high_correlation). Este contraste permite visualizar no solo el rendimiento global, sino también la naturaleza de los errores cometidos por cada modelo.

### Modelo para el grupo con features con alta correlación:

<img width="674" height="553" alt="image" src="https://github.com/user-attachments/assets/62e5fbb6-e10f-4e48-aec5-2898733aa3f2" />  


Esta matriz revela el rendimiento del modelo entrenado únicamente con el subconjunto de características de alta correlación. A diferencia del modelo de alto rendimiento, los errores aquí son considerablemente más altos. El rendimiento de este modelo es notablemente inferior, y los errores nos dicen por qué:

- Alto número de Falsos Positivos (45): El modelo clasificó incorrectamente 52 correos legítimos (HAM) como si fueran SPAM. Este es un error crítico para la captura de datos fiables, que en un entorno real se puede presentar para problemas costosos. Este número es más de 3 veces superior al del modelo de mejor rendimiento, que solo tuvo 14 errores de este tipo.

- Alto número de Falsos Negativos (94): El modelo dejó pasar bastantes correos SPAM, lo que sugiere que no es capaz de identificarlos por su poco aprendizaje. Este fallo en la detección es el problema central que un filtro de spam debe resolver. La cifra es más del triple de los 29 Falsos Negativos que tuvo el modelo superior.

El elevado número de ambos tipos de error confirma por qué este modelo es el de peor desempeño. Al limitarse a las características con la correlación más obvia, el modelo pierde los matices necesarios para manejar los casos más difíciles. Como resultado, es mucho menos fiable, ya que comete un exceso de errores al clasificar tanto los correos legítimos como el spam.

### Modelo para grupo de features sin ruido

<img width="661" height="553" alt="image" src="https://github.com/user-attachments/assets/85b73fb8-6688-4568-b5ac-e96c7981ae42" />  
  

Esta matriz corresponde al modelo de mejor rendimiento. Los resultados confirman visualmente la alta precisión y F1-Score observados previamente, destacando un número de errores notablemente bajo. El rendimiento de este modelo es excelente y se considera altamente fiable por las siguientes razones:

Mínima Cantidad de Falsos Positivos (14): El modelo solo clasificó incorrectamente pocos correos legítimos (HAM) como SPAM. Este es un número excepcionalmente bajo, lo que se traduce en fallos que se pueden aceptar para un beneficio mayor.

Baja Tasa de Falsos Negativos (29): Los pocos errores que se pasaron en este filtro demuestran la buena capacidad para poder generalizar. Esto demuestra una alta capacidad de detección y una protección muy eficaz contra el correo no deseado.

Este rendimiento superior demuestra el valor de una selección de características inteligente. Al eliminar las características "ruidosas" que podían confundir al algoritmo, el modelo es capaz de aprender patrones más claros y robustos. El resultado es un clasificador mucho más preciso que comete significativamente menos errores, protegiendo al usuario tanto de perder correos importantes como de recibir spam.

### Análisis de los Árboles de Decisión
A continuación, se presenta el análisis de la estructura interna de los modelos. Para una mejor comprensión, se recomienda visualizar los archivos .png correspondientes en el repositorio. Este formato gráfico es ideal para explorar interactivamente el árbol, permitiendo hacer zoom para examinar en detalle las condiciones de cada nodo y entender el flujo lógico que sigue el modelo para llegar a una predicción.

#### Árbol de alta correlación
Este árbol es un excelente caso de estudio sobre por qué un modelo complejo no es necesariamente un modelo preciso.
El árbol alcanza una profundidad considerable de 14 niveles. Su nodo raíz (la primera decisión) se basa en la característica cantidad de urls. Sin embargo, este nodo inicial ya muestra una señal de dificultad: su Impureza Gini es de 0.465, un valor alto que indica que las clases SPAM y HAM están muy mezcladas en el punto de partida. Esto obliga al modelo a crear muchas más ramas para intentar separarlas.  

En este punto, es importante interpretar los colores de los nodos que se presentan, pues los nodos azules indican una predicción mayoritaria en ese punto es HAM; por otro lado, los nodos naranjas reflejan la predicción de HAM. La intensidad del color demuestra la pureza, pues entre más oscuro mayor pureza, y viceversa.  

El problema de este modelo, es la baja calidad de información con la que se entrenó, pues está limitado a las características que tienen alta correlación con la salida deseada, por lo que el árbol carece de las variables necesarias para crear reglas de separación claras y efectivas. Esto forzó al modelo a crear una estructura laberíntica y poco proporcionada, pues se denota una coloración mayoritaria azul en todo el sector derecho y centro; mientras que en la izquierda por exclusión y descarte se encuentran las reglas para hallar el HAM (naranja) por lo que demuestra la poca capacidad de detectar SPAM en los casos anteriormente separados.

#### Árbol de grupo sin ruido
El árbol de decisión del grupo without_noise representa un modelo de clasificación altamente eficaz. Su punto de partida o nodo raíz utiliza la característica, al igual que con el grupo anterior, cantidad_de_urls, una pregunta inicial muy poderosa para discriminar el SPAM. A diferencia de otros modelos, su estructura, aunque profunda y compleja, es sumamente eficiente, logrando crear nodos de alta pureza en los 18 niveles que posee. Esto significa que sus reglas internas separan con gran confianza y rapidez los correos legítimos de los que no lo son, lo cual se refleja visualmente en los colores intensos de sus hojas.  

La principal ventaja de este árbol es su alta precisión y robustez, resultado de utilizar un conjunto rico y variado de características informativas. Esto le permite identificar múltiples patrones de SPAM con un bajo margen de error. Sin embargo, su mayor desventaja es el riesgo de sobreajuste (overfitting). Un árbol tan detallado podría estar "memorizando" el ruido de los datos de entrenamiento en lugar de aprender reglas generales, lo que podría afectar su rendimiento ante datos completamente nuevos.  

El éxito de este modelo valida una máxima en el aprendizaje automático: la calidad de las características es fundamental. Al eliminar las variables "ruidosas", el algoritmo pudo concentrarse en las señales predictivas más fuertes y claras. Esta curación de los datos de entrada permitió al árbol construir una estructura lógica y eficiente, explicando directamente por qué este modelo logró una precisión tan alta y unos índices de error tan bajos en las pruebas.  



### Cross Validation Results


| group             | f1_mean  | f1_std   | accuracy_mean | accuracy_std |
| :---------------- | :------- | :------- | :------------ | :----------- |
| without_noise     | 0.975190 | 0.004965 | 0.9690        | 0.005727     |
| all               | 0.963446 | 0.006691 | 0.9496        | 0.008593     |
| balanced          | 0.961029 | 0.004085 | 0.9512        | 0.005636     |
| high_correlation  | 0.891775 | 0.003287 | 0.8684        | 0.005426     |



### Análisis de la variación de los entrenamientos

Al segmentar el conjunto de entrenamiento y test en diferentes porcentajes se busca que modelo es más robusto y eficiente con los datos que se le presentan en diferentes grupos. En este caso, se evaluaron los dos modelos más representativos, el modelo con un rendimiento inferior y el modelo con un rendimiento superior. 

Grupo con features con alta correlación

| Test Size | mean F1-Score | mean Z-score|
|-------------|------------|-------------|
| 15%       | 0.8934   | 0.6174   |
| 20%       | 0.8912   | 0.6041   |
| 25%       | 0.8910   | 0.6039   |
| 30%       | 0.8900   | 0.6031   |
| 35%       | 0.8895   | 0.6042   |

<img width="1014" height="653" alt="image" src="https://github.com/user-attachments/assets/6bbef01c-dc0f-4ce7-8021-7fe4ef0a3323" />  

<img width="1022" height="653" alt="image" src="https://github.com/user-attachments/assets/e5132ec4-febb-4cf6-ad69-3e4252ec6527" />  

Para este caso, el grupo con features con alta correlación demostró ser el modelo más limitado, con un f1-score significativamente inferior, por debajo de los 0.90. Se puede denotar como apenas cambia el rendimiento del modelo respecto a la reducción de los porcentajes del test. El presentarle más datos no ayudó significativamente, ya que eran redundantes y no ofrecían novedad en la información, esto demuestra una baja calidad de las características, entorpeciendo al modelo en la toma de decisiones para la clasificación.   


Grupo de features sin ruido

| Test Size | mean F1-Score | mean Z-score|
|-------------|------------|-------------|
| 15%       | 0.9775   | 0.7163   |
| 20%       | 0.9773   | 0.7166   |
| 25%       | 0.9749   | 0.7129   |
| 30%       | 0.9726   | 0.7095   |
| 35%       | 0.9713   | 0.7062   |


<img width="1022" height="653" alt="image" src="https://github.com/user-attachments/assets/8dc782c9-80b8-4af7-9037-9c250a180087" />  

<img width="1023" height="653" alt="image" src="https://github.com/user-attachments/assets/943a353a-e310-49bb-8269-4ce8aa725517" />  

Por otro lado, en el grupo de características sin ruido, demostró un rendimiento significativamente alto, manteniéndose por encima de 0.97. A medida que se aumenta o disminuye el porcentaje del tamaño del conjunto de entrenamiento y test, su variación es ligera, lo que se esperaría al ajustar el tamaño del conjunto de datos en un entrenamiento normal. Este comportamiento demuestra que el modelo es robusto, si bien depende de los datos, no depende de forma crítica de ellos, llegando a generalizar incluso con un 65% de datos de entrenamiento, aprovechando sustancialmente la información de los datos. Una vez más, se confirma que la mejor elección para el modelo fue la selección del grupo de características sin ruido.  


## Conclusiones

Para la aplicación de este modelo de árboles de decisión, la estandarización de datos puede omitirse debido a una razón en específico: los árboles de decisión funcionan a partir de la creación de reglas basadas en umbrales, más no en la magnitud o distancia entre los puntos de datos. Para un modelo de árbol de decisión, el algoritmo examina cada una de las características predictoras disponibles y, para cada una, evalúa todos los posibles puntos de corte o umbrales que podrían usarse para dividir los datos, de esta forma con estandarización o sin estandarización el algoritmo establecerá los mismo umbrales sin afectar de forma significativa el resultado. 

Por otro lado, la agrupación de features en distintos grupos permitió validar el rendimiento del modelo bajo diferentes escenarios, donde se demostró que la mejor adaptación de las features para el modelo fue el grupo sin ruido, producto del análisis de los primeros ejercicios donde las características con correlación cercana a 0 se excluyeron del grupo de features. Este enfoque no solo alcanzó los niveles más altos de rendimiento en todas las métricas evaluadas (accuracy, precisión, recall y F1-Score) sino que también produjo el modelo más estable y consistente a lo largo de múltiples ejecuciones. 

Sumado a esto, el uso de features con alta correlación resultó en un rendimiento significativamente inferior, con una alta inestabilidad del modelo tal como se demostró en las gráficas, donde se evidencia una alta variación de los resultados con presencia de valores atípicos  respecto a los demás grupos. Si bien estas features presentaban una alta correlación, sus resultados fueron indicativos de una mala calidad de los datos presentados al modelo, dificultando la toma de decisión del mismo, donde una característica con una correlación similar a la otra derivó en la redundancia de los datos. 

También, se pudo evidenciar que a partir de la aplicación de este modelo si bien la presencia de las semillas ayudan en el proceso de reproducibilidad, su ausencia permite evaluar la estabilidad y robustez de este, permitiendo observar la distribución de los datos a lo largo de los entrenamientos. Para este caso se estableció un total de 500 entrenamientos, 125 por cada grupo, permitiendo la captura de más información en cuanto al rendimiento del modelo con una representación más precisa de su estabilidad, tal como se evidenció en las gráficas de cajas y bigotes. Gracias a esta distribución en los entrenamientos, se pudo identificar el modelo con un rendimiento inferior (alta correlación) y el modelo con un rendimiento superior (sin ruido). 

El poder predictivo de un árbol de decisión se basa en su capacidad para construir reglas eficientes, un proceso guiado por el coeficiente de Gini, el cual busca reducir la impureza de los datos en cada división para crear grupos cada vez más puros. Este estudio demostró que la eficacia de este mecanismo depende directamente de la calidad de las características: el conjunto sin ruido permitió que el Gini tomara decisiones claras y efectivas, resultando en un modelo robusto y preciso. En cambio, la información redundante del conjunto con alta correlación obstaculiza este proceso, forzando decisiones débiles que llevaron a un modelo inestable y de bajo rendimiento. En consecuencia, se concluye que la calidad del modelo final es un reflejo directo de qué tan bien las características de entrada permiten al coeficiente de Gini cumplir su función de crear orden a partir de los datos.

En cuanto la validación cruzada (cross-validation), demuestra que el modelo con mayor rendimiento fue el que se trabajó bajo el grupo de características sin ruido, confirmando lo demostrado en las gráficas de las métricas evaluadas. Con un promedio del rendimiento, el grupo sin ruido se posiciona con un alto  f1-score (0.975) y un alto accuracy (0.969), demostrando que la exclusión de características con ruido fue la estrategia más efectiva para el entrenamiento del modelo. Por otro lado los grupos con todas las features y el grupo balanceado si bien son cercanos en cuanto al promedio, su desviación estándar es significativamente alta sugiriendo que el modelo es ligeramente menos fiable. 

Para la división de datos en el conjunto de entrenamiento y test, se segmentan en diferentes porcentajes con el fin de validar el comportamiento del modelo (15%, 20%, 25%, 30% y 35% para el test size). A partir de estos porcentajes se evidenció una relación directa con las métricas de evaluación, a medida que se disminuye el tamaño del conjunto de entrenamiento, las métricas de rendimiento disminuyeron ligeramente. Sin embargo, para el modelo con el grupo sin ruido se pasó de un F1-Score de 0.9775 con un 85% de datos de entrenamiento a un F1-score de 0.9713 con un 65%, indicando una caída mínima demostrando que la división del conjunto de datos con esos porcentajes fueron adecuados y óptimos, lo cual garantiza una rica presentación de datos para el aprendizaje del modelo y su validación. 
















