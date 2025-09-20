# Clasificador de Correos SPAM - HAM aplicando Algoritmo de Regresión Logística

Este proyecto presenta el desarrollo de un modelo de Regresión Logística para la clasificación de correos SPAM y HAM, con el objetivo de identificar y analizar cuáles son las características (features) más influyentes quie permiten al modelo tomar una decisión. Para la validación del modelo, se usó evaluadores como F1Score y matrices de confusión 

## Características Principales del Proyecto
* **Análisis Exploratorio de Datos (EDA):** Se realiza una investigación inicial del dataset para entender la distribución y las características de los datos.  
* **Preprocesamiento de Datos:** Una de las principales tareas de preprocesamiento que se realizó fue el ajuste de features para extraer información importante de los datos existentes de la relación entre el dominio del remitente y el de la respuesta. Para esto se creó una nueva columna llamada 'dominio_coincide', la cual toma un valor de 1 si ambos dominios son iguales y 0 si son diferentes, transformando dos features categóricas en una única feature numérica que el modelo pueda utilizar.  
* **Análisis de Correlación**: Se construyó una matriz de correlación para entender la relación entre las diferentes features  y la variable objetivo (si es SPAM o no). Esta matriz se puede ver representada en forma de mapa de calor usando la librería de Seaborn. Esto permitó identificar que features pueden ser predictores clave, identificando quien tienen la correlación más fuerte con la variable objetivo. Al tener un valor cercano a 1.0 o -1.0 indica una relación predictiva fuerte. En este mapa de calor, los colores más intensos (rojos y azules) señalan correlaciones más significativas, permitiendo una identificación más rápida y fácil.  
* **División del conjunto de datos**: Aquí se divide el conjunto de datos en tres muestras o subconjuntos, entrenamiento, valifdación y prueba. Se realizó dos etapas, en las cuales usando la función 'train_test_split' quedaron de la sigujiente manera: en la primera división se asignó el 70% de los datos al conjunto de entrenamiento (para que el modelo aprenda los patrones de clasificación). El restante de datos es decir el 30%, se asignó para la evaluación. Ese 30% pasó a una segunda división de datos para realizar el conjunto de validación y testeo, donde se dividió a la mitad, asignando 15% para testeo y 15% para evaluación.En ambas divisiones se utilizó el parámetro `stratify` para mantener una distribución proporcional de las clases SPAM y no SPAM en todos los subconjuntos, garantizando que cada uno sea una muestra representativa del dataset original.  
* **Entrenamiento del modelo:** Se construye y entrena un modelo de Regresión Logística utilizando la librería **Scikit-learn**. Para los parámetros del modelo se tuvo en cuenta lo siguiente:   
    - `random_state=42`: Se fijó un número inicial para garantizar que los resultados del entrenamiento sean consistentes y reproducibles en futuras ejecuciones.
    - `max_iter=10000`: Se estableció un número elevado de iteraciones máximas para asegurar que el algoritmo tenga suficiente tiempo para converger a la mejor solución posible.
    - `tol=0.0001`: Se definió una tolerancia para el criterio de parada, optimizando el tiempo de entrenamiento.  
* **Evaluación del Rendimiento:** El modelo se evalúa utilizando la **matriz de confusión** y el **F1-Score**, métricas clave para problemas de clasificación. La matriz de confusión muestra un desglose detallado de los aciertos y errores del clasificador, permitiendo visualizar directamente los Verdaderos Positivos (spam bien clasificado), Verdaderos Negativos (correo legítimo bien clasificado), Falsos Positivos (correo legítimo marcado como spam) y Falsos Negativos (spam que pasó a la bandeja de entrada).Un F1-Score alto como el obtenido indica que el modelo mantiene un excelente equilibrio entre minimizar los falsos positivos y los falsos negativos.  


## Librerías Utilizadas: 
Este proyecto se basa en un conjunto de librerías de Python ampliamente utilizadas en el ecosistema de ciencia de datos. A continuación, se detalla el propósito de cada una:

| Librería / Módulo | Breve Descripción |
| :--- | :--- |
| **`pandas`**  | Utilizada para la manipulación y el análisis de datos. Su estructura principal, el **DataFrame**, es fundamental para cargar y limpiar el dataset. |
| **`numpy`**  | Proporciona soporte para arrays y matrices, junto con una vasta colección de funciones matemáticas para operar sobre ellos de manera eficiente. |
| **`matplotlib.pyplot`**  | Es la librería base para la creación de visualizaciones estáticas en Python, como gráficos de líneas, barras e histogramas. |
| **`seaborn`**  | Construida sobre Matplotlib, esta librería permite crear gráficos estadísticos más complejos y visualmente atractivos con menos código. |
| **`sklearn.model_selection`** | Módulo que contiene herramientas para gestionar y dividir los datos antes de entrenar un modelo. Su función principal es asegurar que se evalúe el modelo de forma justa, probando con datos que nunca ha visto antes. 
| `train_test_split` | Función para dividir el dataset en subconjuntos aleatorios de **entrenamiento (train)** y **prueba (test)**, un paso crucial en el modelado. |
| **`sklearn.preprocessing`** | Módulo para la limpieza y transformación de los datos para que el modelo lo entienda mejor y pueda funcionar de manera más eficiente. 
| `StandardScaler` | Herramienta para **estandarizar** los datos (media 0, varianza 1), mejorando el rendimiento de algoritmos como la Regresión Logística. |
| **`sklearn.linear_model`** | Módulo que contiene los algoritmos de machine learning a entrenar, incluye modelos que se basan en relaciones lineales entre las variables. 
| `LogisticRegression` | Implementación del algoritmo de **Regresión Logística**, el modelo de clasificación seleccionado para este proyecto. |
| **`sklearn.metrics`** | Módulo que contiene las herramientas necesarias para la evaluación y calificación del rendimiento del modelo. 
| `confusion_matrix` | Calcula la **matriz de confusión**, una tabla que resume el rendimiento del modelo mostrando aciertos y errores (TP, FP, FN, TN). |
| `classification_report`| Genera un informe de texto con las métricas clave de clasificación: **precisión, recall y f1-score** para cada clase. |
| `f1_score` | Calcula el **puntaje F1**, la media armónica de la precisión y el recall, una métrica robusta para evaluar clasificadores. |
| `accuracy_score` | Calcula la **exactitud (accuracy)**, que representa la proporción de predicciones correctas sobre el total de casos. |

## Estructura del Proyecto
* **`data/`**: Carpeta que contiene el conjunto de datos (`.csv`) utilizado.
* **`Machine_learning.ipynb`**: Notebook de Google Colab con todo el código fuente, desde la carga de datos hasta la evaluación final.
* **`requirements.txt`**: Archivo que lista todas las dependencias de Python para una fácil instalación del entorno.
* **`README.md`**: Documentación del proyecto.

---


## Instalación y uso

Para replicar este proyecto en su entorno local, siga estos pasos:  

1.  **Clone el repositorio:**
    ```bash
    git clone https://github.com/Lenna888/Classifier_ML_Dsmatallana_Lelatorre_802.git
    cd Classifier_ML_Dsmatallana_Lelatorre_802
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
    Abre el archivo `Machine_learning.ipynb` en Jupyter Notebook, JupyterLab o Google Colab para ver y ejecutar el análisis.

---

## Resultados del Análisis

Los siguientes resultados cabe aclarar que se realizaron con el conjunto de datos sin estandarizar. A pesar de que no están esrtandarizados, entregan una buena relación de features con la variable objetivo.  
 ### **Matriz de Correlación**
 <img width="1197" height="997" alt="image" src="https://github.com/user-attachments/assets/d4d2fa32-e04a-4aa5-99e2-57f35db197f0" />  

En est matriz de correlación se puede observar 6 features que están relacionadas de manera moderada con la variable objetivo, que si bien no es fuerte es estadísticamente significativa. Dentro de las features con relaciones más fuertes se pudo identificar las siguientes:

 - **cantidad_exclamaciones:** 0.22
 - **cantidad_urls:** 0.39
 - **javascript_embebido:** 0.25
 - **adjuntos_ejecutables:** 0.25
 - **adjuntos_sospechosos:** 0.26
 - **lenguaje_imperativo:** 0.27

No se identificaron correlaciones negativas significativas, lo que sugiere que las características seleccionadas contribuyen principalmente a identificar la presencia de SPAM en lugar de su ausencia.  
### Resultados y rendimiento del Modelo

**Prueba con todas las features**
- Accuracy: 0.98533
- F1 Score: 0.98841
<p align="center">
<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/f0ffe426-2602-4cdb-9f64-34bcf863a69c" />
</p>
En esta prueba, se quería evaluar el rendimiento del modelo escogiendo todas las features. En la matriz de confusión se puede observar que solo 5 correos HAM se clasificaron como SPAM (falsos positivos) y solo 6 correos SPAM se clasificaron como correos HAM. Las métricas en esta primera prueba arrojan una precisión superior a comparación de la segunda prueba, donde se seleccionaron los features con alta correlación. 

**Prueba con los features seleccionados**  
- Accuracy: 0.89733
- F1-Score: 0.91817
<p align="center">
<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/64e61016-8a43-4394-a603-b57279b9ee3c" />
</p>

Esta prueba tuvo el peor rendimiento, ya que no solo se eliminó el ruido, también se eliminó parte importante de la información de estos features que seguían aportando utilidad en el modelo, a pesar se que tienen una correlación moderada. 

En esta matriz se puede observar que 34 correos HAM se clasificaron como SPAM (falsos positivos) y 43 correos SPAM se clasificaron como un correo HAM. La elección de estas features fue contraproducente en el modelo, triplicando incluso el número de errores.

**Prueba con features, excluyendo aquellos cuya relación es cercana a 0**
- Accuracy: 0.98667
- F1 Score: 0.98945
<p align="center">
<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/8c4779a7-f65f-4346-8a9c-d0a8403593ff" />
</p>

Esta prueba evidenció una mejora en cuanto a la precisión del modelo, se eliminaron las siguientes features: 
- 'cantidad_interrogaciones'
- 'cantidad_dominios_urls'
- 'dominio_coincide'

Gracias a la eliminación de estas features, se eliminó en parte el ruido generado para el entrenamiento del modleo. Con una correlación cercana a 0, no tiene ninguna relación lineal predecible com el hecho de que sea un correo SPAM o no. Esta prueba sin estos features, ayuda en la generalización del modelo con datos nuevos, reduciendo además el riesgo del sobreajuste (donde prodría encontrar patrones falsos en el ruido del conjunto de entranemiento). Incluso, disminuyó la cantidad de falsos positivos a comparación de la prueba 1. Aquí solo se tuvo 4 falsos postivos. 

**Prueba con features con relación mayor a 0.15**
- Accuracy: 0.95333
- F1 Score: 0.96327  
<p align="center">
<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/d011ec05-b7db-4b90-8067-58706a31d6b4" />
</p>

En esta prueba la idea era descartar las features más débiles o ruidosas en el conjunto de entrenamiento. Fue mejor que la prueba 2 en cuanto la obtención de u modelo más preciso. Bajo esta condición se conservó las características con una señal útil. Sin embargo, fue bajo el rendimiento frente a la prueba 1, donde no logró superarlo. 

Aquí se evidencia el cambio que se puede dar en cuanto a la utilidad de la información y el ruido que los mismos daos pueden generar. 

## Normalización en el conjunto de datos para pruebas

En esta parte del modelo, se normalizaron los datos con el objetivo de evitar la aparición de outliers, esto con el fin de ver la diferencia entre un entrenamiento con normalización y uno sin normalización. Aquí no se implementa la matriz de confusión, se optó por utilizar el accuracy y el f1 Score. A continuación, se presenta una tabla comparativa con las métricas de rendimiento obtenidas en cada uno de los experimentos realizados con datos escalados.

| Prueba | Exactitud (Accuracy) | Puntuación F1 (F1 Score) |
| :--- | :--- | :--- |
| **Prueba 1 (Todas las features)** | 1.00000 | 1.00000 |
| **Prueba 2 (Alta relación)** | 0.89733 | 0.91817 |
| **Prueba 3 (Excluyendo ruido)** | 0.99867 | 0.99895 |
| **Prueba 4 (Equilibrado)** | 0.95600 | 0.96545 |

En la normalización o escalado de datos, permite que todas las características tengan la cualidad de influir en el resultado del modelo. En la prueba 1 si bien el modelo fue capaz de lograr una precisión perfecta, demostrando que el conjunto de datos es linealmente separable. Sin embargo, cabe aclarar que puede indicar también un sobreajuste debido al conjunto de prueba específico, lo que nos advierte que el modelo podría no generalizar bien los datos nuevos que se le presenten. 

La prueba 3, sigue siendo el modelo de referencia con un rendimiento bastante aceptable, aquí se refuerza la idea de que eliminar las características con correlación casi nula, es decir esas features que generan ruido, crea un modelo más simple pero potente. A comparación de la prueba 4, donde si bien las características con relación débil aportan información útil, la prueba 4 sigue siendo inferior que los anteriores modelos. 

Finalmente, la prueba 2, sigue siendo el modelo con peor rendimiento evidenciando una clara pérdida de la información en features al escoger únicamente features con correlaciones altas. Se pierde demasiada información valiosa del resto de las variables, y el resultado es un modelo mucho menos preciso. 

## Prueba final

### Datos sin normalizar

<p align="center">
<img width="1058" height="846" alt="image" src="https://github.com/user-attachments/assets/ccaae6f9-b88d-46f2-9385-41a665a6133d" />
</p>

Se evidencia que las features seleccionadas son independientes y no justifica la necesidad de eliminar alguna, pues todas tienen gran impacto en el entrenamiento. 

**Matriz de Confusión**
<p align="center">
<img width="649" height="548" alt="image" src="https://github.com/user-attachments/assets/be5c84c0-fd53-4fce-ac47-112448ebdd2c" />
</p>

Se puede ver que la precisión que en los 750 casos de pruebas, refleja la mejor precisión sin caer en el sobreajuste permitiendo una generalización en casos nuevos.

### Datos normalizados
<p align="center">
<img width="1058" height="846" alt="image" src="https://github.com/user-attachments/assets/30537d8a-5f91-4e35-97e1-1c570baeb6d4" />
</p>

Al normalizar los datos, el algoritmo distinguió patrones sutiles que antes estaban ocultos por las disparidades en las escalas numéricas, lo que permitió que aprendiera una frontera de decisión mucho más precisa entre las clases, reduciendo los errores a un mínimo absoluto.

**Matriz de Confusión**
<p align="center">
<img width="649" height="548" alt="image" src="https://github.com/user-attachments/assets/5c0276f0-c5e9-461e-9efc-76a43861fd04" />
</p>

Para esta matriz, se visualiza una disminución en los falsos positivos y falsos negativos. El número total de errores se redujo de 14 a solo 2. 

## Conclusión
El modelo de Regresión Logística fue lo suficientemente preciso para extraer señales útiles de todas las características y asignar pesos muy bajos a las que no aportaban, sin que estas generaran "ruido".

Se realizaron múltiples pruebas para determinar el conjunto óptimo de características para el modelo. Se entrenó un modelo base utilizando todas las features disponibles, y se comparó su rendimiento con otros modelos entrenados con subconjuntos de características (seleccionadas por su alta correlación, baja correlación, etc.).  

El análisis demostró que el modelo con el mejor rendimiento fue aquel que utilizó la prueba 3, donde se eliminaron aquellas features con relaciones cercanas a 0, con el fin de eliminar el ruido generado por features con esta relación.

* **Mejor Accuracy obtenido:** **0.98667**
* **Mejor F1 Score obtenido:** **0.98945**

Cualquier intento de selección manual de características, incluso basándose en la matriz de correlación, resultó en una disminución del rendimiento. Esto sugiere que las características con correlaciones individuales más débiles aún proporcionan información valiosa y complementaria que el modelo de Regresión Logística es capaz de aprovechar para lograr una clasificación más precisa. Por lo tanto, se concluye que para este problema, el uso de todas las características es la estrategia óptima teniendo en cuenta la eliminación de features con relación cercana a 0 para este caso, donde una eliminación de ruido mejoró en un pequeño porcentaje para las métricas de evaluación de modelo. 

La normalización de los datos fue el paso más decisivo para optimizar el modelo,  permitiendo que la Regresión Logística funcionara de manera óptima. Estos resultados demuestran que el mejor enfoque para este problema es usar el conjunto completo de características con los datos debidamente escalados junto con la eliminación de ruido, lo que lleva a un modelo con un rendimiento adecuado en el conjunto de prueba.





