# Modelo de regresión Lineal aplicado al dataset de Iris de Sci-kit Learn
El siguiente proyecto presenta el desarrollo de un modelo de Regresión lineal para el dataset Iris de scikit-learn (sklearn). Iris es un dataset de juguete empleado en el Machine Learning gracias a su simplicidad, ya que es pequeño, limpio y fácil de entender. El dataset de Iris contiene un conjunto de datos con 150 muestras de flores Iris, las cuales están divididas de forma equitativa en tres especies diferentes, hay un total de 50 muestras por cada una de las especies. 

## Características del Proyecto

La variable objetivo en este caso son las siguientes especies de flor Iris: 

<div align="center">
  
| Especies         | Imagen de Referencia | 
|-----------------|-----------------|
| **Iris Setosa** | <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg" width="25%"> |
| **Iris Virginica** | <img src="https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg" width="25%"> |
| **Iris Versicolor** | <img width="25%" alt="image" src="https://github.com/user-attachments/assets/1b0dce5a-0807-402a-befb-87e5a2416d59" />|

</div>

Para cada una de estas especies, se comprenden las siguientes features: 

- Largo del sépalo (cm)
- Ancho del sépalo (cm)
- Largo del pétalo (cm)
- Ancho del pétalo (cm)

**Diferencias en las Medidas**  
La mejor manera de ver las diferencias es comparando las distribuciones de cada medida (largo y ancho de sépalo y pétalo) para cada una de las tres especies.

1. Iris Setosa: Generalmente tiene los pétalos más pequeños y los sépalos más anchos pero más cortos en comparación con las otras dos.  

2. Iris Versicolor: Sus medidas suelen estar en un rango intermedio entre Setosa y Virginica.

3. Iris Virginica: Tiende a tener los pétalos y sépalos más grandes (tanto en largo como en ancho).

## Librerías utilizadas

| Librería / Módulo | Descripción |
|-------------------|-------------|
| `sklearn.datasets` (`load_iris`) | Contiene datasets de ejemplo, como **Iris**, muy usado para pruebas de clasificación y regresión. |
| `pandas` | Librería para la **manipulación y análisis de datos** mediante estructuras como DataFrames y Series. |
| `sklearn.model_selection` (`train_test_split`) | Herramienta para **dividir datasets** en subconjuntos de entrenamiento y prueba. |
| `sklearn.preprocessing` (`StandardScaler`, `MinMaxScaler`) | Funciones para **escalar/normalizar datos** y mejorar el rendimiento de los modelos. |
| `sklearn.linear_model` (`LinearRegression`) | Implementación de modelos de **regresión lineal**. |
| `sklearn.metrics` (`mean_absolute_error`, `mean_squared_error`, `r2_score`, `f1_score`) | Métricas para **evaluar el rendimiento** de los modelos. |
| `seaborn` | Librería de **visualización estadística** basada en matplotlib, ideal para gráficos avanzados. |
| `matplotlib.pyplot` | Librería para **crear gráficos y visualizaciones** en 2D de forma flexible. |

## Estructura del proyecto

* **`Linear_Model_Iris_DSMatallana_LELatorre.ipynb`**: Notebook de Google Colab con todo el código fuente, desde la carga de datos hasta la evaluación final.
* * **`Iris_ML`**: Archivo que contiene el script del proyecto en formato .py.
* **`requirements.txt`**: Archivo que lista todas las dependencias de Python para una fácil instalación del entorno.
* **`README.md`**: Documentación del proyecto.
---

## Instalación y uso

Para replicar este proyecto en su entorno local, siga estos pasos:  

1.  **Clone el repositorio:**
    ```bash
    git clone https://github.com/Lenna888/Linear_Model_Iris.git
    cd Linear_Model_Iris
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
    Abre el archivo `Linear_Model_Iris_DSMatallana_LELatorre.ipynb` en Jupyter Notebook, JupyterLab o Google Colab para ver y ejecutar el análisis.  

# Flujo de Trabajo del Modelo de Regresión Lineal

Los siguientes items hacen referencia a los pasos que se realizaron para la aplicación del modelo de regresión lineal antes de transformar las predicciones en clasificaciones.

1. **Cargar el dataset de Iris**  
   Importar el dataset de Iris en el entorno para que esté disponible para el preprocesamiento 
   y el entrenamiento del modelo.

2. **Crear las características (features) y la variable objetivo**  
   Definir las variables independientes (features) y la variable dependiente (objetivo) 
   que se usarán para entrenar el modelo de regresión.

3. **Dividir el conjunto de datos**  
   Separar el dataset en dos subconjuntos: entrenamiento (80%) y prueba (20%).  
   Se establece una semilla aleatoria para garantizar la reproducibilidad.

4. **Normalizar el conjunto de datos**  
   Aplicar escalado de características a los conjuntos de entrenamiento y prueba usando uno de los siguientes escaladores:  
   - **StandardScaler**: estandariza las características con media 0 y desviación estándar 1.  
   - **MinMaxScaler**: transforma las características escalándolas en un rango entre 0 y 1.  

5. **Entrenar el modelo con los datos normalizados**  
   Ajustar un modelo de regresión lineal utilizando el conjunto de entrenamiento normalizado elegido.

6. **Realizar predicciones**  
   Usar el modelo de regresión entrenado para generar predicciones sobre el conjunto de prueba normalizado.

7. **Visualizar los resultados**  
   Graficar un diagrama de dispersión y un gráfico de residuos para representar la relación entre valores reales y predichos, así como para evaluar la precisión del modelo.

8. **Evaluar las métricas de predicción**  
   Calcular las métricas de rendimiento del modelo, incluyendo:  
   - Error Cuadrático Medio (MSE)  
   - Error Absoluto Medio (MAE)  
   - Coeficiente de Determinación (R²)  

-----------------------------------------------------------------

## Clasificación de Especies

Debido a que el objetivo del proyecto es poder determinar qué planta será según los features, se debe redondear sus resultados que son números continuos, a valores de 0 para "setosa", 1 para "versicolor", 2 para "virginica". Debe devolver la precisión de la predicción realizada.  

En este caso se usó la función .rint() para redondear los datos del conjunto de datos al entero más cercano, con la particularidad de que estos se mantienen como un número de tipo float().  

En el caso de .clip(), se usó esta función para limitar los valores de este conjunto de datos (en un array) dentro de un rango específico, se estableció un valor mínimo de 0 y valor máximo de 2, al realizar esta delimitación se asegura que las predicciones no se salgan del rango de clases establecido (0=setosa , 1=versicolor y 2=virginica).  

## K-fold-cross-validation

Esta técnica se usa para poder estudiar la precisión real del modelo, pues la respuesta de una regresión lineal es un número continuo, y debe ser una clase para determinar cual planta es.

Al realizarse un entrenamiento normal, la precisión puede llegar a ser exacta, pero no demuestra la capacidad real del modelo, sino la capacidad bajo los valores determinados por el entrenamiento. En este caso, esta técnica realiza múltiples repartos y pruebas para finalmente promediar los resultados. Esta técnica funciona de la siguiente manera: 

1. Se divide el dataset en K grupos o partes (pliegues) iguales.
2. Se repite K veces: entrena el modelo con K-1 pliegues y lo prueba con el pliegue restante.
3. Promedia las K puntuaciones de rendimiento obtenidas para obtener una evaluación final más fiable.

De esta manera se reduce el riesgo de que la variabilidad del muestreo en una única división de datos, dé una idea equivocada del verdadero rendimiento del modelo.

## Análisis de Resultados

La regresión lineal es un algoritmo utilizado para realizar predicción de valores, por lo que sus resultados son números continuos, esto quiere decir que puede tomar cualquier valor positivo o negativo. En este orden, cuando se entrena la red para poder predecir un tipo de planta se debe realizar un “tratamiento” a las salidas que arroja el modelo, para ello se debió redondear y centralizar al entero más cercano entre 0 y 2.  

Al realizar este “tratamiento” a las salidas, es capaz de realizar clasificaciones,pese a que el modelo no fue pensado para clasificación, pues no es su fuerte; para el caso de estudio, se hablará de los éxitos hallados y de cómo se interpretan los datos hallados para los procedimientos con 4 diferentes semillas para el random (42, 53, 9722, 17) . Además, se realiza una calificación del modelo cuando clasifica una planta usando el dataset de Iris.  


Inicialmente, se realizó la clasificación del modelo utilizando la técnica del k-fold-cross-validation, en este se puede denotar que usando una normalización estándar o usando mínimos/máximos los resultados son exactamente iguales, demostrando que el dataset tiene valores muy cercanos o semejantes y pocos outliers, pues, si existieran más, la diferencia de clasificación al utilizar diferentes normalizaciones sería aún mayor. Esto refleja que a lo largo del procesamiento de los datos no existe pérdida.  

**Evaluación del modelo usando K-fold-cross-validation usando normalización Standard**

<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/e03d6279-f6ae-4dae-9488-95d7ef0f81da" />  

**Evaluación del modelo usando K-fold-cross-validation usando normalización MinMax**

<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/bee92f10-8b0a-4644-a3f4-b1e3633e083b" />  

Ahora bien, en los casos en los que la distribución de datos es normal, el modelo puede tener un alto porcentaje de precisión en los casos del 93%, sin embargo, para el 7% restante, la regresión lineal tiende a tener problemas para clasificar correctamente los datos que no se encuentran linealmente distribuidos en el conjunto de datos.

A lo largo de las pruebas, existe un factor común, los valores de predicción y los valores residuales son muy semejantes entre los diferentes casos, demostrando que su fuerte es la predicción de valores. Antes de aplicar la función de redondeo, la predicción del modelo de clasificación no era el adecuado. 

<img width="1590" height="670" alt="image" src="https://github.com/user-attachments/assets/f63c80c8-eef7-473e-bbaa-02038cc803cd" />  

Si bien la línea de tendencia demuestra que el modelo aprendió la relación principal de los datos, logrando clasificar de acuerdo a las características de las especies, el gráfico de residuos generado demuestra que hay un patrón de cálculo de errores predecible, para este tipo de regresiones los puntos deben estar dispersos de forma aleatoria. 

Entre las pruebas realizadas, el Test 1, que empleó la semilla 42, se destacó como el de mejor rendimiento en todas las evaluaciones realizadas. Sus resultados clave fueron un Error Absoluto Medio (MAE) de 0.1866, lo que indica una alta precisión; un Error Cuadrático Medio (MSE) de 0.0573, que demuestra consistencia; y un coeficiente de determinación (R-squared) de 0.9139, evidenciando una notable capacidad explicativa. Estos valores confirman su superior eficacia y fiabilidad. Además, cuando los datos se normalizaron, el modelo llegó a una precisión del 100%, lo cual afirma su poca capacidad para la clasificación, pues una precisión de este tipo en la práctica real no es posible, solo en espacios donde los datos son controlados se podría llegar a una efectividad de este calibre.  

<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/97b6a151-1163-4784-92db-3a1edca2dd34" />  

Por otro lado, con los datos de los test 2 y 4, donde los resultados Los resultados obtenidos al utilizar las semillas 2(53) y 4(17) en nuestros análisis revelaron métricas de rendimiento del modelo que, aunque ligeramente inferiores a las observadas en el Test 1, demuestran una solidez y una capacidad predictiva notables. Específicamente, el Error Absoluto Medio (MAE) se situó en 0.17640823160531613 y el Error Cuadrático Medio (MSE) en 0.054233682611431334. Ambos valores son indicadores de una desviación promedio y cuadrática entre las predicciones del modelo y los valores reales, respectivamente, y su magnitud reducida subraya la fiabilidad de las predicciones.  

Más allá de estos errores, el coeficiente R-cuadrado (R-squared) alcanzó un impresionante 0.918649476082853. Este valor es crucial porque indica que aproximadamente el modelo trabaja con una precisión del 91.86% y que se debe a variabilidades en los datos del entrenamiento. Una cifra tan elevada sugiere que el modelo no solo es capaz de predecir con precisión, sino que también captura la mayor parte de las relaciones en los datos.

| Métrica | Valor |
| :---- | :---- |
| Error Absoluto Medio (MAE) | 0.17640823160531613 |
| Error Cuadrático Medio (MSE) | 0.054233682611431334 |
| Coeficiente R-cuadrado | 0.918649476082853 |

Un hallazgo particularmente interesante y de gran relevancia es la observación de que, cuando la raíz de la semilla utilizada es pequeña (inferior a 1000 aprox), la precisión del modelo tiende a ser notablemente elevada, con un margen de error que se vuelve insignificante. Esta correlación entre el tamaño de la raíz de la semilla y la precisión del modelo es un descubrimiento clave. Podría implicar que ciertos rangos de valores para la inicialización aleatoria del modelo, a través de las semillas, conducen a una mayor estabilidad en el proceso de entrenamiento y, consecuentemente, a un mejor rendimiento predictivo.  

<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/7b736f1a-803a-42e3-9d6f-d9a5ea0dbdd1" />

La anterior matriz de confusión, del test 3, demuestra la adaptación del modelo de regresión lineal para clasificación de especies. Si bien logra clasificar la especie setosa, tiende a cometer errores con la clasificación de las especies versicolor y virginica. Esto nos indica que setosa es una especie fácil de diferenciar de las demás especies, las cuales tienen características similares que hacen la tarea de clasificación un poco más compleja. 

## Conclusiones

Basado en el análisis detallado de los resultados obtenidos a partir de las pruebas con el modelo de regresión lineal y su adaptación para la clasificación de especies, se pueden extraer las siguientes conclusiones fundamentales que resumen su desempeño y limitaciones.  

El estudio demuestra que la aplicación de un modelo de regresión lineal en un problema de clasificación como el conjunto de datos Iris, aunque factible, presenta una serie de limitaciones dentro del algoritmo. La regresión lineal, diseñada para predecir valores continuos, no es naturalmente apta para la clasificación de datos discretos. El "tratamiento" de las salidas continuas mediante el redondeo y recorte al entero más cercano es una técnica artificial que, si bien permite la obtención de resultados de clasificación, no refleja la verdadera capacidad del modelo para discernir entre categorías.  

El análisis de los resultados obtenidos a través de diferentes semillas aleatorias para la división de los datos (42, 53, 9722, 17) subraya la vulnerabilidad del modelo. Aunque en ciertos casos, como el Test 1 (semilla 42), se observa un rendimiento excepcional con métricas como un R-cuadrado superior a 0.91 y una precisión cercana al 100% después de la normalización, este resultado no es representativo de un desempeño robusto en la práctica. Una precisión tan elevada en un entorno controlado refleja la simplicidad del conjunto de datos y la naturaleza forzada de la clasificación, más que una capacidad de generalización del modelo. Este fenómeno sugiere que el modelo ha "memorizado" la relación lineal dentro de la partición de datos específica, lo que podría llevar a un bajo rendimiento con nuevos datos.  

La consistencia de las métricas de error (MAE y MSE) y la semejanza de los gráficos de residuos entre los diferentes tests indican que el modelo logra capturar la relación principal entre las características y las etiquetas. Sin embargo, el patrón no aleatorio en el gráfico de residuos evidencia un sesgo sistemático en los errores del modelo, la regresión lineal no logra capturar la complejidad de las relaciones no lineales entre las especies de Iris. Este hecho se manifiesta claramente en la matriz de confusión, donde el modelo muestra una alta precisión para la especie setosa (fácilmente separable linealmente) pero comete errores al clasificar las especies versicolor y virginica, que son más difíciles de diferenciar.  

Por último, el estudio confirma la importancia de la validación cruzada (K-fold-cross-validation) para evaluar la precisión del modelo, ofreciendo una medida de rendimiento más confiable al promediar múltiples particiones. La similitud de resultados con normalizaciones estándar y de mínimos/máximos en el conjunto de datos de Iris sugiere pocos valores atípicos, lo que lo hace ideal, pero no representativo de problemas reales.




































