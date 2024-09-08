<h1>Proyecto en Spark con Scala sobre el conjunto de datos Census Income</h1>


<h3>1. Resumen del conjunto de datos</h3>

<ul>
<li> Nombre del conjunto de datos: Census Income (KDD) Data Set</li>
<li> Origen y enlace al conjunto de datos: 
https://archive.ics.uci.edu/dataset/117/census+income+kdd</li>
<li> ¿Para qué se han recopilado y para qué se han usado? Se recopiló con fines de investigación 
como recurso para utilizar en la minería de datos y el aprendizaje automático de manera que 
este dataset es utilizado con frecuencia para desarrollar y evaluar algoritmos de aprendizaje 
automático y probar diferentes técnicas de minería de datos. Este dataset se ha usado para 
evaluar la capacidad de diferentes algoritmos para determinar si un individuo, con base en 
sus características demográficas y personales, tiene ingresos altos o bajos utilizando como 
umbral los 50.000 $. Además, sirve para evaluar la importancia de los diferentes atributos a 
la hora de predecir los ingresos y comprender de esta manera cuales son los factores que más 
determinan los ingresos de una persona.</li>
<li> Número de instancias: 299285 instancias. (199.523 instancias para el entrenamiento y 
99.762 instancias para test)</li>
<li> Número de atributos y su naturaleza:
<ul>
<li> Numéricos continuos: 8</li>
<li> Numéricos discretos: 5</li>
<li> Categóricos: 28</li>
</ul>
</li>
<li> Descripción de la clase: Categórica binaria. Puede tomar los valores “ - 50000.” o “50000+.” con un 93,795% 
y 6,204% respectivamente.</li>
</ul>


<h3>2. Análisis de los atributos numéricos</h3>
Para cada atributo numérico del conjunto de datos se obtiene su media, su desviación estándar, su distribución y cuantos valores nulos. Los valores nulos en este conjunto de datos se encuentran como "?". Esta sección se puede ver en el archivo Analisis-atributos.scala. Los datos obtenidos en estos análisis se guardan en carpetas para su posterior análisis.


<h3>3. Análisis de los atributos categóricos</h3>
Para cada atributo categórico del conjunto de datos se obtiene su moda, sus valores diferentes, su distribución y cuantos valores nulos. Los valores nulos se encuentran como se comenta arriba. Esta sección se puede ver en el archivo Analisis-atributos.scala. Los datos obtenidos en este análisis se guardan en carpetas para su posterior análisis.


<h3>4. Tratamiento de los valores nulos</h3>
Los valores nulos en este conjunto de datos se encuentran como "?" como se ha comentado arriba. En los atributos numéricos se reemplazan los valores nulos por la media de los valores del atributo y en los atributos categóricos se reemplazan los valores nulos por la moda de los valores del atributo. Esta sección se puede ver en los archivos de creación de los modelos.


<h3>5. Tratamiento de outliers</h3>
Se consideran outliers los valores que son menores que el percentil 5 o mayores que el percentil 95 para cada atributo numérico. Estos outliers se reemplazan por el percentil 5 o percentil 95. Esta sección se puede ver en los archivos de creación de los modelos.


<h3>6. Correlación entre atributos numéricos</h3>
Para la correlación entre los atributos numéricos, se va recorriendo cada atributo y se va calculando su coeficiente de correlación con el resto de atributos numéricos. Esta sección se puede ver en el archivo Correlación-atributos.scala.


<h3>7. Correlación entre atributos categóricos</h3>
Para la correlación entre los atributos categóricos, se calcula la matriz de contingencia entre cada atributo y el resto de atributos categóricos. Esta tabla almacena en cada celda el número de registros que tienen un determinado valor en un atributo y otro determinado valor en otro atributo. Se escoge el mayor valor de esta tabla y se guarda en un fichero para su posterior análisis. Esta sección se puede ver en el archivo Correlación-atributos.scala.

<h3>8. Selección de atributos</h3>
Tras el análisis de la correlación entre los atributos, el tratamiento de outliers y valores ausentes, a continuación se definen que atributos se seleccionan para la creación de los modelos finales:
<ul>
<li>class_of_worker</li>
<li>education</li>
<li>marital_status</li>
<li>major_industry_code</li>
<li>major_occupation_code</li>
<li>member_of_labor_union</li>
<li>race</li>
<li>sex</li>
<li>full_or_part_time_employment_status</li>
<li>hispanic_Origin</li>
<li>tax_filer_status</li>
<li>region_of_previous_residence</li>
<li>detailed_household_and_family_status</li>
<li>detailed_household_summary_in_house_instance_weight</li>
<li>live_in_this_house_one_year_ago</li>
<li>family_members_under_18</li>
<li>citizenship</li>
<li>age</li>
<li>wage_per_hour</li>
<li>capital_gains</li>
<li>capital_losses</li>
<li>dividends_from_stocks</li>
<li>total_person_earnings</li>
<li>num_persons_worked_for_employer</li>
<li>own_business_or_self_employed</li>
<li>weeks_worked_in_year</li>


</ul>

<h3>9. Preparación del conjunto de datos para el entrenamiento y test</h3>
Antes del entrenamiento del modelo, se convierte el conjunto de entrenamiento en un formato que los clasificadores lo puedan procesar.

<h3>10. Proceso de entrenamiento de los modelos</h3>
Se utilizan dos clasificadores avanzados, RandomForest y Gradient-boosted trees. Para la selección de los parámetros óptimos de estos clasificadores se utiliza ParamGrid junto a una validación cruzada. Una vez encontrados los parámetros óptimos para cada clasificador, se procede a crear el modelo final con esos parámetros.

<h3>11. Métricas para evaluar los modelos</h3>
Finalmente, se cargan los modelos creados con ambos clasificadores y se evalúa su precisión sobre el conjunto de test. Para evaluar esta precisión se utilizan diferentes medidas como la matriz de confusión, la curva ROC o la tasa de falsos positivos. 
