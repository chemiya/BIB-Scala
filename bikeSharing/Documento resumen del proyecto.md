<h1>Documento resumen del proyecto para predecir el número de bicicletas alquiladas</h1>

<ol>
<h2><li>Resumen del proyecto</li></h2>
<p>Este proyecto consistirá en la creación de varios modelos de regresión para predecir el número de bicicletas alquiladas por horas según el conjunto de datos de “Washington Bike Sharing Dataset” según las condiciones climáticas y otras características asociadas a las horas del día. En los modelos de regresión utilizados se va a tratar de optimizar sus parámetros y sobre el conjunto de datos original se van a realizar operaciones de limpieza.  </p>








<h2><li>Tecnologías utilizadas</li></h2>
<p>Se van a utilizar las siguientes tecnologías:</p>
<ul>
<li>Python: para la limpieza del conjunto de datos</li>
<li>Spark y Scala: para la creación y evaluación de los modelos</li>
</ul>

<h2><li>Descripción del conjunto de datos</li></h2>
Se utiliza el conjunto de datos “Washington Bike Sharing Dataset” que contiene los datos recogidos en estaciones de préstamo de bicicletas en la ciudad de Washington en el año 2011 y 2012. El objetivo es predecir la variable del número de préstamos por horas utilizando para ello el resto de las variables que definen para hora las condiciones climáticas y diferentes características que describen el día. Este conjunto de datos contiene 17.389 instancias sin valores nulos

<h3>Descripcion de los atributos</h3>
 El conjunto tiene un total de 16 atributos más la clase, estos se pueden clasificar según:
 <ul>
<li>Numéricos discretos: 4 +clase</li>
<li>Numéricos continuos: 4</li>
<li>Categóricos: 5</li>
<li>Binarios: 2</li>
<li>Fecha: 1</li>
</ul>

<h3>Exploración de los datos</h3>
En la primera sección del código se realiza una exploración de los datos analizando la distribución de los atributos categóricos y calculando diferentes medidas sobre los atributos numéricos como la media, la desviación estándar o el máximo y el mínimo valor. 


<h3>Limpieza de los datos</h3>
Respecto a la limpieza de los datos, los atributos de "instant", "dteday", "registered" y "casual" se eliminan ya que no se utilizarán para la creación del modelo ya que indican datos que no son necesarios.

Mediante el archivo "evaluacionAtributos.ipynb" se analiza si los atributos de la hora y el mes tienen alguna repercusión sobre la clase a predecir y efectivamente, si tienen repercusión, ya que como se esperaba, en las horas centrales del día y en los meses de verano, el número de alquileres de bicicletas aumenta. 

Mediante el archivo "correlacionContinuos.ipynb" se evalúa la correlación entre los atributos que miden la temperatura, la sensación térmica, la humedad y la velocidad del viento. Además se analiza la influencia que tienen estos atributos con la clase a predecir. Se ha detectado que los atributos de la temperatura y la sensación térmica están fuertemente correlacionados y por tanto el atributo de la sensación térmica se elimina ya que es redundante ya que contiene unos datos prácticamente similares a los de la temperatura. Por otro lado, el atributo de la velocidad del viento parece que no tiene influencia sobre la clase y por tanto, también se elimina. 

Mediante el archivo "correlacionDiscretos.ipynb" se evalúa la correlación entre los atributos que definen la estación, si es día festivo, el día de la semana, si es día de trabajo y la meteorología. Además se evalúa la influencia de estos atributos sobre la clase. Se ha podido observar mediante la tabla de contingencia generada con el programa en python que se encuentra en "tablaContingencia.xlsx" que no existen ninguna correlación entre estos atributos y que por otro lado no tienen una gran influencia sobre la clase, pero sin embargo, en cierta medida si influyen. Por lo tanto, todos estos atributos se mantienen para la creación del modelo.

En este conjunto de datos no se han detectad outliers ni valores faltantes, por lo tanto, no es necesario aplicar ninguna medida en este caso.

<h3>Transformación de los datos</h3>
Los atributos continuos de este conjunto de datos que son la temperatura, la sensación térmica, la humedad y el viento ya vienen normalizados originalmente, por lo tanto no se debe aplicar ninguna medida adicional.

Los atributos numéricos discretos, en este caso vienen transformados respecto a los atributos categóricos originales, de manera que por ejemplo, el atributo asociado a la estación, sus valores, ya vienen un formato numérico donde cada número representa una estación. Por lo tanto, en este caso tampoco es necesario aplicar ninguna medida adicional. 






<h3>Separación conjunto de datos y de test</h3>
Se han seleccionado el 70% de los registros para el conjunto de entrenamiento y 30% de los registros faltantes para el conjunto de test.


<h2><li>Modelos evaluados</li></h2>
A continuación, se detallan los modelos evaluados junto a los parámetros que se han tratado de optimizar:
<ul>
<li>LinearRegression: se tratarán de optimizar los parametros de regParam y elasticNetParam</li>
<li>GBTRegressor: se tratarán de optimizar los parámetros de maxDepth y maxIter</li>
<li>RandomForestRegressor: se tratarán de optimizar los parámetros de maxDepth y numTrees</li>
<li>DecissionTreeRegressor: se tratarán de optimizar los parámetros de maxDepth y maxBins</li>
</ul>

El modelo que ha obtenido mejor resultado finalmente ha sido el GBTRegressor con Mean Absolute Error de 30,91, un root-mean-square error de 49,7 y un valor de r cuadrado de 0,92


<h2><li>Resultado final: vídeo youtube y repositorio</li></h2>
Repositorio Github:


<h2><li>Conclusiones</li></h2>
He podido probar diferentes modelos de regresión y ver cómo se comporta cada uno sobre un conjunto de datos, además de tratar de optimizar los parámetros de cada uno. Por otro lado, previamente, he trabajado sobre la limpieza del conjunto de datos analizando correlaciones entre ellos mediante archivos en python.

</ol>
