/*
CDs_and_Vinyl
Es un conjunto de datos que proviene de Amazon y contiene información sobre productos de la categoría de CD y vinilos, así como las reseñas de los clientes sobre estos productos. Sus atributos son:
-item: el identificador único para el producto.
-user: el identificador único para el usuario.
-rating: la calificación que el usuario dio al producto, en una escala de 1 a 5 estrellas.
-timestamp: la fecha y hora de la reseña en formato Unix.

*/





// Arrancar spark-shell como: $ spark-shell --driver-memory 8g --executor-memory 8g --executor-cores 4

// Configuraciones iniciales
// Establece el directorio de checkpoint para la aplicación de Spark 
sc.setCheckpointDir("checkpoint")
// Ajusta el nivel de logging para el contexto de Spark
sc.setLogLevel("ERROR")


import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.util.Calendar
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SparkSession

// Variables para el acceso a los datos
val PATH = "/home/usuario/bib-scala/"
val ARCHIVO = "CDs_and_Vinyl.csv"
val ARCHIVO_TEST = "conjunto-test"
val ARCHIVO_MODELO = "modeloALS"





















// Carga de los datos-------------------------------------------------------------------------------

// Definición del esquema de datos
// Formato del archivo de datos: item,user,rating,timestamp separados por coma
case class Rating( item: String, user: String, rating: Float, timestamp: Int )


// Función para procesar los datos leídos
def parseRating(str: String): Rating = {
    // Separar por comas
    val fields = str.split(",")
    //Validar el número de elementos leídos
    assert(fields.size == 4)

    // Crear un elemento
    Rating(
        fields(0).toString,
        fields(1).toString,
        fields(2).toFloat,
        fields(3).toInt
    )
}

// Cargar los datos del csv aplicando la función que los procesa y convierte
val ratings = spark.read.textFile(PATH + ARCHIVO).map(parseRating).toDF()

// Eliminar nulos
var dfCDsVinyl = ratings.na.drop()






















// Exploración--------------------------------------------------------------------------------------
println("Número total de registros: " + dfCDsVinyl.count())

println("Primeros 5 registros:")
dfCDsVinyl.show(5)

println("Estructura del DataFrame:")
dfCDsVinyl.printSchema()

println("Resumen estadístico de Rating:")
dfCDsVinyl.describe("rating").show()

// Con cada valor de rating, cuantos hay
println("Número de votos por valor:")
dfCDsVinyl.groupBy("rating").count().orderBy(desc("count")).withColumnRenamed("count", "cuenta").show()

println("Rango de timestamp:")
val MinMaxTime = dfCDsVinyl.agg(min("timestamp"), max("timestamp")).head()

// Convertir el timestamp a formato de fecha
val dateFormat = new SimpleDateFormat("dd-MM-yyyy hh:mm")

// Multiplicamos por 1000L porque Timestamp espera milisegundos
val minFechaStr = dateFormat.format( new Timestamp(MinMaxTime.getInt(0) * 1000L) )
val maxFechaStr = dateFormat.format( new Timestamp(MinMaxTime.getInt(1) * 1000L) )

println("Mínimo valor de timestamp: " + MinMaxTime(0) + " -> " +  minFechaStr)
println("Máximo valor de timestamp: " + MinMaxTime(1) + " -> " +  maxFechaStr)

// Contar usuarios y productos diferentes
println("Número de productos (Item): " + dfCDsVinyl.select("item").distinct.count())
println("Número de usuarios (User): " + dfCDsVinyl.select("user").distinct.count())






























// Transformaciones-------------------------------------------------------------------------------------

// Mediante StringIndexer se asigna a user e item un índice entero
val columnasIndexer = Seq("item", "user")

val indexadores = columnasIndexer.map {
    columna =>new StringIndexer().setInputCol(columna).setOutputCol(s"${columna}Id")
}

indexadores.foreach {
    indexador =>dfCDsVinyl = indexador.fit(dfCDsVinyl).transform(dfCDsVinyl)
}

// Convertir a Int los ids
var dfCDsVinylTransformado = dfCDsVinyl.
    withColumn("itemId", col("itemId").cast("Int")).
    withColumn("userId", col("userId").cast("Int")).
    select("itemId", "item", "userId", "user", "rating", "timestamp")

println("Conjunto de datos transformado:")
dfCDsVinylTransformado.show(5)

// Guardar cada uno de los usuarios e items diferentes
dfCDsVinylTransformado.select("user", "userId").distinct().write.mode("overwrite").csv(PATH + "dfUserLookup")

dfCDsVinylTransformado.select("item", "itemId").distinct().write.mode("overwrite").csv(PATH + "dfItemLookup")



















// Creación de conjuntos training y test-------------------------------------------------------------

// Factor de escala para estratificar el conjunto de entrenamiento
val SCALE_FACTOR = .8

// Convertir a String la columna, aplicar factor de escala y eliminar columna
var training = dfCDsVinylTransformado.
    withColumn("ratingId", col("rating").cast("String")).
    stat.sampleBy("ratingId",
        fractions = Map(
            "1.0" -> SCALE_FACTOR,
            "2.0" -> SCALE_FACTOR,
            "3.0" -> SCALE_FACTOR,
            "4.0" -> SCALE_FACTOR,
            "5.0" -> SCALE_FACTOR
), seed = 10).drop("ratingId")

// Generar el dataset de test eliminando los elementos de training
val test = dfCDsVinylTransformado.except(training)
println("Registros test: " + test.count())

// Guardar conjunto de test
test.write.mode("overwrite").csv(PATH + "conjunto-test")
println("Conjunto de test guardado")

println("Registros training antes de limpieza: " + training.count())



// Buscar usuarios diferentes, limitamos a 100 para hacerlo más breve
val valoresUnicosUsuario = training.select("userId").distinct().limit(100).collect()
val usuariosDiferentes=valoresUnicosUsuario.length
var contador=0

// Recorrer los usuarios
for (fila <- valoresUnicosUsuario) {
     println(s"Evaluando datos de: $contador / $usuariosDiferentes")
     contador=contador+1
     // Extraer el valor
     val valorUnicoInt = fila.getInt(0)

     // Filtrar opiniones de cada usuario
     val dfFiltrado = training.filter(col("userId") === valorUnicoInt)

     // Cuántas opiniones ha hecho
     val filas=dfFiltrado.count()

     // Desviación estándar de los timestamp de sus opiniones
     val desviacionEstandar = dfFiltrado.
         agg(stddev("timestamp").
         alias("desviacion_estandar")).
         collect()(0).
         getAs[Double]("desviacion_estandar")

    // Condicional para eliminar datos si cumple condiciones. 
    // Si tiene muchas opiones como poca desviación en el timestamp, puede que sean falsas
     if(desviacionEstandar < 31622400 && filas > 50) {
         // Filtrar filas que no sean de ese usuario
         training = training.filter(col("userId") =!= valorUnicoInt)
         val numeroFilas = training.count()
         println(s"Se han eliminado los registros del usuario por no cumplir condición 1. El número de filas en el DataFrame es: $numeroFilas")
     }

     // Cuántas opiniones con cada rating han hecho
     var cuentaOpiniones = dfFiltrado.
         groupBy("rating").
         count().
         orderBy(desc("count")).
         withColumnRenamed("count", "cuenta")

     // Convertir en porcentaje sobre el total de sus opiniones
     cuentaOpiniones = cuentaOpiniones.
         select( format_number((col("cuenta") / filas) * 100, 2).alias("Resultado") )

     // Porcentaje opiniones rating 5
     val primerValorPorcentaje = cuentaOpiniones.
         select("Resultado").
         first().
         getAs[String](0)

     val valorDouble = primerValorPorcentaje.toDouble

     // Condicional para eliminar datos si cumple condiciones
     // Si mas del 90% de sus opiniones son de rating 5 y ha hecho más de 50 opiniones
     if(valorDouble > 90 && filas > 50) {
        // Filtrar filas que no sean de ese usuario
         training = training.filter( col("userId") =!= valorUnicoInt )
         val numeroFilas = training.count()
         println(s"Se han eliminado los registros del usuario por no cumplir condicion 2. El número de filas en el DataFrame es: $numeroFilas")
     }
 }

// Dejar solo la calificación más reciente que el usuario hizo a determinado ítem
// Se crea una ventana a partir del dataframe de training
training.createOrReplaceTempView("training")

// Asignar un id a cada combinación de userId e itemId y ordenar por timestamp
// Finalmente, se selecciona el primer elemento
// De esta forma se selecciona el ultimo rating para cada par unico de userId e itemId
training = spark.sql("""
WITH CTE AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY userId, itemId ORDER BY timestamp DESC) AS rating_order
  FROM training
)
SELECT * FROM CTE WHERE rating_order = 1;
""").drop("rating_order")

println("Registros training después de limpieza: " + training.count())

// Eliminación de columnas no usadas en conjunto de training
training = training.select("itemId", "userId", "rating")




















// Validación cruzada--------------------------------------------------------------------------


// Crear modelo ALS
val als = new ALS().
    setUserCol("userId").
    setItemCol("itemId").
    setRatingCol("rating")

// Crear "grid" de hiperparámetros
/*val paramGrid = new ParamGridBuilder().
    addGrid(als.rank, Array(10, 20, 50)).
    addGrid(als.regParam, Array(0.01, 0.1, 0.5)).
    addGrid(als.maxIter, Array(10, 15, 20)).
    addGrid(als.alpha, Array(0.01, 0.1,0.5)).
    build()*/
val paramGrid = new ParamGridBuilder().
    addGrid(als.rank, Array( 20)).
    build()

// Crear el evaluador por métrica RMSE
val evaluator = new RegressionEvaluator().
    setMetricName("rmse").
    setLabelCol("rating").
    setPredictionCol("prediction")

// Crear el validador cruzado
val cv1 = new CrossValidator().
    setEstimator(als).
    setEstimatorParamMaps(paramGrid).
    setEvaluator(evaluator).
    setNumFolds(2).
    setParallelism(2)

// Control del tiempo de la validación cruzada
println(s"Inicio: ${Calendar.getInstance.getTime}")
val cvmodel1 = cv1.fit(training)
println(s"Fin: ${Calendar.getInstance.getTime}")



// Seleccionar el mejor modelo
val model = cvmodel1.bestModel.asInstanceOf[ALSModel]

println(s"Mejor valor para 'rank': ${model.rank}")

//Guardar el modelo
model.write.overwrite().save(PATH + "modeloALS")
println("Modelo guardado")






















// Validación del modelo-----------------------------------------------------------------------

// Configurar la estrategia y generación de predicciones
// Estrategia que el modelo seguirá cuando se enfrente a datos de inicio en frío
// Usuarios o ítems nuevos que no tienen suficientes datos históricos para hacer predicciones precisas
model.setColdStartStrategy("drop")
val predictions = model.transform(training)

// Cálculo de la raíz del error cuadrático medio
val rmse = evaluator.evaluate(predictions)
println(s"Raíz del error cuadrático medio: $rmse")





























// Carga datos conjunto de test y modelo-------------------------------------------------------------


// Crear clase Rating, Item y User con los datos del conjunto de test y los datos guardados
case class RatingTest(
    itemId: Int,
    item: String,
    userId: Int,
    user: String,
    rating: Float,
    timestamp: Int
)

case class ItemDictTest(
    item: String,
    itemId: Int
)

case class UserDictTest(
    user: String,
    userId: Int
)





// A partir de un String genera un objeto RatingTest
def parseRatingTest(str: String): RatingTest = {
    // Separa los campos y valida que existan 6
    val fields = str.split(",")
    assert(fields.size == 6)

    // Utilizar campos
    RatingTest(
        fields(0).toInt,
        fields(1).toString,
        fields(2).toInt,
        fields(3).toString,
        fields(4).toFloat,
        fields(5).toInt
    )
}

// Convierte String a elemento UserDictTest
def parseUserLookup(str: String): UserDictTest = {
    val fields = str.split(",")
    assert(fields.size == 2)

    UserDictTest(
        fields(0).toString,
        fields(1).toInt
    )
}

// Lo mismo pero para ItemDictTest
def parseItemLookup(str: String): ItemDictTest = {
    val fields = str.split(",")
    assert(fields.size == 2)

    ItemDictTest(
        fields(0).toString,
        fields(1).toInt
    )
}

// Leer los datos y convertirlos en un DataFrame aplicando la conversión a RatingTest
val test = spark.read.textFile(PATH + ARCHIVO_TEST).map(parseRatingTest).toDF()
test.show(5)

// Cargar diccionarios para obtener el user e item a partir de sus Ids aplicando la conversión correspondiente
val dfUserLookup = spark.read.textFile(PATH + "dfUserLookup").map(parseUserLookup).toDF()
val dfItemLookup = spark.read.textFile(PATH + "dfItemLookup").map(parseItemLookup).toDF()
// Cargar el modelo creado
val model = ALSModel.load(PATH + ARCHIVO_MODELO)



























// Evaluación del modelo--------------------------------------------------------------------------------

// Crear evaluador por métrica RMSE
val evaluator = new RegressionEvaluator().
    setMetricName("rmse").
    setLabelCol("rating").
    setPredictionCol("prediction")

// Configurar la estrategia y generar predicciones
model.setColdStartStrategy("drop")
val predictions = model.transform(test)

// Calcular la raíz del error cuadrático medio
val rmse = evaluator.evaluate(predictions)
println(s"Raíz del error cuadrático medio: $rmse")






















// Generación de recomendaciones------------------------------------------------------------------

// Generar tres recomendaciones de items para un conjunto de usuarios
val users = test.select(model.getUserCol).distinct().limit(10)
println(users.getClass)

val userSubsetRecs = model.recommendForUserSubset(users, 3)
println(userSubsetRecs.getClass)
userSubsetRecs.show()

// Seleccionar un usuario de ejemplo para visualizar recomendaciones
val usuario = 463

println(s"Usuario:")
test.filter(col("userId") === usuario).show()

// Ver las recomendaciones
println("Recomendaciones:")
userSubsetRecs.filter(col("userId") === usuario).take(1)

// Detalle de las recomendaciones
dfItemLookup.
    filter(col("itemId") === 749 || col("itemId") === 375 ).
    select("itemId", "item").
    distinct().
    show()