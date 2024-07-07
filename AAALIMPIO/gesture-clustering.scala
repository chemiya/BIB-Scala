/*
Gesture Phase Segmentation Dataset
Es un conjunto de datos utilizado para la investigación y desarrollo de algoritmos de segmentación y reconocimiento de gestos humanos. Este dataset está diseñado para ayudar a identificar las diferentes fases de un gesto, como preparación, ejecución y finalización. A continuación se describen los atributos más importantes del Gesture Phase Segmentation Dataset:

-gesture_id: Identificador único para cada gesto registrado.
-timestamp: Marca de tiempo que indica el momento exacto en que se registró el dato.
-x_acceleration: Aceleración en el eje X.
-y_acceleration: Aceleración en el eje Y.
-z_acceleration: Aceleración en el eje Z.
-roll: Ángulo de rotación alrededor del eje X.
-pitch: Ángulo de rotación alrededor del eje Y.
-yaw: Ángulo de rotación alrededor del eje Z.
-phase: Fase del gesto en ese momento (p.ej., Preparación, Ejecución, Finalización).
*/







import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel}
import org.apache.spark.ml.feature.{ StandardScaler}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.evaluation.ClusteringEvaluator













// Carga de datos---------------------------------------------------------------
val PATH = "/home/usuario/bib-scala/"
val ARCHIVO = "a1_merged.csv"

val gestureDF = spark.read.format("csv").
    option("inferSchema", true).
    option("header", true).
    load(PATH + ARCHIVO)

println("Datos cargados:")
gestureDF.show(10)


//Atributos para probar diferentes combinaciones
val eliminarAtributos: Boolean = true
val transformacion: Boolean=true






















// Análisis atributos----------------------------------------------------------------------
val atributosNumericos = Seq("lhx",
 "lhy",
 "lhz",
 "rhx",
 "rhy",
 "rhz",
 "hx",
 "hy",
 "hz",
 "sx",
 "sy",
 "sz",
 "lwx",
 "lwy",
 "lwz",
 "rwx",
 "rwy",
 "rwz",
 "1",
 "2",
 "3",
 "4",
 "5",
 "6",
 "7",
 "8",
 "9",
 "10",
 "11",
 "12",
 "13",
 "14",
 "15",
 "16",
 "17",
 "18",
 "19",
 "20",
 "21",
 "22",
 "23",
 "24",
 "25",
 "26",
 "27",
 "28",
 "29",
 "30",
 "31",
 "32")

for (atributo <- atributosNumericos) {
    gestureDF.describe(atributo).show()
}

println("\nAnálisis de la clase:")
gestureDF.groupBy("Phase").count().orderBy(asc("Phase")).withColumnRenamed("count", "cuenta").show()

























// Seleccionar columnas-------------------------------------------------------
val columnasAEliminarVarias = Seq("lwx", "lwy", "lwz", "rwx", "rwy", "rwz", "sx", "7", "8", "9", "10", "11", "12", "20", "23", "27", "28", "31", "32","timestamp")
val columnasAEliminarTimestamp = Seq("timestamp")

val columnasAEliminar = if (eliminarAtributos) columnasAEliminarVarias else columnasAEliminarTimestamp

val gestureDFColumnasSeleccion = gestureDF.drop(columnasAEliminar: _*)
gestureDFColumnasSeleccion.show()





















// Normalizar datos---------------------------------------------------------------------
val assembler = new VectorAssembler()
  .setInputCols(gestureDFColumnasSeleccion.columns.diff(Array("Phase")))
  .setOutputCol("featuresAssembler")

val dfConVector = assembler.transform(gestureDFColumnasSeleccion)
dfConVector.show()


val scaler = if (transformacion) {
  new MinMaxScaler()
    .setInputCol("featuresAssembler")
    .setOutputCol("features")
} else {
  new StandardScaler()
    .setInputCol("featuresAssembler")
    .setOutputCol("features")
}


val scalerModel = scaler.fit(dfConVector)


val scaledDataGestureDF = scalerModel.transform(dfConVector).select("features","Phase")


println("DataFrame con datos escalados:")
scaledDataGestureDF.show()



// Sin ninguna normalización
/*val vecAssembler = new VectorAssembler()
  .setInputCols(gestureDFColumnasSeleccion.columns.diff(Array("Phase")))
  .setOutputCol("features")
val scaledDataGestureDF = vecAssembler.transform(gestureDFColumnasSeleccion).select("features","Phase")
scaledDataGestureDF.show()
*/
























// Creación del modelo-----------------------------------------------------
// Clustering con Kmeans
val kmeans = new KMeans()
  .setK(5) 
  .setMaxIter(500)
  .setTol(0.0001)
  .setDistanceMeasure("cosine")

val model = kmeans.fit(scaledDataGestureDF)

val predictions = model.transform(scaledDataGestureDF)

println("Centroides:")
model.clusterCenters.foreach(println)

println("Predicciones del clustering:")
predictions.show()



























// Métricas--------------------------------------------------------------------
val evaluator = new ClusteringEvaluator()
var silhouette = evaluator.evaluate(predictions)
println(s"Silhouette = $silhouette")

val numClusters = 2 to 20 toArray
for(k<-numClusters){
    var model = kmeans.setK(k).fit(scaledDataGestureDF)
    var predicts = model.transform(scaledDataGestureDF)
    var silhouette = evaluator.evaluate(predicts)
    println(s"Silhouette(k= ${k}) = ${silhouette}")
}



























// Tabla de contingencia y guardar predicciones---------------------------------------------------
val indexer = new StringIndexer()
  .setInputCol("Phase")
  .setOutputCol("phaseIndex")

val indexedData = indexer.fit(predictions).transform(predictions)
indexedData.show()
var tablaContingencia = indexedData.groupBy("phaseIndex").pivot("prediction").count().na.fill(0)
tablaContingencia = tablaContingencia.drop("phaseIndex")
tablaContingencia.show()
val predicionesIndices=indexedData.select("phaseIndex","prediction")


predicionesIndices.write.mode("overwrite").csv("predicionesIndices")
