/*

Boston Housing Dataset 
Es un conjunto de datos clásico utilizado para el análisis de precios de viviendas. Contiene información sobre diversos aspectos socioeconómicos y físicos de viviendas en los alrededores de Boston. A continuación se describen los atributos del dataset:

-CRIM: Tasa de criminalidad per cápita por ciudad.
-ZN: Proporción de terrenos residenciales considerados para lotes de más de 25,000 pies cuadrados.
-INDUS: Proporción de acres comerciales no minoristas por ciudad.
-CHAS: Variable ficticia del río Charles (1 si el tramo limita con el río, 0 en caso contrario).
-NOX: Concentración de óxidos nítricos (partes por 10 millones).
-RM: Número promedio de habitaciones por vivienda.
-AGE: Proporción de unidades ocupadas por propietarios construidas antes de 1940.
-DIS: Distancias ponderadas a cinco centros de empleo de Boston.
-RAD: Índice de accesibilidad a carreteras radiales.
-TAX: Tasa de impuesto a la propiedad por cada $10,000.
-PTRATIO: Proporción alumno-maestro por ciudad. 
-B:1000(Bk−0.63)^2 donde Bk es la proporción de población afroamericana por ciudad.
-LSTAT: Porcentaje de la población con bajo estatus socioeconómico.
-MEDV: Valor medio de las viviendas ocupadas por sus propietarios en miles de dólares.



*/





import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler,MinMaxScaler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression}
import scala.collection.mutable.ArrayBuffer











// Cargar datos y preparar para los modelos--------------------------------------------------
// Cargar el conjunto de datos 
val inicDF = spark.read.option("header", "true").csv("HousingData.csv")

// Convertir las columnas a Double
val doubleDF = inicDF.select(inicDF.columns.map(col(_) cast DoubleType): _*)

// Eliminar filas con nulos
val housingDF = doubleDF.na.drop()


// Prepara los datos finales
val finalData = housingDF.withColumnRenamed("MEDV", "label")

// Dividir el dataset en conjuntos de entrenamiento, validación y prueba
val Array(trainingData, validationData, testData) = finalData.randomSplit(Array(0.6, 0.2, 0.2), seed = 123)


// VectorAssembler para convertir las características en un solo vector de características
val featureCols = Array("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT")
val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("unscaledFeatures")

// StandardScaler para escalar las características
/*val scaler = new StandardScaler()
  .setInputCol("unscaledFeatures")
  .setOutputCol("features")
  .setWithMean(true)
  .setWithStd(true)*/


// MinMaxScaler para escalar las características
val scaler = new MinMaxScaler()
  .setInputCol("unscaledFeatures")
  .setOutputCol("features")






















// Linear regression-------------------------------------------------------
// Define el modelo de regresión lineal
val lr = new LinearRegression()
  .setLabelCol("label")
  .setFeaturesCol("features")

// Construye una rejilla de parámetros para la búsqueda de hiperparámetros
/*val paramGridLR = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(0.1, 0.01))
  .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
  .build()*/
val paramGridLR = new ParamGridBuilder()
  .addGrid(lr.regParam, Array( 0.01))
  .build()

// Define el evaluador de regresión
val evaluatorLR = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("label")
  .setPredictionCol("prediction")

// Configura la validación cruzada
val crossvalLR = new CrossValidator()
  .setEstimator(new Pipeline().setStages(Array(assembler, scaler, lr)))
  .setEvaluator(evaluatorLR)
  .setEstimatorParamMaps(paramGridLR.build())
  .setNumFolds(3) 


// Ajusta el modelo usando validación cruzada
val cvModelLR = crossvalLR.fit(trainingData)

// Realiza predicciones en el conjunto de prueba
val predictionsLR = cvModelLR.transform(testData)
predictionsLR.select("features", "label", "prediction").show()

// Evalúa las métricas del modelo
val rmseLR = evaluatorLR.evaluate(predictionsLR)
println(s"Root Mean Squared Error (RMSE) en el conjunto de test = $rmseLR")





















// Random forest---------------------------------------------------------------

// Definir el modelo Random Forest
val rf = new RandomForestRegressor()
  .setLabelCol("label")
  .setFeaturesCol("features")


// Construir la rejilla de parámetros
/*val paramGridRF = new ParamGridBuilder()
  .addGrid(rf.numTrees, Array(10, 20, 30))
  .addGrid(rf.maxDepth, Array(5, 10, 15))
  .build()*/
val paramGridRF = new ParamGridBuilder()
  .addGrid(rf.numTrees, Array(20))
  .build()

// Configurar la evaluación
val evaluatorRF = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

// Configura la validación cruzada
val crossvalRF = new CrossValidator()
  .setEstimator(new Pipeline().setStages(Array(assembler, scaler, rf)))
  .setEvaluator(evaluatorRF)
  .setEstimatorParamMaps(paramGridRF.build())
  .setNumFolds(3)

// Ajustar el modelo utilizando TrainValidationSplit
val modelRF = crossvalRF.fit(trainingData)

// Evaluar el modelo en el conjunto de validación
val resultsRF = modelRF.transform(testData)
val rmseRF = evaluatorRF.evaluate(resultsRF)

println(s"Root Mean Squared Error en el conjunto de test = $rmseRF")


















//GBT regressor--------------------------------------------------------------

// Configura el modelo GBTRegressor
val gbt = new GBTRegressor()
  .setLabelCol("label")
  .setFeaturesCol("features")

// Construye una rejilla de parámetros para la búsqueda de hiperparámetros
/*val paramGridGBT = new ParamGridBuilder()
  .addGrid(gbt.maxDepth, Array(5, 10))
  .addGrid(gbt.maxIter, Array(50, 100))
  .build()*/
val paramGridGBT = new ParamGridBuilder()
  .addGrid(gbt.maxDepth, Array( 10))
  .build()

// Configura el evaluador de regresión
val evaluatorGBT = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("label")
  .setPredictionCol("prediction")

// Configura la validación cruzada
val crossvalGBT = new CrossValidator()
  .setEstimator(new Pipeline().setStages(Array(assembler, scaler, gbt)))
  .setEvaluator(evaluatorGBT)
  .setEstimatorParamMaps(paramGridGBT.build())
  .setNumFolds(3) // Usa 3 conjuntos para validación cruzada


// Ajusta el modelo usando validación cruzada
val cvModelGBT = crossvalGBT.fit(trainingData)

// Realiza predicciones en el conjunto de prueba
val predictionsGBT = cvModelGBT.transform(testData)
predictionsGBT.select("features", "label", "prediction").show()

// Evalúa las métricas del modelo
val rmseGBT = evaluatorGBT.evaluate(predictionsGBT)
println(s"Root Mean Squared Error (RMSE) en el conjunto de test = $rmseGBT")
























//Decissiontree regressor-------------------------------------------------------------
val dt = new DecisionTreeRegressor()
  .setLabelCol("label")
  .setFeaturesCol("features")


// Construye una rejilla de parámetros para la búsqueda de hiperparámetros
/*val paramGridDT = new ParamGridBuilder()
  .addGrid(dt.maxDepth, Array(5, 10))
  .addGrid(dt.maxBins, Array(32, 64))
  .build()*/
val paramGridDT = new ParamGridBuilder()
  .addGrid(dt.maxDepth, Array( 10))
  .build()

// Define el evaluador de regresión
val evaluatorDT = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("label")
  .setPredictionCol("prediction")

// Configura la validación cruzada
val crossvalDT = new CrossValidator()
  .setEstimator(new Pipeline().setStages(Array(assembler, scaler, dt)))
  .setEvaluator(evaluatorDT)
  .setEstimatorParamMaps(paramGridDT.build())
  .setNumFolds(3) // Usa 3 conjuntos para validación cruzada


// Ajusta el modelo usando validación cruzada
val cvModelDT = crossvalDT.fit(trainingData)

// Realiza predicciones en el conjunto de prueba
val predictionsDT = cvModelDT.transform(testData)
predictionsDT.select("features", "label", "prediction").show()

// Evalúa las métricas del modelo
val rmseDT = evaluatorDT.evaluate(predictionsDT)
println(s"Root Mean Squared Error (RMSE) en el conjunto de test = $rmseDT")


























//Evaluar modelos con una función
val evaluatorAll = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("label")
  .setPredictionCol("prediction")


// ArrayBuffer para almacenar los resultados
val results = ArrayBuffer[(String, Double)]()

// Función para ajustar y evaluar un modelo
def trainAndEvaluate(modelName: String, estimator: Pipeline): Unit = {
  val crossval = new CrossValidator()
    .setEstimator(estimator)
    .setEvaluator(evaluatorAll)
    .setEstimatorParamMaps(new ParamGridBuilder().build())
    .setNumFolds(3) // Usa 3 conjuntos para validación cruzada

  // Ajusta el modelo usando validación cruzada
  val cvModel = crossval.fit(trainingData)

  // Realiza predicciones en el conjunto de prueba
  val predictions = cvModel.transform(testData)

  // Evalúa las métricas del modelo
  val rmse = evaluatorAll.evaluate(predictions)
  results += ((modelName, rmse))
}

// Define los modelos y sus pipelines
val models = Seq(
  ("GBTRegressor", new Pipeline().setStages(Array(assembler, scaler, new GBTRegressor().setLabelCol("label").setFeaturesCol("features")))),
  ("DecisionTreeRegressor", new Pipeline().setStages(Array(assembler, scaler, new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("features"))))
)

// Entrena y evalúa cada modelo
models.foreach { case (name, pipeline) => trainAndEvaluate(name, pipeline) }

// Imprime los resultados
println("Modelo\t\tRMSE")
results.foreach { case (modelName, rmse) =>
  println(s"$modelName\t$rmse")
}






// Detener la sesión de Spark
spark.stop()
