import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types.DoubleType

// Cargar el conjunto de datos 
val inicDF = spark.read.option("header", "true").csv("HousingData.csv")

// Convertir las columnas a Double
val doubleDF = inicDF.select(inicDF.columns.map(col(_) cast DoubleType): _*)

//eliminar filas con nulos
val housingDF = doubleDF.na.drop()


// Dividir el dataset en conjuntos de entrenamiento, validación y prueba
val Array(trainData, valData, testData) = housingDF.randomSplit(Array(0.6, 0.2, 0.2), seed = 123)

// Definir el modelo Random Forest
val rf = new RandomForestRegressor()
  .setLabelCol("MEDV")
  .setFeaturesCol("scaledFeatures")

// Construir el ensamblador de características
val assembler = new VectorAssembler()
  .setInputCols(housingDF.columns.dropRight(1)) // todas las columnas menos la última (target)
  .setOutputCol("features")

// Construir el escalador de características
val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(true)

// Construir la pipeline
val pipeline = new Pipeline()
  .setStages(Array(assembler, scaler, rf))

// Construir la rejilla de parámetros
val paramGrid = new ParamGridBuilder()
  .addGrid(rf.numTrees, Array(10, 20, 30))
  .addGrid(rf.maxDepth, Array(5, 10, 15))
  .build()

// Configurar la evaluación
val evaluator = new RegressionEvaluator()
  .setLabelCol("MEDV")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

// Configurar la validación cruzada para la selección del modelo
val trainValidationSplit = new TrainValidationSplit()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid.build())
  .setTrainRatio(0.8)

// Ajustar el modelo utilizando TrainValidationSplit
val model = trainValidationSplit.fit(trainData)

// Evaluar el modelo en el conjunto de validación
val results = model.transform(valData)
val rmse = evaluator.evaluate(results)

println(s"Root Mean Squared Error on validation set = $rmse")

// Detener la sesión de Spark
spark.stop()
