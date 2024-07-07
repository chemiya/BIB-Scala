/*
Bike Rental Dataset
Es un conjunto de datos que contiene información sobre el uso de bicicletas compartidas en una ciudad, registrando el número de bicicletas alquiladas por hora y otros atributos relacionados. A continuación, se describen los atributos de este dataset:
-instant: Identificador único de cada registro.
-dteday: Fecha en formato YYYY-MM-DD.
-season: Estación del año (1: invierno, 2: primavera, 3: verano, 4: otoño).
-yr: Año (0: 2011, 1: 2012).
-mnth: Mes del año (1 a 12).
-hr: Hora del día (0 a 23).
-holiday: Indica si el día es festivo (0: no, 1: sí).
-weekday: Día de la semana (0: domingo, 1: lunes, ..., 6: sábado).
-workingday: Indica si el día es laborable (0: no, 1: sí).
-weathersit: Condición climática (1: despejado, 2: nublado, 3: lluvia ligera, 4: lluvia fuerte).
-temp: Temperatura normalizada en una escala de 0 a 1 (temperatura real en grados Celsius escalada/41).
-atemp: Sensación térmica normalizada en una escala de 0 a 1 (sensación térmica real en grados Celsius escalada/50).
-hum: Humedad relativa normalizada (0 a 1).
-windspeed: Velocidad del viento normalizada (0 a 1).
-casual: Número de alquileres de usuarios ocasionales.
-registered: Número de alquileres de usuarios registrados.
-cnt: Total de alquileres de bicicletas (casual + registered).
*/






sc.setLogLevel("ERROR")
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, LinearRegressionSummary}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}














// Carga de datos--------------------------------------------------------------
val PATH = "/home/usuario/bib-scala/"
val ARCHIVO = "hour.csv"
val ARCHIVO_TEST = "testData"

val bikeDF = spark.read.format("csv").
    option("inferSchema", true).
    option("header", true).
    load(PATH + ARCHIVO)

println("Datos cargados:")
bikeDF.show(10)

println("Esquema:")
bikeDF.printSchema























// Exploración de los datos--------------------------------------------------------------------
// Contar nulos
bikeDF.select(bikeDF.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show()


// Análisis atributos
val listaAtributosAnalizar=List("instant","casual","registered","cnt","temp","atemp","hum","windspeed","hr")
for (nombreColumna <- listaAtributosAnalizar) {
    bikeDF.describe(nombreColumna).show()
}

val listaAtributosAgrupar=List("season","yr","mnth","hr","weekday","weathersit","holiday","workingday") 
for (nombreColumnaGroup <- listaAtributosAnalizar) {
    bikeDF.groupBy(nombreColumnaGroup).count().orderBy(asc(nombreColumnaGroup)).withColumnRenamed("count", "cuenta").show()
}

























// Preparación de los datos--------------------------------------------------------------------------

// Eliminar atributos no necesarios
val columnasAEliminar = Seq(
    "instant",
    "dteday",
    "atemp",
    "windspeed",
    "casual",
    "registered"
)

val nuevoDF = bikeDF.drop(columnasAEliminar: _*)
nuevoDF.count()



// Partición datos
val splitSeed = 123
val Array(trainingData, testData) = nuevoDF.randomSplit(Array(0.7, 0.3), splitSeed)

testData.write.mode("overwrite").csv(PATH + "testData")
println("Conjunto de pruebas guardado")
testData.show(5)


val featureCols = Array(
    "holiday",
    "workingday",
    "temp",
    "hum",
    "season",
    "yr",
    "mnth",
    "hr",
    "weekday",
    "weathersit"
)

val assembler = new VectorAssembler().
    setInputCols(featureCols).
    setOutputCol("features")




























// Linear regression--------------------------------------------------------------------------

val lr = new LinearRegression().
    setLabelCol("cnt").
    setFeaturesCol("features")

val pipeline = new Pipeline().
    setStages(Array(assembler, lr))

val paramGrid = new ParamGridBuilder().
    addGrid(lr.regParam, Array(0.1)).
    addGrid(lr.elasticNetParam, Array(0.5)).
    build()


val evaluator = new RegressionEvaluator()
  .setLabelCol("cnt")
  .setPredictionCol("prediction")
  .setMetricName("rmse")


val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3) 


val cvModel = cv.fit(trainingData)

println("\nPARÁMETROS MEJOR MODELO REGRESIÓN LINEAL")
val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
val lrModel = bestModel.stages(1).asInstanceOf[LinearRegressionModel]
println(s"""Parámetros del mejor modelo:
regParam = ${lrModel.getRegParam}, elasticNetParam = ${lrModel.getElasticNetParam}
""")


lrModel.write.overwrite().save(PATH + "modelos/best_LinearRegressionModel")
println("Mejor modelo regresión lineal guardado")



























// GBT regressor------------------------------------------------------------------------------

val gbt = new GBTRegressor().
    setLabelCol("cnt").
    setFeaturesCol("features")


val pipeline1 = new Pipeline().
    setStages(Array(assembler, gbt))


val paramGrid1 = new ParamGridBuilder().
    addGrid(gbt.maxDepth, Array(5)).
    addGrid(gbt.maxIter, Array(10)).
    build()


val evaluator1 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")


val cv1 = new CrossValidator().
    setEstimator(pipeline1).
    setEvaluator(evaluator1).
    setEstimatorParamMaps(paramGrid1).
    setNumFolds(3) 


val cvModel1 = cv1.fit(trainingData)

println("\nPARÁMETROS MEJOR MODELO REGRESOR GBT")
val bestModel1 = cvModel1.bestModel.asInstanceOf[PipelineModel]
val gbtModel = bestModel1.stages(1).asInstanceOf[GBTRegressionModel]
println(s"""Parámetros del mejor modelo:
maxDepth = ${gbtModel.getMaxDepth}, maxIter = ${gbtModel.getMaxIter}
""")


gbtModel.write.overwrite().save(PATH + "modelos/best_GBTRegressionModel")
println("Mejor modelo regresor GBT guardado")

























// Random Forest--------------------------------------------------------------------

val rf = new RandomForestRegressor().
    setLabelCol("cnt").
    setFeaturesCol("features")


val pipeline3 = new Pipeline().
    setStages(Array(assembler, rf))


val paramGrid3 = new ParamGridBuilder().
    addGrid(rf.maxDepth, Array(5)).
    addGrid(rf.numTrees, Array(5)).
    build()


val evaluator3 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")


val cv3 = new CrossValidator().
    setEstimator(pipeline3).
    setEvaluator(evaluator3).
    setEstimatorParamMaps(paramGrid3).
    setNumFolds(3) 


val cvModel3 = cv3.fit(trainingData)

println("\nPARÁMETROS MEJOR MODELO REGRESOR RF")
val bestModel3 = cvModel3.bestModel.asInstanceOf[PipelineModel]
val rfModel = bestModel3.stages(1).asInstanceOf[RandomForestRegressionModel]
println(s"""Parámetros del mejor modelo:
maxDepth = ${rfModel.getMaxDepth}, numTrees = ${rfModel.getNumTrees}
""")


rfModel.write.overwrite().save(PATH + "modelos/best_RandomForestRegressionModel")
println("Mejor modelo regresor RF guardado")

























// Decission tree-------------------------------------------------------------------------------

val dt = new DecisionTreeRegressor().
    setLabelCol("cnt").
    setFeaturesCol("features")


val pipeline2 = new Pipeline().
    setStages(Array(assembler, dt))


val paramGrid2 = new ParamGridBuilder().
    addGrid(dt.maxDepth, Array(5)).
    addGrid(dt.maxBins, Array(16)).
    build()


val evaluator2 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")


val cv2 = new CrossValidator().
    setEstimator(pipeline2).
    setEvaluator(evaluator2).
    setEstimatorParamMaps(paramGrid2).
    setNumFolds(3)


val cvModel2 = cv2.fit(trainingData)

println("\nPARÁMETROS MEJOR MODELO REGRESOR DT")
val bestModel2 = cvModel2.bestModel.asInstanceOf[PipelineModel]
val dtModel = bestModel2.stages(1).asInstanceOf[DecisionTreeRegressionModel]
println(s"""Parámetros del mejor modelo:
maxDepth = ${dtModel.getMaxDepth}, maxBins = ${dtModel.getMaxBins}
""")


dtModel.write.overwrite().save(PATH + "modelos/best_DecisionTreeRegressionModel")
println("Mejor modelo regresor DT guardado")































// Carga de los datos de test---------------------------------------------------------------------------------
val testRaw = spark.read.format("csv").
    option("inferSchema", true).
    load(PATH + ARCHIVO_TEST).
    toDF(
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "hum",
        "cnt"
    )

val testDataLoaded = assembler.transform(testRaw)
testDataLoaded.show(5)




























// Evaluación regresión lineal-----------------------------------------------------------------
val lrModel = LinearRegressionModel.load(PATH + "modelos/best_LinearRegressionModel")

val evaluatorLR = new RegressionEvaluator()
  .setLabelCol("cnt")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val predictionsLR = lrModel.transform(testDataLoaded)
val rmseLR = evaluatorLR.evaluate(predictionsLR)

val metricsLR = evaluatorLR.getMetrics(predictionsLR)
println(s"MSE: ${metricsLR.meanSquaredError}")
println(s"R²: ${metricsLR.r2}")
println(s"root MSE: ${metricsLR.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricsLR.meanAbsoluteError}")
























// Evaluación GBT regression---------------------------------------------------------

val gbtModel = GBTRegressionModel.load(PATH + "modelos/best_GBTRegressionModel")

val evaluatorGBT = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")


val predictionsGBT = gbtModel.transform(testDataLoaded)
val rmseGBT = evaluatorGBT.evaluate(predictionsGBT)


val metricsGBT = evaluatorGBT.getMetrics(predictionsGBT)
println(s"MSE: ${metricsGBT.meanSquaredError}")
println(s"R²: ${metricsGBT.r2}")
println(s"root MSE: ${metricsGBT.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricsGBT.meanAbsoluteError}")























// Evaluar Random forest-------------------------------------------------------------------


val rfModel = RandomForestRegressionModel.load(PATH + "modelos/best_RandomForestRegressionModel")

val evaluatorRF = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")


val predictionsRF = rfModel.transform(testDataLoaded)
val rmseRF = evaluatorRF.evaluate(predictionsRF)


val metricsRF = evaluatorRF.getMetrics(predictionsRF)
println(s"MSE: ${metricsRF.meanSquaredError}")
println(s"R²: ${metricsRF.r2}")
println(s"root MSE: ${metricsRF.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricsRF.meanAbsoluteError}")

























// Evaluación Decission tree----------------------------------------------------------------

val dtModel = DecisionTreeRegressionModel.load(PATH + "modelos/best_DecisionTreeRegressionModel")


val evaluatorDT = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")


val predictionsDT = dtModel.transform(testDataLoaded)
val rmseDT = evaluatorDT.evaluate(predictionsDT)


val metricsDT = evaluatorDT.getMetrics(predictionsDT)
println(s"MSE: ${metricsDT.meanSquaredError}")
println(s"R²: ${metricsDT.r2}")
println(s"root MSE: ${metricsDT.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricsDT.meanAbsoluteError}")




























// Seelección del mejor modelo--------------------------------------------------------------
println("\nSELECCIÓN DEL MEJOR MODELO")

println(s"RMSE en el conjunto de test para mejor modelo de LinearRegression: ${metricsLR.rootMeanSquaredError}")
println(s"RMSE en el conjunto de test para mejor modelo de GBTRegressor: ${metricsGBT.rootMeanSquaredError}")
println(s"RMSE en el conjunto de test para mejor modelo de DecisionTreeRegressor: ${metricsDT.rootMeanSquaredError}")
println(s"RMSE en el conjunto de test para mejor modelo de RandomForestRegressor: ${metricsRF.rootMeanSquaredError}")

println("\nGUARDADO DEL MEJOR MODELO: GBTRegressor")
gbtModel.write.overwrite().save(PATH + "modelos/mejor_modelo")


























// Evaluación del mejor modelo---------------------------------------------
println("\nEVALUACIÓN DEL MEJOR MODELO (GBTRegressionModel)")


val bestModel = GBTRegressionModel.load(PATH + "modelos/mejor_modelo")

val bestEvaluator = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")

val bestPredictions = bestModel.transform(testDataLoaded)
val bestRmse = bestEvaluator.evaluate(bestPredictions)


val bestMetrics = bestEvaluator.getMetrics(bestPredictions)
println(s"MSE: ${bestMetrics.meanSquaredError}")
println(s"R²: ${bestMetrics.r2}")
println(s"root MSE: ${bestMetrics.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${bestMetrics.meanAbsoluteError}")