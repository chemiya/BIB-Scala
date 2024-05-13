
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



//carga de datos
val PATH = "/home/usuario/Regresion/MiniProyecto2/"
val ARCHIVO_TEST = "testData"

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

val testData = assembler.transform(testRaw)

testData.show(5)


//evaluar modelo linear regression
println("\nEVALUACIÓN DE MODELO LinearRegressionModel")


val lrModel = LinearRegressionModel.load(PATH + "best_LinearRegressionModel")


val evaluator = new RegressionEvaluator()
  .setLabelCol("cnt")
  .setPredictionCol("prediction")
  .setMetricName("rmse")


val predictions = lrModel.transform(testData)
val rmse = evaluator.evaluate(predictions)


val metrics = evaluator.getMetrics(predictions)
println(s"MSE: ${metrics.meanSquaredError}")
println(s"R²: ${metrics.r2}")
println(s"root MSE: ${metrics.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metrics.meanAbsoluteError}")




//evaluar modelo gbt regression
println("\nEVALUACIÓN DE MODELO GBTRegressionModel")


val gbtModel = GBTRegressionModel.load(PATH + "best_GBTRegressionModel")


val evaluator1 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")


val predictions1 = gbtModel.transform(testData)
val rmse1 = evaluator1.evaluate(predictions1)


val metrics1 = evaluator1.getMetrics(predictions1)
println(s"MSE: ${metrics1.meanSquaredError}")
println(s"R²: ${metrics1.r2}")
println(s"root MSE: ${metrics1.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metrics1.meanAbsoluteError}")




//evaluar modelo randomforest
println("\nEVALUACIÓN DE MODELO RandomForestRegressionModel")


val rfModel = RandomForestRegressionModel.
    load(PATH + "best_RandomForestRegressionModel")


val evaluator3 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")


val predictions3 = rfModel.transform(testData)
val rmse3 = evaluator3.evaluate(predictions3)


val metrics3 = evaluator3.getMetrics(predictions3)
println(s"MSE: ${metrics3.meanSquaredError}")
println(s"R²: ${metrics3.r2}")
println(s"root MSE: ${metrics3.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metrics3.meanAbsoluteError}")


//evaluar modelo decission tree regresor
println("\nEVALUACIÓN DE MODELO DecisionTreeRegressor")


val dtModel = DecisionTreeRegressionModel.
    load(PATH + "best_DecisionTreeRegressionModel")


val evaluator2 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")


val predictions2 = dtModel.transform(testData)
val rmse2 = evaluator2.evaluate(predictions2)


val metrics2 = evaluator2.getMetrics(predictions2)
println(s"MSE: ${metrics2.meanSquaredError}")
println(s"R²: ${metrics2.r2}")
println(s"root MSE: ${metrics2.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metrics2.meanAbsoluteError}")


//seleccion del mejor modelo
println("\nSELECCIÓN DEL MEJOR MODELO")

println(s"RMSE en el conjunto de test para mejor modelo de LinearRegression: ${metrics.rootMeanSquaredError}")
println(s"RMSE en el conjunto de test para mejor modelo de GBTRegressor: ${metrics1.rootMeanSquaredError}")
println(s"RMSE en el conjunto de test para mejor modelo de DecisionTreeRegressor: ${metrics2.rootMeanSquaredError}")
println(s"RMSE en el conjunto de test para mejor modelo de RandomForestRegressor: ${metrics3.rootMeanSquaredError}")

println("\nGUARDADO DEL MEJOR MODELO: GBTRegressor")

gbtModel.write.overwrite().save(PATH + "modelo")


//evaluacion del mejor modelo
println("\nEVALUACIÓN DEL MEJOR MODELO (GBTRegressionModel)")


val bestModel = GBTRegressionModel.load(PATH + "modelo")


val bestEvaluator = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")


val bestPredictions = bestModel.transform(testData)
val bestRmse = bestEvaluator.evaluate(bestPredictions)


val bestMetrics = bestEvaluator.getMetrics(bestPredictions)
println(s"MSE: ${bestMetrics.meanSquaredError}")
println(s"R²: ${bestMetrics.r2}")
println(s"root MSE: ${bestMetrics.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${bestMetrics.meanAbsoluteError}")