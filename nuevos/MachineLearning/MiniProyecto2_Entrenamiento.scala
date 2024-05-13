
sc.setLogLevel("ERROR")

// Importación modulos
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
println("\nCARGA DE LOS DATOS")

val PATH = "/home/usuario/Regresion/MiniProyecto2/"
val ARCHIVO = "hour.csv"

val bikeDF = spark.read.format("csv").
    option("inferSchema", true).
    option("header", true).
    load(PATH + ARCHIVO)

println("Datos cargados:")
bikeDF.show(10)











//explorar los datos
println("\nEXPLORACIÓN DE LOS DATOS")


bikeDF.printSchema


bikeDF.
    select(
        bikeDF.columns.map(
            c => sum(col(c).isNull.cast("int")).alias(c)): _*
    ).show()






//analisis atributos
bikeDF.describe("instant").show()
bikeDF.describe("casual").show()
bikeDF.describe("registered").show()
bikeDF.describe("cnt").show()
bikeDF.describe("temp").show()
bikeDF.describe("atemp").show()
bikeDF.describe("hum").show()
bikeDF.describe("windspeed").show()
bikeDF.describe("hr").show()


println("Número de alquileres por época del año:")
bikeDF.
    groupBy("season").
    count().
    orderBy(asc("season")).
    withColumnRenamed("count", "cuenta").
    show()

println("Número de alquileres por año:")
bikeDF.
    groupBy("yr").
    count().
    orderBy(asc("yr")).
    withColumnRenamed("count", "cuenta").
    show()

println("Número de alquileres por mes:")
bikeDF.
    groupBy("mnth").
    count().
    orderBy(asc("mnth")).
    withColumnRenamed("count", "cuenta").
    show()


println("Número de alquileres por hora:")
bikeDF.
    groupBy("hr").
    count().
    orderBy(asc("hr")).
    withColumnRenamed("count", "cuenta").
    show(24)

println("Número de alquileres por día de la semana:")
bikeDF.
    groupBy("weekday").
    count().
    orderBy(asc("weekday")).
    withColumnRenamed("count", "cuenta").
    show()

println("Número de alquileres por meteorología:")
bikeDF.
    groupBy("weathersit").
    count().
    orderBy(asc("weathersit")).
    withColumnRenamed("count", "cuenta").
    show()

println("Número de alquileres por día festivo:")
bikeDF.
    groupBy("holiday").
    count().
    orderBy(asc("holiday")).
    withColumnRenamed("count", "cuenta").
    show()

println("Número de alquileres por día laboral:")
bikeDF.
    groupBy("workingday").
    count().
    orderBy(asc("workingday")).
    withColumnRenamed("count", "cuenta").
    show()









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










//particion datos
println("\nPARTICIÓN DE LOS DATOS")

val splitSeed = 123
val Array(trainingData, testData) = nuevoDF.
    randomSplit(Array(0.7, 0.3), splitSeed)

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


// validacion cruzada sobre regresion lineal
println("\nVALIDACIÓN CRUZADA PARA REGRESIÓN LINEAL")

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


lrModel.write.overwrite().save(PATH + "best_LinearRegressionModel")
println("Mejor modelo regresión lineal guardado")



// validacion cruzada sobre gbt regressor
println("\nVALIDACIÓN CRUZADA PARA REGRESOR GBT")


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


gbtModel.write.overwrite().save(PATH + "best_GBTRegressionModel")
println("Mejor modelo regresor GBT guardado")



//validacion cruzada sobre random forest
println("\nVALIDACIÓN CRUZADA PARA REGRESOR RF")


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


rfModel.write.overwrite().save(PATH + "best_RandomForestRegressionModel")
println("Mejor modelo regresor RF guardado")


//validacion cruzada sobre decissiontree
println("\nVALIDACIÓN CRUZADA PARA REGRESOR DT")


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


dtModel.write.overwrite().save(PATH + "best_DecisionTreeRegressionModel")
println("Mejor modelo regresor DT guardado")