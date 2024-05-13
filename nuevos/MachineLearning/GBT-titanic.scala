import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassifier, GBTClassificationModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Definir el esquema 
val schema = StructType(Array(
  StructField("PassengerId", DoubleType, true),
  StructField("Survived", DoubleType, true),
  StructField("Pclass", DoubleType, true),
  StructField("Name", StringType, true),
  StructField("Sex", StringType, true),
  StructField("Age", DoubleType, true),
  StructField("SibSp", DoubleType, true),
  StructField("Parch", DoubleType, true),
  StructField("Ticket", StringType, true),
  StructField("Fare", DoubleType, true),
  StructField("Cabin", StringType, true),
  StructField("Embarked", StringType, true)
))

// Cargar el conjunto 
val inicDF = spark.read.schema(schema).option("header", "true").csv("titanic.csv")

//eliminar filas con nulos
val titanicDF = inicDF.na.drop()

// Preprocesamiento: Convertir variables categóricas a numéricas
val genderIndexer = new StringIndexer()
  .setInputCol("Sex")
  .setOutputCol("SexIndex")
  .fit(titanicDF)
val embarkedIndexer = new StringIndexer()
  .setInputCol("Embarked")
  .setOutputCol("EmbarkedIndex")
  .fit(titanicDF)

val assembler = new VectorAssembler()
  .setInputCols(Array("Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "EmbarkedIndex"))
  .setOutputCol("features")

//  Convertir variables categóricas a numéricas
val genderIndexerModel: StringIndexerModel = genderIndexer.fit(titanicDF)
val embarkedIndexerModel: StringIndexerModel = embarkedIndexer.fit(titanicDF)

val transformedDF = assembler.transform(embarkedIndexerModel.transform(genderIndexerModel.transform(titanicDF)))
val labeledDF = transformedDF.withColumnRenamed("Survived", "label")

// Dividir el dataset en conjuntos de entrenamiento, validación y prueba
val Array(trainData, valData, testData) = labeledDF.randomSplit(Array(0.6, 0.2, 0.2), seed = 123)

// Definir el modelo GBT
val gbt = new GBTClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")

// Construir la pipeline
val pipeline = new Pipeline()
  .setStages(Array( gbt))

// Construir la rejilla de parámetros
val paramGrid = new ParamGridBuilder()
  .addGrid(gbt.maxDepth, Array(3, 5))
  .addGrid(gbt.maxBins, Array(24, 32))
  .addGrid(gbt.maxIter, Array(10, 20))
  .build()

// Configurar la evaluación
val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

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
val accuracy = evaluator.evaluate(results)

println(s"Accuracy on validation set = $accuracy")

// Detener la sesión de Spark
spark.stop()