import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types._

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

// Cargar el conjunto de datos del Titanic
val inicDF = spark.read.schema(schema).option("header", "true").csv("titanic.csv")

//eliminar filas con nulos
val titanicDF = inicDF.na.drop()

// Indexar y codificar variables categóricas
val indexer = new StringIndexer()
  .setInputCols(Array("Sex", "Embarked"))
  .setOutputCols(Array("SexIndex", "EmbarkedIndex"))

val assembler = new VectorAssembler()
  .setInputCols(Array("Pclass", "SexIndex", "EmbarkedIndex", "Age", "SibSp", "Parch", "Fare"))
  .setOutputCol("features")

// Definir el clasificador 
val dt = new DecisionTreeClassifier()
  .setLabelCol("Survived")
  .setFeaturesCol("features")

// Construir el pipeline
val pipeline = new Pipeline()
  .setStages(Array(indexer, assembler, dt))



// Dividir los datos 
val Array(trainData, validationData, testData) = titanicDF.randomSplit(Array(0.6, 0.2, 0.2), seed = 1234)

// métrica de evaluación
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("Survived")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

// cuadrícula de parámetros para la optimización
val paramGrid = new ParamGridBuilder()
  .addGrid(dt.maxDepth, Array(3, 5, 7))
  .addGrid(dt.maxBins, Array(24, 32, 40))
  .build()

// validador cruzado
val crossValidator = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid.build())
  .setNumFolds(5)

// Entrenar el modelo
val cvModel = crossValidator.fit(trainData)

// Evaluar el modelo en el conjunto de validación
val validationPredictions = cvModel.transform(validationData)
val validationAccuracy = evaluator.evaluate(validationPredictions)
println(s"Accuracy on validation set = $validationAccuracy")

// Hacer predicciones en el conjunto de prueba
val testPredictions = cvModel.transform(testData)
val testAccuracy = evaluator.evaluate(testPredictions)
println(s"Accuracy on test set = $testAccuracy")

// Detener la sesión de Spark
spark.stop()
