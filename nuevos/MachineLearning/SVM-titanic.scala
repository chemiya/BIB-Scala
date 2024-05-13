import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LinearSVC
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

// Convertir variables categóricas a numéricas
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

// Dividir el dataset en 3 conjuntos
val Array(trainData, valData, testData) = labeledDF.randomSplit(Array(0.6, 0.2, 0.2), seed = 123)

// modelo SVM
val svm = new LinearSVC()

// rejilla de parámetros
val paramGrid = new ParamGridBuilder()
    .addGrid(svm.regParam, Array(0.01, 0.1, 1.0))
    .addGrid(svm.maxIter, Array(10, 100))
    .build()

// Configurar la evaluación
val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

// Configurar la validación cruzada
val trainValidationSplit = new TrainValidationSplit()
    .setEstimator(svm)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid.build())
    .setTrainRatio(0.8)

// Ajustar el modelo 
val model = trainValidationSplit.fit(trainData)

// Evaluar el modelo en el conjunto de validación
val results = model.transform(valData)
val accuracy = evaluator.evaluate(results)

println(s"Accuracy on validation set = $accuracy")

spark.stop()

