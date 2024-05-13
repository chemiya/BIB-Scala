import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.StringIndexerModel


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

// Dividir el dataset en conjuntos
val Array(trainData, valData, testData) = labeledDF.randomSplit(Array(0.6, 0.2, 0.2), seed = 123)

// regresión logística
val lr = new LogisticRegression()

// rejilla de parámetros
val paramGrid = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(0.01, 0.1, 0.3))
    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
    .addGrid(lr.maxIter, Array(10, 100))
    .build()

// evaluador
val evaluator = new BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC")

// validación cruzada para la selección del modelo
val trainValidationSplit = new TrainValidationSplit()
    .setEstimator(lr)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid.build())
    .setTrainRatio(0.8)

// Ajustar el modelo 
val model = trainValidationSplit.fit(trainData)

// Evaluar el modelo en el conjunto de validación
val results = model.transform(valData)
val areaUnderROC = evaluator.evaluate(results)

println(s"Area under ROC on validation set = $areaUnderROC")

spark.stop()

