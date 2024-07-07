//spark-shell
//:load programa.scala


/*
Adult income dataset
Este conjunto de datos se compone de varias características que describen las características demográficas y laborales de las personas y se utiliza para predecir si una persona gana más de $50,000 al año o no. Consta de los siguientes atributos:
-age: Edad de la persona en años. Tipo numérico.
-workclass: Tipo de empleador o clase de trabajo. Tipo categórico.
-fnlwgt: Peso final de la persona en el estudio del censo. Tipo numérico.
-education: Nivel educativo alcanzado. Tipo categórico.
-education_num: Años de educación completados. Tipo numérico.
-marital_status: Estado civil de la persona. Tipo categórico.
-occupation: Ocupación o tipo de trabajo de la persona. Tipo categórico.
-relationship: Relación de la persona con el cabeza de familia. Tipo categórico.
-race: Raza de la persona. Tipo categórico.
-sex: Sexo de la persona (Masculino o Femenino). Tipo categórico.
-capital_gain: Ganancias obtenidas por inversiones. Tipo numérico.
-capital_loss: Pérdidas incurridas por inversiones. Tipo numérico.
-hours_per_week: Número de horas trabajadas por semana. Tipo numérico.
-native_country: País de origen de la persona. Tipo categórico.
-income: Clase de ingresos (">50K" o "<=50K"). Tipo categórico.

*/

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SparkSession, DataFrame,Row}
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler,StringIndexerModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.{GBTClassifier, GBTClassificationModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.StringIndexerModel






















//Carga de datos y operaciones basicas---------------------------------------------


// Cargar el conjunto de datos 
var adultDF = spark.read.option("header", "true").csv("adult-income.csv")



// Imprimir el esquema
println("Esquema del dataset:")
adultDF.printSchema()

// Imprimir los datos
println("Muestra del dataset:")
adultDF.show()

// Tipo de variable
println("Tipo de la variable:")
println(adultDF.getClass)

// Nombres de las columnas
println("Nombres de las columnas:")
adultDF.columns.foreach(println)

// Mostrar solo dos columnas
println("Datos de workclass y capital_loss:")
adultDF.select("workclass","capital_loss").show()

// Tipos de las columnas
println("Tipos de las columnas:")
adultDF.dtypes.foreach(println)

// Convertir columna de string a int con la función cast()
// var se puede reasignar
adultDF= adultDF.withColumn("age", col("age").cast(IntegerType))


// Convertir columna a double
adultDF = adultDF.withColumn("capital_loss", adultDF("capital_loss").cast(DoubleType))


// Convertir columnas y evaluar si existen nulos
adultDF = adultDF.withColumn("capital_gain", when(col("capital_gain").isNull || col("capital_gain") === "", 0).otherwise(col("capital_gain").cast(DoubleType))).withColumn("age", when(col("age").isNull || col("age") === "", 0.0).otherwise(col("age").cast(IntegerType)))


// Tipos de las columnas
println("Tipos de las columnas:")
adultDF.dtypes.foreach(println)

// Resumen de los datos
println("Resumen de age y capital_loss:")
adultDF.describe("age","capital_loss").show()


// Crear columna
adultDF = adultDF.withColumn("capital_loss_multiplied", col("capital_loss") * 0.21)

// Renombrar columna
adultDF = adultDF.withColumnRenamed("capital_loss_multiplied", "new_capital_loss")

// Mostrar datos
println("Dataset con la nueva columna:")
adultDF.show()

// Eliminar columna
adultDF = adultDF.drop("new_capital_loss")

// Mostrar otra vez
println("Eliminando la columna:")
adultDF.show()

























//Carga de los datos-----------------------------------------------------------------

// Definir el esquema personalizado 
val adultSchema = StructType(Array(
  StructField("age", IntegerType, nullable = true),
  StructField("workclass", StringType, nullable = true),
  StructField("fnlwgt", DoubleType, nullable = true),
  StructField("education", StringType, nullable = true),
  StructField("education_num", StringType, nullable = true),
  StructField("marital_status", StringType, nullable = true),
  StructField("occupation", StringType, nullable = true),
  StructField("relationship", StringType, nullable = true),
  StructField("race", StringType, nullable = true),
  StructField("sex", StringType, nullable = true),
  StructField("capital_gain", DoubleType, nullable = true),
  StructField("capital_loss", DoubleType, nullable = true),
  StructField("hours_per_week", IntegerType, nullable = true),
  StructField("native_country", StringType, nullable = true),
  StructField("income", IntegerType, nullable = true)
))



// Cargar el datos con el esquema personalizado
adultDF = spark.read.schema(adultSchema).option("header", "true").csv("adult-income.csv")

// Agregar indice
adultDF = adultDF.withColumn("index", monotonically_increasing_id())

// Imprimir los datos
println("Muestra del dataset:")
adultDF.show()

// Tipos de las columnas
println("Tipos de las columnas:")
adultDF.dtypes.foreach(println)


























//Tratamiento nulos------------------------------------------------------------------------

// Cuantas filas con nulos
var nullRowsCount = adultDF.filter(row => row.anyNull).count()
println(s"Número de filas con al menos un nulo: $nullRowsCount")

// Filas con nulos 
val nullRows = adultDF.filter(row => row.anyNull)
nullRows.show()


// Nulos por columnas
println("Número de nulos por columnas:")
adultDF.select(adultDF.columns.map(c => sum(when(col(c).isNull, 1).otherwise(0)).alias(c)): _*).show()


// Eliminar las filas con nulo en atributos
adultDF = adultDF.na.drop(Seq( "education","relationship"))


// Media de la columna age
val meanAge = adultDF.select(avg("Age")).head().getDouble(0)

// Completar los nulos de age con la media
adultDF = adultDF.na.fill(meanAge, Seq("Age"))

// Reemplazar nulos por cadena
adultDF= adultDF.na.fill(Map("occupation" -> "?"))

// Eliminar filas con nulos
adultDF= adultDF.na.drop()

// Mostrar filas que tenian nulos
var indexToShow=1
var row = adultDF.filter(adultDF("index") === indexToShow).show()

// Lista de índices a mostrar
val indicesToShow = Seq(1, 2, 9, 13, 14)
// Filtrar el DataFrame por los índices especificados
val filteredRows = adultDF.filter($"index".isin(indicesToShow:_*))
// Mostrar las filas filtradas
filteredRows.show()

// Cuantas filas con nulos
nullRowsCount = adultDF.filter(row => row.anyNull).count()
println(s"Número de filas con al menos un nulo: $nullRowsCount")

// Nulos por columnas
println("Número de nulos por columnas:")
adultDF.select(adultDF.columns.map(c => sum(when(col(c).isNull, 1).otherwise(0)).alias(c)): _*).show()


























// Otras operaciones de limpieza-------------------------------------------------------


// Obtener una lista de las columnas del DataFrame
val columns = adultDF.columns
print(columns)

// Crear una secuencia de expresiones de conteo condicional para cada columna
val countExpressions = columns.map(colName => count(when(col(colName) === lit("?"), 1)).alias(colName))

// Aplicar las expresiones de conteo para contar el número de veces que aparece "?" en cada columna
val counts = adultDF.agg(countExpressions.head, countExpressions.tail: _*)
// Mostrar el resultado
counts.show()

// Eliminar columna con valores de educacion
adultDF = adultDF.drop("education_num")

// Calcular la moda de la columna
val modeValue = adultDF.groupBy("native_country").count().orderBy($"count".desc).select("native_country").first()(0)

// Reemplazar las apariciones de "?" por la moda de la columna
adultDF = adultDF.withColumn("native_country", when(col("native_country") === lit("?"), modeValue).otherwise(col("native_country")))

// Mostrar el DataFrame con los valores reemplazados
adultDF.show()



// Mostrar el número de filas y columnas 
val numRows = adultDF.count()
val numCols = adultDF.columns.length
println(s"Número de filas: $numRows")
println(s"Número de columnas: $numCols")



// detectar y reemplazar outliers en el atributo 'age'
def replaceOutliers(df: DataFrame, column: String): DataFrame = {
  val quantiles = df.stat.approxQuantile(column, Array(0.25, 0.75), 0.05)
  val q1 = quantiles(0)
  val q3 = quantiles(1)
  val iqr = q3 - q1
  val lowerRange = q1 - 1.5 * iqr
  val upperRange = q3 + 1.5 * iqr
  df.withColumn(column, when(col(column) < lowerRange || col(column) > upperRange, null).otherwise(col(column)))
}

// Reemplazar outliers
adultDF = replaceOutliers(adultDF, "age").na.drop("any", Seq("age"))





















//Operaciones de analisis---------------------------------------------------------------------------

// Filtrar filas con mayor a 20 de edad
val filteredAgeDF = adultDF.filter(adultDF("age") > 20)

// Mostrar resultados
println("Dataframe filtrando por mayores de 20:")
filteredAgeDF.show()

// Filtrar filas con mas de 25 hours_per_week y marital_status como Widowed
val filteredHoursMaritalDF = adultDF.filter(adultDF("hours_per_week") > 25 && adultDF("marital_status") === "Widowed")

println("Dataframe filtrando por hours_per_week y marital_status:")
filteredHoursMaritalDF.show()



// suma de capital_loss agrupando por education
val lossSumDF = adultDF.groupBy("education").sum("capital_loss").withColumnRenamed("sum(capital_loss)", "total_capital_loss")

println("Suma total de capital_loss agrupando por education:")
lossSumDF.show()


// media de edad agrupando por relationship
val ageMeanDF = adultDF.groupBy("relationship").avg("age").withColumnRenamed("avg(age)", "average_age")

println("Media de edad por relationship:")
ageMeanDF.show()


// agrupar por sexo y media de los que sobrevivien y suma de las capital_loss
val groupedLowLossDF = adultDF.groupBy("sex").agg(
  avg(when($"income" === 0, $"age")).as("Avg_Age_Low"),
  sum($"capital_loss").as("total_capital_loss")
)

println("Agrupando por sexo - Media de edad para income <=50K y suma total de capital_loss:")
groupedLowLossDF.show()



// Agrupar por race y contar cuantas personas hay
val raceCount = adultDF.groupBy("race").count().orderBy("race")
println("Número de personas por raza:")
raceCount.show()



// Contar hombres y >50K
val male50KCount = adultDF.filter($"sex" === "male" && $"income" === 1).count()
println(s"Número de hombres con >50K: $male50KCount")

// Valores diferentes columna education
println("Valores diferentes de la columna education:")
adultDF.select("education").distinct().show()



// nueva columna en base al atributo 'age' marcando en qué intervalo de edad 
adultDF = adultDF.withColumn("age_interval", floor(col("age") / 5) * 5)
val ageFreq  = adultDF.groupBy("age_interval").count()
ageFreq.show()


// Agrupar por marital_status, race y sex 
val groupedDF = adultDF.groupBy("marital_status", "sex", "race").count().orderBy("marital_status", "sex", "race")
println("Agrupando por marital_status, sex y race:")
groupedDF.show()


// tabla de frecuencias del atributo 'age'
val ageFreqTable = adultDF.groupBy("age").count().orderBy("age")
println("Tabla de frecuencia por age:")
ageFreqTable.show()



// Calcular el porcentaje de hombres y mujeres segun income
val genderIncomePercentage = adultDF.groupBy("sex", "income")
  .agg(count("income").alias("Count"))
  .withColumn("Total", sum("Count").over())
  .withColumn("Percentage", col("Count") / col("Total") * 100)
  .drop("Count", "Total")
  .orderBy("sex", "income")
  .show()


// Calcular el porcentaje de personas sobre el total según la raza, el sexo y su income
val raceGenderIncomePercentage = adultDF.groupBy("race", "sex", "income")
  .agg(count("income").alias("Count"))
  .withColumn("Total", sum("Count").over())
  .withColumn("Percentage", col("Count") / col("Total") * 100)
  .drop("Count", "Total")
  .orderBy("race", "sex", "income").show()



// Construir una tabla de frecuencia por la edad por intervalos de 5 años
val ageFrequencyTable = adultDF.select("age")
  .withColumn("AgeGroup", floor(col("age") / 5) * 5)
  .groupBy("AgeGroup")
  .agg(count("AgeGroup").alias("Frequency"))
  .orderBy("AgeGroup").show()


// Encontrar la native_country donde la media de edad es la más alta
val nativeCountryWithHighestAvgAge = adultDF.groupBy("native_country")
  .agg(avg("age").alias("AvgAge"))
  .orderBy(desc("AvgAge")).show()


// Calcular el porcentaje de personas con <50K y age<30
val hihgIncomeYoungPercentage = adultDF.filter("age<30")
  .groupBy()
  .agg((sum(when(col("income") === 1, 1).otherwise(0)) / count("*") * 100).alias("Percentage")).show()


// Calcular la fnlwgt media por race y sex de embarque
val avgRepByRaceSex= adultDF.groupBy("race", "sex")
  .agg(avg("fnlwgt").alias("AvgFnlwgt"))
  .orderBy("race", "sex").show()


// Crear una matriz de correlación entre el sexo, race y la marital_status
val correlationMatrix = adultDF.stat.crosstab("sex", "race")
  .join(adultDF.stat.crosstab("sex", "marital_status"))
  .drop("sex_race", "sex_marital_status").show()



// distribución de pasajeros según la edad y la native_country por intervalos
val ageClassDistribution = adultDF.select("age", "native_country")
  .na.drop(Seq("age"))
  .withColumn("AgeGroup", floor(col("age") / 10) * 10)
  .groupBy("AgeGroup", "native_country")
  .count()
  .orderBy("AgeGroup", "native_country").show()


// Encontrar el marital_status en la que más hombres >50K de todos los hombres que tienen relationship Not-in-family
val maritalWithMostMen = adultDF.filter(col("sex") === "male" && col("relationship") === "Not-in-family")
  .groupBy("marital_status")
  .agg(sum(when(col("income") === 1, 1).otherwise(0)).alias("HighMen"))
  .orderBy(desc("HighMen"))
  .select("marital_status")
  .show()



// Agrupar por sexo, marital_status y native_country
val groupedByAttributes = adultDF.groupBy("sex", "marital_status", "native_country").count()
groupedByAttributes.show()








// zip combina dos colecciones en una sola colección de pares
val ages = adultDF.select("age").as[Int].collect().toList
val incomes = adultDF.select("income").as[String].collect().toList

val ageIncomePairs = ages.zip(incomes)
ageIncomePairs.take(5).foreach(println)

// map se usa para transformar una colección aplicando una función a cada elemento. 
val retirementAges = ages.map(age => 65 - age)
retirementAges.take(5).foreach(println)

// join combina dos DataFrames basados en una clave común
val additionalData = adultDF.select("education", "income").withColumnRenamed("education", "edu")
val joinedData = adultDF.join(additionalData, adultDF("education") === additionalData("edu"), "inner")
joinedData.show(5)

// El operador :: se usa para construir listas en Scala
val ageList = List(30, 40, 50)
val newAgeList = 25 :: ageList
println(newAgeList)

// iterar sobre colecciones
val firstFiveAges = ages.take(5)
for (age <- firstFiveAges) {
  println(age)
}


// lista de ingresos en Double y calculamos el ingreso medio
val incomeStrings = adultDF.select("capital_gain").as[String].collect().toList
val incomeDoubles = incomeStrings.map(_.toDouble)

val totalIncome = incomeDoubles.sum
val averageIncome = totalIncome / incomeDoubles.size
println(s"Average Income: $averageIncome")


// Umbral de perdidas (usamos 3000 como ejemplo)
val lossThreshold = 3000

// Añadir columna 'income_level' basada en el umbral
val dataWithLossLevel = adultDF.withColumn("income_level", 
  when(col("capital_loss") > lit(lossThreshold), lit("high"))
  .otherwise(lit("low"))
)

dataWithLossLevel.show()


















//Transformacion atributos------------------------------------------------------------

// Nombres de las columnas
println("Nombres de las columnas:")
adultDF.columns.foreach(println)

// Crear una lista de columnas categóricas a indexar
val categoricalColumns = Array("sex","education", "race", "workclass","marital_status","occupation","relationship","native_country")
val indexers = categoricalColumns.map { colName =>
    new StringIndexer()
    .setInputCol(colName)
    .setOutputCol(s"${colName}_indexed")
    .setHandleInvalid("keep")
}




/*
// Correlación entre el sex y race
val correlationMatrix = adultIndexed.stat.corr("sex_indexed", "race_indexed")  
// Correlación entre sex y education
val correlationMatrix2 = adultIndexed.stat.corr("sex_indexed", "education_indexed")  
// Correlación entre education y race
val correlationMatrix3 = adultIndexed.stat.corr("education_indexed", "race_indexed")  

println(s"Correlación entre sex y race: $correlationMatrix")
println(s"Correlación entre sex y education: $correlationMatrix2")
println(s"Correlación entre education y race: $correlationMatrix3")


// Crear un DataFrame para mostrar las correlaciones en una tabla
val correlationDF = Seq(
  ("Sex-Race", correlationMatrix),
  ("Sex-Education", correlationMatrix2),
  ("Education-Race", correlationMatrix3)
).toDF("Attribute Pair", "Correlation")

correlationDF.show()
*/



























// Preparar los datos para los modelos---------------------------------------------------

// Dividir los datos 
val Array(trainData, validationData, testData) = adultDF.randomSplit(Array(0.6, 0.2, 0.2), seed = 1234)


// VectorAssembler para combinar las características en un solo vector
val assembler = new VectorAssembler()
    .setInputCols(Array("age", "education_indexed","workclass_indexed", "race_indexed", "native_country_indexed", "occupation_indexed", "relationship_indexed", "sex_indexed", "marital_status_indexed", "capital_gain", "capital_loss",
    "hours_per_week"))
    .setOutputCol("features")






























//Decission tree------------------------------------------------------------------------


// Definir el clasificador 
val modelDT = new DecisionTreeClassifier()
  .setLabelCol("income")
  .setFeaturesCol("features")

// Crear una Pipeline para encadenar las transformaciones y el modelo
val pipelineDT = new Pipeline().setStages(indexers ++ Array(assembler, modelDT))


// métrica de evaluación
// Para metricas como accuracy, f1, weightedPrecision, y weightedRecall y si hay mas de 2 clases posibles
val evaluatorDT = new MulticlassClassificationEvaluator()
  .setLabelCol("income")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

// cuadrícula de parámetros para la optimización
/*val paramGridDT = new ParamGridBuilder()
  .addGrid(modelDT.maxDepth, Array(3, 7))
  .addGrid(modelDT.maxBins, Array(24, 40))
  .build()*/
val paramGridDT = new ParamGridBuilder()
  .addGrid(modelDT.maxDepth, Array( 7))
  .build()

// validador cruzado
val crossValidatorDT = new CrossValidator()
  .setEstimator(pipelineDT)
  .setEvaluator(evaluatorDT)
  .setEstimatorParamMaps(paramGridDT.build())
  .setNumFolds(2)

// Entrenar el modelo
val cvModelDT = crossValidatorDT.fit(trainData)

// Evaluar el modelo en el conjunto de validación
val validationPredictionsDT = cvModelDT.transform(validationData)
val validationAccuracyDT = evaluatorDT.evaluate(validationPredictionsDT)
println(s"Exactitud en el conjunto de validación = $validationAccuracyDT")

// Hacer predicciones en el conjunto de prueba
val testPredictionsDT = cvModelDT.transform(testData)
val testAccuracyDT = evaluatorDT.evaluate(testPredictionsDT)
println(s"Exactitud en el conjunto de test = $testAccuracyDT")


















//Regresion logistica-----------------------------------------------------------------


// Definir el modelo de regresión logística
val modelLG = new LogisticRegression()
  .setLabelCol("income")
  .setFeaturesCol("features")


// Crear una Pipeline para encadenar las transformaciones y el modelo
val pipelineLG = new Pipeline().setStages(indexers ++ Array(assembler, modelLG))


// Configurar el ParamGridBuilder para ajustar los hiperparámetros
/*val paramGridLG = new ParamGridBuilder()
  .addGrid(modelLG.regParam, Array(0.01, 0.3))
  .addGrid(modelLG.elasticNetParam, Array(0.0, 1.0))
  .addGrid(modelLG.maxIter, Array(10, 100))
  .build()*/
val paramGridLG = new ParamGridBuilder()
  .addGrid(modelLG.regParam, Array(0.01))
  .build()


// Configurar el evaluador
// Para metricas como areaUnderROC y areaUnderPR y si solo hay dos clases posibles
val evaluatorLG_ROC = new BinaryClassificationEvaluator()
  .setLabelCol("income")
  .setMetricName("areaUnderROC")

val evaluatorLG_PR = new BinaryClassificationEvaluator()
  .setLabelCol("income")
  .setMetricName("areaUnderPR")


// Configurar el CrossValidator
val crossValidatorLG  = new CrossValidator()
  .setEstimator(pipelineLG)
  .setEvaluator(evaluatorLG_ROC)
  .setEstimatorParamMaps(paramGridLG.build())
  .setNumFolds(2) 

// Entrenar el modelo
val cvModelLG = crossValidatorLG.fit(trainData)

// Evaluar el modelo en el conjunto de validación
val validationPredictionsLG_ROC = cvModelLG.transform(validationData)
val validationAccuracyLG_ROC = evaluatorLG_ROC.evaluate(validationPredictionsLG_ROC)
println(s"Area Under ROC conjunto de validación = $validationAccuracyLG_ROC")

// Hacer predicciones en el conjunto de prueba
val testPredictionsLG_PR = cvModelLG.transform(testData)
val testAccuracyLG_PR = evaluatorLG_PR.evaluate(testPredictionsLG_PR)
println(s"Area Under ROC conjunto de test = $testAccuracyLG_PR")














//SVM---------------------------------------------------------------------------


// modelo SVM
val modelSVM = new LinearSVC()
  .setLabelCol("income")
  .setFeaturesCol("features")

// rejilla de parámetros
/*val paramGridSVM = new ParamGridBuilder()
    .addGrid(modelSVM.regParam, Array(0.01, 1.0))
    .addGrid(modelSVM.maxIter, Array(10, 100))
    .build()*/
val paramGridSVM = new ParamGridBuilder()
    .addGrid(modelSVM.regParam, Array(0.01))
    .build()

// Crear una Pipeline para encadenar las transformaciones y el modelo
val pipelineSVM  = new Pipeline().setStages(indexers ++ Array(assembler, modelSVM))



// Configurar el CrossValidator
val crossValidatorSVM = new CrossValidator()
  .setEstimator(pipelineSVM)
  .setEvaluator(new MulticlassClassificationEvaluator()
  .setLabelCol("income")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")) 
  .setEstimatorParamMaps(paramGridSVM.build())
  .setNumFolds(2) 


// Entrenar el modelo
val cvModelSVM  = crossValidatorSVM.fit(trainData)

// métrica de evaluación
val evaluatorSVM= new MulticlassClassificationEvaluator()
  .setLabelCol("income")
  .setPredictionCol("prediction")


// Evaluar el modelo en el conjunto de validación
val validationPredictionsSVM  = cvModelSVM.transform(validationData)
// Hacer predicciones en el conjunto de prueba
val testPredictionsSVM  = cvModelSVM.transform(testData)


// Calcular Accuracy
evaluatorSVM.setMetricName("accuracy")
val validationAccuracySVM = evaluatorSVM.evaluate(validationPredictionsSVM)
println(s"Exactitud validation set = $validationAccuracySVM")

val testAccuracySVM = evaluatorSVM.evaluate(testPredictionsSVM)
println(s"Exactitud test set = $testAccuracySVM")



// Calcular Precision
evaluatorSVM.setMetricName("weightedPrecision")
val validationWeightedPrecisionSVM = evaluatorSVM.evaluate(validationPredictionsSVM)
println(s"weightedPrecision validation set = $validationWeightedPrecisionSVM")

val testWeightedPrecisionSVM = evaluatorSVM.evaluate(testPredictionsSVM)
println(s"weightedPrecision test set = $testWeightedPrecisionSVM")


// Calcular Recall
evaluatorSVM.setMetricName("weightedRecall")
val validationWeightedRecallSVM = evaluatorSVM.evaluate(validationPredictionsSVM)
println(s"weightedRecall validation set = $validationWeightedRecallSVM")

val testWeightedRecallSVM = evaluatorSVM.evaluate(testPredictionsSVM)
println(s"weightedRecall test set = $testWeightedRecallSVM")


// Calcular F1-Score
evaluatorSVM.setMetricName("f1")
val validationF1SVM = evaluatorSVM.evaluate(validationPredictionsSVM)
println(s"f1 validation set = $validationF1SVM")

val testF1SVM = evaluatorSVM.evaluate(testPredictionsSVM)
println(s"f1 test set = $testF1SVM")




























//GBT----------------------------------------------------------------------------------------


// Definir el modelo GBT
val modelGBT = new GBTClassifier()
  .setLabelCol("income")
  .setFeaturesCol("features")

// Crear una Pipeline para encadenar las transformaciones y el modelo
val pipelineGBT = new Pipeline().setStages(indexers ++ Array(assembler, modelGBT))

// Construir la rejilla de parámetros
/*val paramGrid = new ParamGridBuilder()
  .addGrid(gbt.maxDepth, Array(3, 5))
  .addGrid(gbt.maxBins, Array(24, 32))
  .addGrid(gbt.maxIter, Array(10, 20))
  .build()*/
val paramGridGBT = new ParamGridBuilder()
  .addGrid(modelGBT.maxDepth, Array(3))
  .build()

// Configurar la evaluación
val evaluatorGBT = new MulticlassClassificationEvaluator()
    .setLabelCol("income")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")


// Configurar el CrossValidator
val crossValidatorGBT  = new CrossValidator()
  .setEstimator(pipelineGBT)
  .setEvaluator(evaluatorGBT)
  .setEstimatorParamMaps(paramGridGBT.build())
  .setNumFolds(2) 

// Entrenar el modelo
val cvModelGBT = crossValidatorGBT.fit(trainData)

// Evaluar el modelo en el conjunto de validación
val validationPredictionsGBT = cvModelGBT.transform(validationData)
val validationAccuracyGBT= evaluatorGBT.evaluate(validationPredictionsGBT)
println(s"Exactitud validation set = $validationAccuracyGBT")

// Hacer predicciones en el conjunto de prueba
val testPredictionsGBT = cvModelGBT.transform(testData)
val testAccuracyGBT= evaluatorGBT.evaluate(testPredictionsGBT)
println(s"Exactitud test set = $testAccuracyGBT")















//Random forest-------------------------------------------------------------------


val modelRF = new RandomForestClassifier()
  .setLabelCol("income")
  .setFeaturesCol("features")


// Crear una Pipeline para encadenar las transformaciones y el modelo
val pipelineRF = new Pipeline().setStages(indexers ++ Array(assembler, modelRF))

// Configurar el ParamGridBuilder para ajustar los hiperparámetros
/*val paramGridRF = new ParamGridBuilder()
  .addGrid(modelRF.numTrees, Array(10, 20, 30))
  .addGrid(modelRF.maxDepth, Array(5, 10, 15))
  .build()*/
val paramGridRF = new ParamGridBuilder()
  .addGrid(modelRF.numTrees, Array(20))
  .build()

// Configurar el CrossValidator
val cvRF = new CrossValidator()
  .setEstimator(pipelineRF)
  .setEvaluator(new BinaryClassificationEvaluator()
    .setLabelCol("income")
    .setRawPredictionCol("rawPrediction"))
  .setEstimatorParamMaps(paramGridRF.build())
  .setNumFolds(3) 



// Entrenar el modelo usando CrossValidator
val cvModelRF = cvRF.fit(trainData)

// Realizar predicciones en el conjunto de validación
val valPredictionsRF = cvModelRF.transform(validationData)

// Evaluar el modelo en el conjunto de validación
val evaluatorRF = new BinaryClassificationEvaluator()
  .setLabelCol("income")
  .setRawPredictionCol("rawPrediction")

val aucROC_RF = evaluatorRF.setMetricName("areaUnderROC").evaluate(valPredictionsRF)
val aucPR_RF = evaluatorRF.setMetricName("areaUnderPR").evaluate(valPredictionsRF)

println(s"Validation Set - Area Under ROC: $aucROC_RF")
println(s"Validation Set - Area Under PR: $aucPR_RF")

// Realizar predicciones en el conjunto de prueba
val testPredictionsRF = cvModelRF.transform(testData)

// Evaluar el modelo en el conjunto de prueba
val testAucROC_RF = evaluatorRF.setMetricName("areaUnderROC").evaluate(testPredictionsRF)
val testAucPR_RF = evaluatorRF.setMetricName("areaUnderPR").evaluate(testPredictionsRF)

println(s"Test Set - Area Under ROC: $testAucROC_RF")
println(s"Test Set - Area Under PR: $testAucPR_RF")

















//Naive bayes---------------------------------------------------------------------


val modelNB = new NaiveBayes()
  .setLabelCol("income")
  .setFeaturesCol("features")


// Crear una Pipeline para encadenar las transformaciones y el modelo
val pipelineNB = new Pipeline().setStages(indexers ++ Array(assembler, modelNB))

// Configurar el ParamGridBuilder para ajustar los hiperparámetros
val paramGridNB = new ParamGridBuilder()
  .addGrid(modelNB.smoothing, Array(0.5, 1.0, 1.5))
  .build()

// Configurar el CrossValidator
val cvNB = new CrossValidator()
  .setEstimator(pipelineNB)
  .setEvaluator(new BinaryClassificationEvaluator()
    .setLabelCol("income")
    .setRawPredictionCol("rawPrediction"))
  .setEstimatorParamMaps(paramGridNB.build())
  .setNumFolds(3) 



// Entrenar el modelo usando CrossValidator
val cvModelNB = cvNB.fit(trainData)

// Realizar predicciones en el conjunto de validación
val valPredictionsNB = cvModelNB.transform(validationData)

// Evaluar el modelo en el conjunto de validación
val evaluatorNB = new BinaryClassificationEvaluator()
  .setLabelCol("income")
  .setRawPredictionCol("rawPrediction")

val aucROC_NB = evaluatorNB.setMetricName("areaUnderROC").evaluate(valPredictionsNB)
val aucPR_NB = evaluatorNB.setMetricName("areaUnderPR").evaluate(valPredictionsNB)

println(s"Validation Set - Area Under ROC: $aucROC_NB")
println(s"Validation Set - Area Under PR: $aucPR_NB")

// Realizar predicciones en el conjunto de prueba
val testPredictionsNB = cvModelNB.transform(testData)

// Evaluar el modelo en el conjunto de prueba
val testAucROC_NB = evaluatorNB.setMetricName("areaUnderROC").evaluate(testPredictionsNB)
val testAucPR_NB = evaluatorNB.setMetricName("areaUnderPR").evaluate(testPredictionsNB)

println(s"Test Set - Area Under ROC: $testAucROC_NB")
println(s"Test Set - Area Under PR: $testAucPR_NB")








































//RDD-------------------------------------------------------------------------------------

// Crear un RDD 
val rdd = spark.sparkContext.parallelize(Seq(
  (1, "apple", 5),
  (2, "banana", 15),
  (3, "orange", 10),
  (4, "grape", 20),
  (5, "orange", 20)
))


// Obtener el número de filas 
val numRows = rdd.count()
println(s"Number of rows in the RDD: $numRows")



// Imprimir
println("All elements in the RDD:")
rdd.collect().foreach(println)



// columna que convierta a mayúsculas otra columna
val uppercaseRDD = rdd.map { case (id, fruit, value) => (id, fruit.toUpperCase, value) }


// Filtrar las filas superior a 10
val filteredRDD = uppercaseRDD.filter { case (_, _, value) => value > 10 }


// Sumar los valores reduciendo por columna
val sumByFruitRDD = filteredRDD.map { case (_, fruit, value) => (fruit, value) }
  .reduceByKey(_ + _)

// Imprimir el resultado 
println("Sum of values by fruit:")
sumByFruitRDD.collect().foreach(println)





























//SQL------------------------------------------------------------------------------------

// Crear un DataFrame
val df = spark.createDataFrame(Seq(
  (1, "apple", 5),
  (2, "banana", 15),
  (3, "orange", 10),
  (4, "grape", 20)
)).toDF("id", "fruit", "value")

// Registrar el DataFrame como tabla temporal
df.createOrReplaceTempView("fruits")


// consulta SQL con condición
val result = spark.sql("SELECT * FROM fruits WHERE value > 10")

// Mostrar el resultado
result.show()


// consulta SQL para calcular el valor medio con agrupación
val avgValueByFruit = spark.sql("SELECT fruit, AVG(value) AS avg_value FROM fruits GROUP BY fruit")

// Mostrar el resultado
avgValueByFruit.show()


// consulta SQL con la cláusula IN
val result = spark.sql("SELECT * FROM fruits WHERE fruit IN ('apple', 'banana')")

// Mostrar el resultado
result.show()


// Importar las funciones de ventana
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

// Definir la ventana
val windowSpec = Window.partitionBy("fruit").orderBy("value")

// consulta SQL utilizando una ventana
val result = spark.sql("SELECT *, ROW_NUMBER() OVER (PARTITION BY fruit ORDER BY value) AS row_number FROM fruits")

// Mostrar el resultado
result.show()



























// Regresion logistica con train validation split-------------------------------------

// Convertir variables categóricas a numéricas
val genderIndexer = new StringIndexer()
    .setInputCol("sex")
    .setOutputCol("sexIndexNew")
    .fit(adultDF)
val raceIndexer = new StringIndexer()
    .setInputCol("race")
    .setOutputCol("raceIndexNew")
    .fit(adultDF)

val assemblerNew = new VectorAssembler()
    .setInputCols(Array("age", "sexIndexNew","raceIndexNew","hours_per_week"))
    .setOutputCol("features")



//  Convertir variables categóricas a numéricas
val genderIndexerModel: StringIndexerModel = genderIndexer.fit(adultDF)
val raceIndexerModel: StringIndexerModel = raceIndexer.fit(adultDF)

val transformedDF = assemblerNew.transform(raceIndexerModel.transform(genderIndexerModel.transform(adultDF)))
val labeledDF = transformedDF.withColumnRenamed("income", "label")

// Dividir el dataset en conjuntos
val Array(trainDataNew, valDataNew, testDataNew) = labeledDF.randomSplit(Array(0.6, 0.2, 0.2), seed = 123)

// regresión logística
val lrNew = new LogisticRegression()

// rejilla de parámetros
val paramGridLRNew = new ParamGridBuilder()
    .addGrid(lrNew.regParam, Array(0.01, 0.1, 0.3))
    .addGrid(lrNew.elasticNetParam, Array(0.0, 0.5, 1.0))
    .addGrid(lrNew.maxIter, Array(10, 100))
    .build()

// evaluador
val evaluatorLRNew = new BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC")

// validación cruzada para la selección del modelo
val trainValidationSplitLRNew = new TrainValidationSplit()
    .setEstimator(lrNew)
    .setEvaluator(evaluatorLRNew)
    .setEstimatorParamMaps(paramGridLRNew.build())
    .setTrainRatio(0.8)

// Ajustar el modelo 
val modelLRNew = trainValidationSplitLRNew.fit(trainDataNew)

// Evaluar el modelo en el conjunto de validación
val resultsLRNew = modelLRNew.transform(valDataNew)
val areaUnderROC_LRNew = evaluatorLRNew.evaluate(resultsLRNew)

println(s"Area under ROC on validation set = $areaUnderROC_LRNew")



// Detener la sesión de Spark
spark.stop()
