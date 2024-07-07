/*

Census Income (KDD) Data Set
Es una versión extendida del conocido Adult Income Dataset, utilizada principalmente en la conferencia KDD (Knowledge Discovery and Data Mining). Este conjunto de datos se usa para tareas de clasificación, como predecir si una persona gana más de $50,000 al año. En total, este dataset consta de 41 atributos pero a continuación se describen los atributos más importantes de este dataset:

-age (edad): La edad de la persona en años.
-workclass (clase de trabajo): Tipo de empleador o clase de trabajo, por ejemplo, Privado, Gobierno local.
-fnlwgt (peso final): Peso de la muestra que indica el número de personas representadas por e la persona en el censo.
-education (educación): Nivel educativo alcanzado por e la persona, por ejemplo, Bachillerato, Doctorado.
-education_num (número de educación): Años de educación completados.
-marital_status (estado civil): Estado civil de la persona, por ejemplo, Casado, Nunca casado.
-occupation (ocupación): Ocupación de la persona, por ejemplo, Ejecutivo/gerencial, Técnico/soporte.
-relationship (relación): Relación de la persona con el cabeza de familia, por ejemplo, Esposo, Hijo propio.
-race (raza): Raza de la persona, por ejemplo, Blanco, Negro, Asiático.
-sex (sexo): Sexo de la persona (Masculino o Femenino).
-capital_gain (ganancia de capital): Ganancias obtenidas por inversiones.
-capital_loss (pérdida de capital): Pérdidas incurridas por inversiones.
-hours_per_week (horas por semana): Número de horas trabajadas por semana.
-native_country (país de origen): País de origen de la persona, por ejemplo, Estados Unidos, Canadá.
-income (ingresos): Clase de ingresos de la persona, que puede ser ">50K" para ingresos superiores a $50,000 o "<=50K" para ingresos inferiores o iguales a $50,000.
*/



import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.{DecisionTreeClassifier,RandomForestClassifier, GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderModel, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, RegressionMetrics, BinaryClassificationMetrics}
import org.apache.spark.sql.{DataFrame, SparkSession,Row}
import org.apache.spark.sql.types.{IntegerType, StringType, DoubleType, StructField, StructType}
import org.apache.spark.ml.linalg.Vector


val PATH="/home/usuario/bib-scala/"
val FILE_CENSUS="census-income.data"


















// Cargar datos-------------------------------------------------------------------------
/* Crear un esquema para leer los datos */
val censusSchema = StructType(Array(
  StructField("age", IntegerType, false),
  StructField("class_of_worker", StringType, true),
  StructField("industry_code", StringType, true),
  StructField("occupation_code", StringType, true),
  StructField("education", StringType, true),
  StructField("wage_per_hour", IntegerType, false),
  StructField("enrolled_in_edu_last_wk", StringType, true),  
  StructField("marital_status", StringType, true),
  StructField("major_industry_code", StringType, true),
  StructField("major_occupation_code", StringType, true),
  StructField("race", StringType, true),
  StructField("hispanic_Origin", StringType, true),
  StructField("sex", StringType, true),
  StructField("member_of_labor_union", StringType, true),
  StructField("reason_for_unemployment", StringType, true),
  StructField("full_or_part_time_employment_status", StringType, true),
  StructField("capital_gains", IntegerType, false),
  StructField("capital_losses", IntegerType, false),
  StructField("dividends_from_stocks", IntegerType, false),
  StructField("tax_filer_status", StringType, true),
  StructField("region_of_previous_residence", StringType, true),
  StructField("state_of_previous_residence", StringType, true),
  StructField("detailed_household_and_family_status", StringType, true),
  StructField("detailed_household_summary_in_house_instance_weight", StringType, false),
  StructField("total_person_earnings", DoubleType, false),  
  StructField("migration_code_change_in_msa", StringType, true),
  StructField("migration_code_change_in_reg", StringType, true),
  StructField("migration_code_move_within_reg", StringType, true),
  StructField("live_in_this_house_one_year_ago", StringType, true),
  StructField("migration_prev_res_in_sunbelt", StringType, true),
  StructField("num_persons_worked_for_employer", IntegerType, false),
  StructField("family_members_under_18", StringType, true),  
  StructField("country_of_birth_father", StringType, true),
  StructField("country_of_birth_mother", StringType, true),
  StructField("country_of_birth_self", StringType, true),
  StructField("citizenship", StringType, true),
  StructField("own_business_or_self_employed", IntegerType, true),
  StructField("fill_inc_questionnaire_for_veterans_ad", StringType, true),
  StructField("veterans_benefits", StringType, false),
  StructField("weeks_worked_in_year", IntegerType, false),
  StructField("year", IntegerType, false),
  StructField("income", StringType, false)
));


var census_df = spark.read.format("csv").
option("delimiter", ",").option("ignoreLeadingWhiteSpace","true").
schema(censusSchema).load(PATH + FILE_CENSUS)



// Listas con los tipos de atributos
/*val listaAtributosNumericos = List("age","wage_per_hour","capital_gains","capital_losses","dividends_from_stocks","total_person_earnings","num_persons_worked_for_employer","own_business_or_self_employed","weeks_worked_in_year","year")
val listaAtributosCategoricos = List("industry_code","occupation_code","class_of_worker","education","enrolled_in_edu_last_wk","marital_status","major_industry_code","major_occupation_code","member_of_labor_union","race","sex","full_or_part_time_employment_status","reason_for_unemployment","hispanic_Origin","tax_filer_status","region_of_previous_residence","state_of_previous_residence","detailed_household_and_family_status","detailed_household_summary_in_house_instance_weight","migration_code_change_in_msa","migration_code_change_in_reg","migration_code_move_within_reg","live_in_this_house_one_year_ago","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father","country_of_birth_mother","country_of_birth_self","citizenship","fill_inc_questionnaire_for_veterans_ad","veterans_benefits")*/
val listaAtributosNumericos = List("age","wage_per_hour","capital_gains","capital_losses")
val listaAtributosCategoricos = List("industry_code","occupation_code","class_of_worker","education")






















// Tratamiento outliers------------------------------------------------------------------
// Reemplazo de outliers por límite
for (nombre_columna <- listaAtributosNumericos) {
    val percentiles = census_df.stat.approxQuantile(nombre_columna, Array(0.05, 0.95), 0.01)
    val limiteInferior = percentiles(0)
    val limiteSuperior = percentiles(1)
    census_df = census_df.withColumn(nombre_columna, when(col(nombre_columna) < limiteInferior, limiteInferior).otherwise(when(col(nombre_columna) > limiteSuperior, limiteSuperior).otherwise(col(nombre_columna))))
}













// Tratamiento nulos---------------------------------------------------------------------
// Reemplazo de nulos por la moda
for (nombre_columna <- listaAtributosCategoricos) {
  val numero_cada_uno=census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
  val moda_atributo = numero_cada_uno.first().getString(0) 
  census_df = census_df.withColumn(nombre_columna, when(col(nombre_columna) === "?", moda_atributo).otherwise(col(nombre_columna)))
  
}
















// Limpieza y transformación---------------------------------------------------------
:load TransformDataframeCensus.scala
:load CleanDataframeCensus.scala

import TransformDataframeCensus._
import CleanDataframeCensus._
val census_df_limpio=cleanDataframe(census_df)
val trainCensusDFProcesado = transformDataFrame(census_df_limpio)






















// Gradient Boosted Tree------------------------------------------------------------------
val gbt = new GBTClassifier().setLabelCol("label").setFeaturesCol("features")
val paramGrid = new ParamGridBuilder().addGrid(gbt.maxDepth, Array(3)).addGrid(gbt.maxIter, Array(20)).addGrid(gbt.maxBins, Array(50)).build()
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val pipeline = new Pipeline().setStages(Array(gbt))

val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)  
val cvModel = cv.fit(trainCensusDFProcesado)

val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
val bestGBTModel = bestPipelineModel.stages(0).asInstanceOf[GBTClassificationModel]
println(s"Best max depth: ${bestGBTModel.getMaxDepth}")
println(s"Best max iterations: ${bestGBTModel.getMaxIter}")
println(s"Best max bins: ${bestGBTModel.getMaxBins}")



// Utilizar mejores parámetros
val GBT = new GBTClassifier().setFeaturesCol("features").setLabelCol("label").setMaxIter(bestGBTModel.getMaxIter).
 setMaxDepth(bestGBTModel.getMaxDepth).
 setMaxBins(bestGBTModel.getMaxBins).
 setMinInstancesPerNode(1).
 setMinInfoGain(0.0).
 setCacheNodeIds(false).
 setCheckpointInterval(10)

val GBTModel_D =GBT.fit(trainCensusDFProcesado)


// Guardar modelo
GBTModel_D.write.overwrite().save(PATH + "modeloGBT")

























// Random forest---------------------------------------------------------------------------
val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features")
val paramGrid = new ParamGridBuilder().addGrid(rf.maxDepth, Array(10)).addGrid(rf.numTrees, Array(140)).addGrid(rf.maxBins, Array(150)).build()
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val cv = new CrossValidator().setEstimator(rf).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2) 
val cvModel = cv.fit(trainCensusDFProcesado)
import org.apache.spark.ml.classification.RandomForestClassificationModel
val bestModel = cvModel.bestModel.asInstanceOf[RandomForestClassificationModel]

println(s"Number of Trees: ${bestModel.getNumTrees}")
println(s"Max Depth: ${bestModel.getMaxDepth}")
println(s"Max bins: ${bestModel.getMaxBins}")


// Utilizar mejores parámetros
val RF = new RandomForestClassifier().setFeaturesCol("features").
 setLabelCol("label").
 setNumTrees(bestModel.getNumTrees).
 setMaxDepth(bestModel.getMaxDepth).
 setMaxBins(bestModel.getMaxBins).
 setMinInstancesPerNode(1).
 setMinInfoGain(0.0).
 setCacheNodeIds(false).
 setCheckpointInterval(10)

val RFModel_D =RF.fit(trainCensusDFProcesado)
RFModel_D.toDebugString


// Guardar modelo
RFModel_D.write.overwrite().save(PATH + "modeloRF")


























// Carga conjunto de test y modelo--------------------------------------------------
val loadedGBTcensusModel = GBTClassificationModel.load(PATH + "modeloGBT")
val loadedRFcensusModel = RandomForestClassificationModel.load(PATH + "modeloRF")
val FILE_CENSUS_TEST="census-income.test"


val census_df_test = spark.read.format("csv").
option("delimiter", ",").option("ignoreLeadingWhiteSpace","true").
schema(censusSchema).load(PATH + FILE_CENSUS_TEST)





















// Limpieza y transformación---------------------------------------------------------------------
val census_df_limpio=cleanDataframe(census_df_test)
val testCensusDF = transformDataFrame(census_df_limpio)

































// Evaluar metricas---------------------------------------------------------------
val predictionsAndLabelsDF_GBT = loadedGBTcensusModel.transform(testCensusDF).select("prediction", "label","rawPrediction", "probability")
predictionsAndLabelsDF_GBT.show()

val rm_GBT = new RegressionMetrics(predictionsAndLabelsDF_GBT.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))
println("Test metrics:")
println("Test Explained Variance: ")
println(rm_GBT.explainedVariance) 
println("R² Coefficient")
println(rm_GBT.r2)
//println("Test MSE: ")
//println(rm_GBT.meanSquaredError)
//println("Test RMSE: ")
//println(rm_GBT.rootMeanSquaredError)


// Métricas
val predictionsGBT = loadedGBTcensusModel.transform(testCensusDF).select("prediction").rdd.map(_.getDouble(0))
val labelsGBT = loadedGBTcensusModel.transform(testCensusDF).select("label").rdd.map(_.getDouble(0))
val metricsGBT = new MulticlassMetrics(predictionsGBT.zip(labelsGBT))

println("Confusion matrix:")
println(metricsGBT.confusionMatrix)

val accuracyGBT = metricsGBT.accuracy
println("Estadísticas resumen")
println(f"Accuracy = $accuracyGBT%1.4f")

val labelsGBT = metricsGBT.labels
labelsGBT.foreach {l => val pl = metricsGBT.precision(l) 
        println(f"PrecisionByLabel($l) = $pl%1.4f")}

labelsGBT.foreach {l => val fpl = metricsGBT.falsePositiveRate(l)
        println(f"falsePositiveRate($l) = $fpl%1.4f")}

labelsGBT.foreach {l => val fpl = metricsGBT.truePositiveRate(l)
        println(f"truePositiveRate($l) = $fpl%1.4f")}


// Curva ROC---------------------------------------------
val probabilitiesAndLabelsRDD_GBT = predictionsAndLabelsDF_GBT.select("label", "probability").rdd.map{row => (row.getAs[Vector](1).toArray, row.getDouble(0))}.map{r => ( r._1(1), r._2)}

val MLlib_binarymetricsGBT = new BinaryClassificationMetrics(probabilitiesAndLabelsRDD_GBT,15)

val MLlib_auROC_GBT = MLlib_binarymetricsGBT.areaUnderROC
println(f"%nAUC de la curva ROC para la clase income")
println(f"con MLlib, métrica binaria, probabilitiesAndLAbelsRDD, 15 bins: $MLlib_auROC_GBT%1.4f%n")

val MLlib_auPR_GBT = MLlib_binarymetricsGBT.areaUnderPR
println(f"%nAUC de la curva PR para la clase income")
println(f"con MLlib, métrica binaria, probabilitiesAndLAbelsRDD, 15 bins: $MLlib_auPR_GBT%1.4f%n")

val MLlib_curvaROC_GBT =MLlib_binarymetricsGBT.roc
println("Puntos para construir curva ROC con MLlib, probabilitiesAndLabelsRDD, 15 bins:")
MLlib_curvaROC_GBT.take(17).foreach(x => println(x))




















// Evaluar métricas---------------------------------------------------
val predictionsAndLabelsDF_RF = loadedRFcensusModel.transform(testCensusDF).select("prediction", "label","rawPrediction", "probability")

val predictionsRF = loadedRFcensusModel.transform(testCensusDF).select("prediction").rdd.map(_.getDouble(0))
val labelsRF = loadedRFcensusModel.transform(testCensusDF).select("label").rdd.map(_.getDouble(0))


// Métricas
val metricsRF = new MulticlassMetrics(predictionsRF.zip(labelsRF))

println("Confusion matrix:")
println(metricsRF.confusionMatrix)

val accuracyRF = metricsRF.accuracy
println("Estadísticas resumen")
println(f"Accuracy = $accuracyRF%1.4f")

val labelsRF = metricsRF.labels
labelsRF.foreach {l => val pl = metricsRF.precision(l) 
        println(f"PrecisionByLabel($l) = $pl%1.4f")}

labelsRF.foreach {l => val fpl = metricsRF.falsePositiveRate(l)
        println(f"falsePositiveRate($l) = $fpl%1.4f")}

labelsRF.foreach {l => val fpl = metricsRF.truePositiveRate(l)
        println(f"truePositiveRate($l) = $fpl%1.4f")}



// Curva roc
val probabilitiesAndLabelsRDD_RF = predictionsAndLabelsDF_RF.select("label", "probability").rdd.map{row => (row.getAs[Vector](1).toArray, row.getDouble(0))}.map{r => ( r._1(1), r._2)}

val MLlib_binarymetricsRF = new BinaryClassificationMetrics(probabilitiesAndLabelsRDD_RF,15)

val MLlib_auROC_RF = MLlib_binarymetricsRF.areaUnderROC
println(f"%nAUC de la curva ROC para la clase income")
println(f"con MLlib, métrica binaria, probabilitiesAndLAbelsRDD, 15 bins: $MLlib_auROC_RF%1.4f%n")

val MLlib_auPR_RF = MLlib_binarymetricsRF.areaUnderPR
println(f"%nAUC de la curva PR para la clase income")
println(f"con MLlib, métrica binaria, probabilitiesAndLAbelsRDD, 15 bins: $MLlib_auPR_RF%1.4f%n")

val MLlib_curvaROC_RF =MLlib_binarymetricsRF.roc
println("Puntos para construir curva ROC con MLlib, probabilitiesAndLabelsRDD, 15 bins:")
MLlib_curvaROC_RF.take(17).foreach(x => println(x))