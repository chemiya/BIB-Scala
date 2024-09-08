



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


val PATH="/home/usuario/Scala/Proyecto4/"
val FILE_CENSUS="census-income-reducido1000.data"




/*creamos un esquema para leer los datos */
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





//listas con los tipos de atributos
val listaAtributosNumericos = List("age","wage_per_hour","capital_gains","capital_losses","dividends_from_stocks","total_person_earnings","num_persons_worked_for_employer","own_business_or_self_employed","weeks_worked_in_year","year")
val listaAtributosCategoricos = List("industry_code","occupation_code","class_of_worker","education","enrolled_in_edu_last_wk","marital_status","major_industry_code","major_occupation_code","member_of_labor_union","race","sex","full_or_part_time_employment_status","reason_for_unemployment","hispanic_Origin","tax_filer_status","region_of_previous_residence","state_of_previous_residence","detailed_household_and_family_status","detailed_household_summary_in_house_instance_weight","migration_code_change_in_msa","migration_code_change_in_reg","migration_code_move_within_reg","live_in_this_house_one_year_ago","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father","country_of_birth_mother","country_of_birth_self","citizenship","fill_inc_questionnaire_for_veterans_ad","veterans_benefits")















//outliers
for (nombre_columna <- listaAtributosNumericos) {
    val percentiles = census_df.stat.approxQuantile(nombre_columna, Array(0.05, 0.95), 0.01)
    val limiteInferior = percentiles(0)
    val limiteSuperior = percentiles(1)
    census_df = census_df.withColumn(nombre_columna, when(col(nombre_columna) < limiteInferior, limiteInferior).otherwise(when(col(nombre_columna) > limiteSuperior, limiteSuperior).otherwise(col(nombre_columna))))
}














//nulos
for (nombre_columna <- listaAtributosCategoricos) {
  val numero_cada_uno=census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
  val moda_atributo = numero_cada_uno.first().getString(0) 
  census_df = census_df.withColumn(nombre_columna, when(col(nombre_columna) === "?", moda_atributo).otherwise(col(nombre_columna)))
  
}
















//cargamos dataset----------------------------

:load TransformDataframeV2.scala
:load CleanDataframe.scala


import TransformDataframeV2._
import CleanDataframe._
val census_df_limpio=cleanDataframe(census_df)
val trainCensusDFProcesado = transformDataFrame(census_df_limpio)






















//validacion cruzada y parametros---------------------------
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















//utilizamos mejores parametros----------------------
val GBT = new GBTClassifier().setFeaturesCol("features").setLabelCol("label").setMaxIter(bestGBTModel.getMaxIter).
 setMaxDepth(bestGBTModel.getMaxDepth).
 setMaxBins(bestGBTModel.getMaxBins).
 setMinInstancesPerNode(1).
 setMinInfoGain(0.0).
 setCacheNodeIds(false).
 setCheckpointInterval(10)

val GBTModel_D =GBT.fit(trainCensusDFProcesado)


GBTModel_D.write.overwrite().save(PATH + "modeloGBT")










//RF
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









//utilizamos mejores parametros------------------------------
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



RFModel_D.write.overwrite().save(PATH + "modeloRF")



