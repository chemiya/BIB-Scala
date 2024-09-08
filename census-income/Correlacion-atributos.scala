import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier,GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderModel, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{ Column}
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, RegressionMetrics, BinaryClassificationMetrics}
import org.apache.spark.sql.{DataFrame, SparkSession,Row}
import org.apache.spark.sql.types.{IntegerType, StringType, DoubleType, StructField, StructType}
import org.apache.spark.ml.feature.{QuantileDiscretizer, StringIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.DenseMatrix



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











//creacion dataframe
var census_df = spark.read.format("csv").
option("delimiter", ",").option("ignoreLeadingWhiteSpace","true").
schema(censusSchema).load(PATH + FILE_CENSUS)












//listas con los tipos de atributos
val listaAtributosNumericos = List("age","wage_per_hour","capital_gains","capital_losses","dividends_from_stocks","total_person_earnings","num_persons_worked_for_employer","own_business_or_self_employed","weeks_worked_in_year","year")
val listaAtributosCategoricos = List("industry_code","occupation_code","class_of_worker","education","enrolled_in_edu_last_wk","marital_status","major_industry_code","major_occupation_code","member_of_labor_union","race","sex","full_or_part_time_employment_status","reason_for_unemployment","hispanic_Origin","tax_filer_status","region_of_previous_residence","state_of_previous_residence","detailed_household_and_family_status","detailed_household_summary_in_house_instance_weight","migration_code_change_in_msa","migration_code_change_in_reg","migration_code_move_within_reg","live_in_this_house_one_year_ago","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father","country_of_birth_mother","country_of_birth_self","citizenship","fill_inc_questionnaire_for_veterans_ad","veterans_benefits")


















//correlaccion atributos numericos---------------------

for (i <- 0 until listaAtributosNumericos.length) {
  val columna_actual = listaAtributosNumericos(i)
    
    for (j <- (i + 1) until listaAtributosNumericos.length) {
        val siguiente_columna = listaAtributosNumericos(j)

        val assembler = new org.apache.spark.ml.feature.VectorAssembler().setInputCols(Array(columna_actual, siguiente_columna)).setOutputCol("features")
        val dfVector = assembler.transform(census_df)

        val correlacionMatrix = Correlation.corr(dfVector, "features").head().getAs[DenseMatrix](0)
        val valor = correlacionMatrix.apply(0, 1)


        println("Correlaccion entre "+columna_actual+" con "+siguiente_columna+" es: "+valor)

    }

}





























//correlaccion categoricos----------------
var correlaccion_categoricas=Array[String]()

def comprobarValoresMayores(df: DataFrame):Long= {
  val columnNames = df.columns
  //obtenemos maximo de cada columna
  val maxValues = columnNames.map(col => df.agg(max(col)).collect()(0)(0).asInstanceOf[Long])

  val maxAmongMaxValues = maxValues.max

  maxAmongMaxValues
}



//para cada atributo categorico
for (i <- 0 until listaAtributosCategoricos.length) {
  val columna_actual = listaAtributosCategoricos(i)

    //compruebo su correlacion con el resto de atributos
    for (j <- (i + 1) until listaAtributosCategoricos.length) {
      val siguiente_columna = listaAtributosCategoricos(j)
      //se hace la tabla de contingecnia
      var tablaContingencia = census_df.groupBy(columna_actual).pivot(siguiente_columna).count().na.fill(0)

      tablaContingencia = tablaContingencia.drop(columna_actual)

      tablaContingencia.show()

      tablaContingencia.printSchema()

      //se guardan los resultados
      val resultado = comprobarValoresMayores(tablaContingencia)

      val fila = columna_actual+";"+siguiente_columna+";"+resultado
      println(fila)
      correlaccion_categoricas = correlaccion_categoricas :+ fila
      val nombre=columna_actual+"-"+siguiente_columna
      tablaContingencia.write.mode("overwrite").csv(nombre)
    }
}

sc.parallelize(correlaccion_categoricas.toSeq,1).saveAsTextFile("correlacion_categoricas")



