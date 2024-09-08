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



















// recorremos lista atributos numericos
for (nombre_columna <- listaAtributosNumericos) {
	println("Atributo: "+nombre_columna);

    //valores distintos
	val valores_distintos = census_df.select(nombre_columna).distinct.count();
	println("Valores distintos: "+valores_distintos);

    //resumen del atributo
	val describe = census_df.describe(nombre_columna).show();
    val media = census_df.describe(nombre_columna).filter("summary = 'mean'").select(nombre_columna).first().getString(0)
	
    //valores nulos
    val conteoValoresNulos = census_df.filter(col(nombre_columna) === "?").count()
    println("Nulos: "+conteoValoresNulos);

    

    //distribucion de los valores del atributo
    val distribucion = census_df.groupBy(nombre_columna).agg(count("*").alias("cantidad")).orderBy(desc("cantidad"))
    distribucion.show()


    // guardar en un csv los datos
    distribucion.write.mode("overwrite").csv(nombre_columna)
}

























def numero_diferentes(nombre_columna:String): Long = {
    val cuenta=census_df.select(nombre_columna).distinct().count()
    cuenta
  }

  def valores_diferentes(nombre_columna:String): String = {
    val distintos=census_df.select(nombre_columna).distinct().collect().map(row => row.getString(0))
    val resultado_linea = distintos.mkString(", ")
    resultado_linea
  }

  def numero_cada_uno_diferentes(nombre_columna:String): DataFrame = {
      val numero_cada_uno=census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
      numero_cada_uno
  }

  def crear_fichero_resultados(df:DataFrame,nombre_columna:String):Unit={
    sc.parallelize(df.collect().toSeq,1).saveAsTextFile(nombre_columna+"_ordered")
  }


var array_valores_columnas_categoricas = Array[String]()

for (nombre_columna <- listaAtributosCategoricos) {
  println("Atributo: "+nombre_columna);

  //diferentes y su distribucion
  val numero_atributo_diferentes =numero_diferentes(nombre_columna)
  val valores_atributo_diferentes = valores_diferentes(nombre_columna)

  val numero_cada_uno_diferentes_atributo = numero_cada_uno_diferentes(nombre_columna)
  numero_cada_uno_diferentes_atributo.show(numero_cada_uno_diferentes_atributo.count().toInt, false)
  crear_fichero_resultados(numero_cada_uno_diferentes_atributo,nombre_columna)

  //nulos
  val conteoValoresNulos = census_df.filter(col(nombre_columna) === "?").count()
  println("Nulos: "+conteoValoresNulos+"\n\n");
  
  //moda
  val moda_atributo = numero_cada_uno_diferentes_atributo.first().getString(0)


  var escribir=nombre_columna+": "+valores_atributo_diferentes
  array_valores_columnas_categoricas=array_valores_columnas_categoricas:+escribir

}

sc.parallelize(array_valores_columnas_categoricas.toSeq,1).saveAsTextFile("resumen")




