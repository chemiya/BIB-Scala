import org.apache.spark.sql.types.{IntegerType, StringType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession,Row}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.OneHotEncoderModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer


object CleanDataframe {

def cleanDataframe(census_df: DataFrame): DataFrame = {




val intervalosAge = Array(0, 18, 35, 50, 70, 100)
val labelsAge = Array("0-18", "19-35", "36-50", "51-70", "71-100")


var census_df_nuevo = census_df.withColumn("age-converted",
  when(col("age").between(intervalosAge(0), intervalosAge(1)), labelsAge(0))
    .when(col("age").between(intervalosAge(1), intervalosAge(2)), labelsAge(1))
    .when(col("age").between(intervalosAge(2), intervalosAge(3)), labelsAge(2))
    .when(col("age").between(intervalosAge(3), intervalosAge(4)), labelsAge(3))
    .when(col("age").between(intervalosAge(4), intervalosAge(5)), labelsAge(4))
    .otherwise("Out")
)














val intervalosWageHour = Array(0, 740,  1480,  2220, 3000, 10000)
val labelsWageHour = Array("0-740", "740-1480", "1480-2220", "2220-3000", "3000-10000")


census_df_nuevo = census_df_nuevo.withColumn("wage_per_hour-converted",
  when(col("wage_per_hour").between(intervalosWageHour(0), intervalosWageHour(1)), labelsWageHour(0))
    .when(col("wage_per_hour").between(intervalosWageHour(1), intervalosWageHour(2)), labelsWageHour(1))
    .when(col("wage_per_hour").between(intervalosWageHour(2), intervalosWageHour(3)), labelsWageHour(2))
    .when(col("wage_per_hour").between(intervalosWageHour(3), intervalosWageHour(4)), labelsWageHour(3))
    .when(col("wage_per_hour").between(intervalosWageHour(4), intervalosWageHour(5)), labelsWageHour(4))
    .otherwise("Out")
)












val intervalosCapitalGains = Array(0, 500,  1000,  1500, 2000, 100000)
val labelsCapitalGains = Array("0-500", "500-1000", "1000-1500", "1500-2000", "2000-100000")


census_df_nuevo = census_df_nuevo.withColumn("capital_gains-converted",
  when(col("capital_gains").between(intervalosCapitalGains(0), intervalosCapitalGains(1)), labelsCapitalGains(0))
    .when(col("capital_gains").between(intervalosCapitalGains(1), intervalosCapitalGains(2)), labelsCapitalGains(1))
    .when(col("capital_gains").between(intervalosCapitalGains(2), intervalosCapitalGains(3)), labelsCapitalGains(2))
    .when(col("capital_gains").between(intervalosCapitalGains(3), intervalosCapitalGains(4)), labelsCapitalGains(3))
    .when(col("capital_gains").between(intervalosCapitalGains(4), intervalosCapitalGains(5)), labelsCapitalGains(4))
    .otherwise("Out")
)











val intervalosCapitalLosses = Array(0, 1120,  1680,  2240, 2800, 5000)
val labelsCapitalLosses = Array("0-1120", "1120-1680", "1680-2240", "2240-2800", "2800-5000")


census_df_nuevo = census_df_nuevo.withColumn("capital_losses-converted",
  when(col("capital_losses").between(intervalosCapitalLosses(0), intervalosCapitalLosses(1)), labelsCapitalLosses(0))
    .when(col("capital_losses").between(intervalosCapitalLosses(1), intervalosCapitalLosses(2)), labelsCapitalLosses(1))
    .when(col("capital_losses").between(intervalosCapitalLosses(2), intervalosCapitalLosses(3)), labelsCapitalLosses(2))
    .when(col("capital_losses").between(intervalosCapitalLosses(3), intervalosCapitalLosses(4)), labelsCapitalLosses(3))
    .when(col("capital_losses").between(intervalosCapitalLosses(4), intervalosCapitalLosses(5)), labelsCapitalLosses(4))
    .otherwise("Out")
)












val intervalosDividens = Array(0, 500,  1000,  5000, 10000, 100000)
val labelsDividens = Array("0-500", "500-1000", "1000-5000", "5000-10000", "10000-100000")


census_df_nuevo = census_df_nuevo.withColumn("dividends_from_stocks-converted",
  when(col("dividends_from_stocks").between(intervalosDividens(0), intervalosDividens(1)), labelsDividens(0))
    .when(col("dividends_from_stocks").between(intervalosDividens(1), intervalosDividens(2)), labelsDividens(1))
    .when(col("dividends_from_stocks").between(intervalosDividens(2), intervalosDividens(3)), labelsDividens(2))
    .when(col("dividends_from_stocks").between(intervalosDividens(3), intervalosDividens(4)), labelsDividens(3))
    .when(col("dividends_from_stocks").between(intervalosDividens(4), intervalosDividens(5)), labelsDividens(4))
    .otherwise("Out")
)












val intervalosTotalEarnings = Array(0, 740,  1480,  2220, 3000, 100000)
val labelsTotalEarnings = Array("0-740", "740-1480", "1480-2220", "2220-3000", "3000-100000")


census_df_nuevo = census_df_nuevo.withColumn("total_person_earnings-converted",
  when(col("total_person_earnings").between(intervalosTotalEarnings(0), intervalosTotalEarnings(1)), labelsTotalEarnings(0))
    .when(col("total_person_earnings").between(intervalosTotalEarnings(1), intervalosTotalEarnings(2)), labelsTotalEarnings(1))
    .when(col("total_person_earnings").between(intervalosTotalEarnings(2), intervalosTotalEarnings(3)), labelsTotalEarnings(2))
    .when(col("total_person_earnings").between(intervalosTotalEarnings(3), intervalosTotalEarnings(4)), labelsTotalEarnings(3))
    .when(col("total_person_earnings").between(intervalosTotalEarnings(4), intervalosTotalEarnings(5)), labelsTotalEarnings(4))
    .otherwise("Out")
)







val intervalosPersonsWorkerEmployer = Array(0, 2,  4,  6, 8, 10)
val labelsPersonsWorkerEmployer= Array("0-2", "2-4", "4-6", "6-8", "8-10")


census_df_nuevo = census_df_nuevo.withColumn("num_persons_worked_for_employer-converted",
  when(col("num_persons_worked_for_employer").between(intervalosPersonsWorkerEmployer(0), intervalosPersonsWorkerEmployer(1)), labelsPersonsWorkerEmployer(0))
    .when(col("num_persons_worked_for_employer").between(intervalosPersonsWorkerEmployer(1), intervalosPersonsWorkerEmployer(2)), labelsPersonsWorkerEmployer(1))
    .when(col("num_persons_worked_for_employer").between(intervalosPersonsWorkerEmployer(2), intervalosPersonsWorkerEmployer(3)), labelsPersonsWorkerEmployer(2))
    .when(col("num_persons_worked_for_employer").between(intervalosPersonsWorkerEmployer(3), intervalosPersonsWorkerEmployer(4)), labelsPersonsWorkerEmployer(3))
    .when(col("num_persons_worked_for_employer").between(intervalosPersonsWorkerEmployer(4), intervalosPersonsWorkerEmployer(5)), labelsPersonsWorkerEmployer(4))
    .otherwise("Out")
)












val intervalosWeeksWorked = Array(0, 10,  20,  30, 40, 55)
val labelsWeeksWorked = Array("0-10", "10-20", "20-30", "30-40", "40-55")


census_df_nuevo = census_df_nuevo.withColumn("weeks_worked_in_year-converted",
  when(col("weeks_worked_in_year").between(intervalosWeeksWorked(0), intervalosWeeksWorked(1)), labelsWeeksWorked(0))
    .when(col("weeks_worked_in_year").between(intervalosWeeksWorked(1), intervalosWeeksWorked(2)), labelsWeeksWorked(1))
    .when(col("weeks_worked_in_year").between(intervalosWeeksWorked(2), intervalosWeeksWorked(3)), labelsWeeksWorked(2))
    .when(col("weeks_worked_in_year").between(intervalosWeeksWorked(3), intervalosWeeksWorked(4)), labelsWeeksWorked(3))
    .when(col("weeks_worked_in_year").between(intervalosWeeksWorked(4), intervalosWeeksWorked(5)), labelsWeeksWorked(4))
    .otherwise("Out")
)






val intervalosOwnBusiness = Array(0, 1,2,6)
val labelsOwnBusiness  = Array("0-1", "1-2", "2-6")


census_df_nuevo = census_df_nuevo.withColumn("own_business_or_self_employed-converted",
  when(col("own_business_or_self_employed").between(intervalosOwnBusiness(0), intervalosOwnBusiness(1)), labelsOwnBusiness(0))
    .when(col("own_business_or_self_employed").between(intervalosOwnBusiness(1), intervalosOwnBusiness(2)), labelsOwnBusiness(1))
    .when(col("own_business_or_self_employed").between(intervalosOwnBusiness(2), intervalosOwnBusiness(3)), labelsOwnBusiness(2))
    .otherwise("own_business_or_self_employed")
)

















    //val nuevoDataFrame = census_df_nuevo.select("class_of_worker","education","marital_status","major_industry_code","major_occupation_code","income")
    

    val nuevoDataFrame = census_df_nuevo.select("class_of_worker","education","marital_status","major_industry_code","major_occupation_code","member_of_labor_union","race","sex","full_or_part_time_employment_status","hispanic_Origin","tax_filer_status","region_of_previous_residence","detailed_household_and_family_status","detailed_household_summary_in_house_instance_weight","live_in_this_house_one_year_ago","family_members_under_18","citizenship","age-converted","wage_per_hour-converted","capital_gains-converted","capital_losses-converted","dividends_from_stocks-converted","total_person_earnings-converted","num_persons_worked_for_employer-converted","weeks_worked_in_year-converted","own_business_or_self_employed-converted","income")
    nuevoDataFrame
}
}


//"class_of_worker","industry_code","occupation_code","education","enrolled_in_edu_last_wk","marital_status","major_industry_code","major_occupation_code","member_of_labor_union","race","sex","full_or_part_time_employment_status","reason_for_unemployment","hispanic_Origin","tax_filer_status","region_of_previous_residence","state_of_previous_residence","detailed_household_and_family_status","detailed_household_summary_in_house_instance_weight","migration_code_change_in_msa","migration_code_change_in_reg","migration_code_move_within_reg","live_in_this_house_one_year_ago","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father","country_of_birth_mother","country_of_birth_self","citizenship","fill_inc_questionnaire_for_veterans_ad"