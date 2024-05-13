
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


// Leer el conjunto de datos 
val titanicDF = spark.read.option("header", "true").option("inferSchema", "true").csv("titanic.csv")

// detectar y reemplazar outliers en el atributo 'Age'
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
val titanicDFWithoutOutliers = replaceOutliers(titanicDF, "Age").na.drop("any", Seq("Age"))

// nueva columna en base al atributo 'Age' marcando en qué intervalo de edad 
val titanicDFWithAgeInterval = titanicDFWithoutOutliers.withColumn("Age_Interval", floor(col("Age") / 5) * 5)
val ageFreq = titanicDFWithAgeInterval.groupBy("Age_Interval").count()
ageFreq.show()

// Agrupar por Embarked, Pclass y Sex 
val groupedDF = titanicDFWithAgeInterval.groupBy("Embarked", "Pclass", "Sex").count().orderBy("Embarked", "Pclass", "Sex")
println("Grouped by Embarked, Pclass, and Sex:")
groupedDF.show()

// tabla de frecuencias del atributo 'Fare'
val fareFreq = titanicDFWithAgeInterval.groupBy("Fare").count().orderBy("Fare")
println("Fare frequency table:")
fareFreq.show()

// Detener la sesión de Spark
spark.stop()
