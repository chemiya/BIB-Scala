// Importar las bibliotecas necesarias
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StringIndexer


// Cargar el conjunto de datos
val titanicDF = spark.read.option("header", "true").csv("titanic.csv")



// distribución de pasajeros según la edad y la clase por intervalos
val ageClassDistribution = titanicDF.select("Age", "Pclass")
  .na.drop(Seq("Age"))
  .withColumn("AgeGroup", floor(col("Age") / 10) * 10)
  .groupBy("AgeGroup", "Pclass")
  .count()
  .orderBy("AgeGroup", "Pclass").show()

// Encontrar la clase en la que más hombres sobrevivieron de todos los hombres que embarcaron en el puerto S
val classWithMostSurvivedMen = titanicDF.filter(col("Sex") === "male" && col("Embarked") === "S")
  .groupBy("Pclass")
  .agg(sum(when(col("Survived") === 1, 1).otherwise(0)).alias("SurvivedMen"))
  .orderBy(desc("SurvivedMen"))
  .select("Pclass")
  .show()


//eliminamos filas con nulos en estos atributos
val cleanedDF = titanicDF.na.drop(Seq("Embarked", "Sex"))
// String indexer 
val stringIndexer = new StringIndexer()
  .setInputCols(Array("Sex", "Embarked"))
  .setOutputCols(Array("SexIndex", "EmbarkedIndex"))

val titanicIndexed = stringIndexer.fit(cleanedDF).transform(cleanedDF)


import org.apache.spark.sql.types.IntegerType

// Con la función cast()
val titanicDFWithIntPclass = titanicIndexed.withColumn("Pclass", col("Pclass").cast(IntegerType))




// Correlación entre el sexo, puerto de embarque y clase
val correlationMatrix = titanicDFWithIntPclass.stat.corr("SexIndex", "EmbarkedIndex")  // Correlación entre sexo y puerto de embarque
val correlationMatrix2 = titanicDFWithIntPclass.stat.corr("SexIndex", "Pclass")  // Correlación entre sexo y clase
val correlationMatrix3 = titanicDFWithIntPclass.stat.corr("EmbarkedIndex", "Pclass")  // Correlación entre puerto de embarque y clase

println(s"Correlación entre sexo y puerto de embarque: $correlationMatrix")
println(s"Correlación entre sexo y clase: $correlationMatrix2")
println(s"Correlación entre puerto de embarque y clase: $correlationMatrix3")


// Crear un DataFrame para mostrar las correlaciones en una tabla
val correlationDF = Seq(
  ("Sex-Embarked", correlationMatrix),
  ("Sex-Pclass", correlationMatrix2),
  ("Embarked-Pclass", correlationMatrix3)
).toDF("Attribute Pair", "Correlation")

correlationDF.show()


// Agrupar por sexo, puerto de embarque y clase
val groupedByAttributes = titanicIndexed.groupBy("Sex", "Embarked", "Pclass").count()

groupedByAttributes.show()


// Detener la sesión de Spark
spark.stop()
