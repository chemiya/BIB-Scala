// Importar las bibliotecas necesarias
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._


// Definir el esquema personalizado 
val titanicSchema = StructType(Array(
  StructField("PassengerId", IntegerType, nullable = true),
  StructField("Survived", IntegerType, nullable = true),
  StructField("Pclass", IntegerType, nullable = true),
  StructField("Name", StringType, nullable = true),
  StructField("Sex", StringType, nullable = true),
  StructField("Age", IntegerType, nullable = true),
  StructField("SibSp", IntegerType, nullable = true),
  StructField("Parch", IntegerType, nullable = true),
  StructField("Ticket", StringType, nullable = true),
  StructField("Fare", StringType, nullable = true),
  StructField("Cabin", StringType, nullable = true),
  StructField("Embarked", StringType, nullable = true)
))

// Cargar el datos con el esquema personalizado
val titanicDF = spark.read.schema(titanicSchema).csv("titanic.csv")

// Convertir columna a double
val titanicDFConverted = titanicDF.withColumn("Fare", titanicDF("Fare").cast(DoubleType))

// Filtrar filas con mayor a 20 de edad
val filteredAgeDF = titanicDFConverted.filter(titanicDFConverted("Age") > 20)

// Mostrar resultados
println("Filtered Age DataFrame:")
filteredAgeDF.show()

// Filtrar filas con mas de 25 de fare y embarked s
val filteredFareEmbarkedDF = titanicDFConverted.filter(titanicDFConverted("Fare") > 25 && titanicDFConverted("Embarked") === "S")

println("Filtered Fare and Embarked DataFrame:")
filteredFareEmbarkedDF.show()

// suma de fare agrupando por pclass
val fareSumDF = titanicDFConverted.groupBy("Pclass").sum("Fare").withColumnRenamed("sum(Fare)", "Total_Fare")

println("Total Fare Sum by Pclass:")
fareSumDF.show()

// media de edad agrupando por embarked
val ageMeanDF = titanicDFConverted.groupBy("Embarked").avg("Age").withColumnRenamed("avg(Age)", "Average_Age")

println("Average Age by Embarked:")
ageMeanDF.show()

// agrupar por sexo y media de los que sobrevivien y suma de las fare
val groupedSurvivedFareDF = titanicDFConverted.groupBy("Sex").agg(
  avg(when($"Survived" === 1, $"Age")).as("Avg_Age_Survived"),
  sum($"Fare").as("Total_Fare")
)

println("Grouped Data by Sex - Average Age of Survived and Total Fare:")
groupedSurvivedFareDF.show()

// Detener la sesión de Spark
spark.stop()
