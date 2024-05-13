// Importar las bibliotecas necesarias
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._



// Cargar el conjunto de datos del Titanic desde un archivo CSV
val titanicDF = spark.read.option("header", "true").csv("titanic.csv")

// Mostrar el esquema del DataFrame
titanicDF.printSchema()

// Calcular el porcentaje de hombres y mujeres que sobrevivieron y murieron
val genderSurvivalPercentage = titanicDF.groupBy("Sex", "Survived")
  .agg(count("Survived").alias("Count"))
  .withColumn("Total", sum("Count").over())
  .withColumn("Percentage", col("Count") / col("Total") * 100)
  .drop("Count", "Total")
  .orderBy("Sex", "Survived")
  .show()

// Calcular el porcentaje de personas sobre el total según el puerto de embarque, sexo y si sobrevivieron
val portGenderSurvivalPercentage = titanicDF.groupBy("Embarked", "Sex", "Survived")
  .agg(count("Survived").alias("Count"))
  .withColumn("Total", sum("Count").over())
  .withColumn("Percentage", col("Count") / col("Total") * 100)
  .drop("Count", "Total")
  .orderBy("Embarked", "Sex", "Survived").show()



// Construir una tabla de frecuencia por la edad por intervalos de 5 años
val ageFrequencyTable = titanicDF.select("Age")
  .withColumn("AgeGroup", floor(col("Age") / 5) * 5)
  .groupBy("AgeGroup")
  .agg(count("AgeGroup").alias("Frequency"))
  .orderBy("AgeGroup").show()



// Encontrar el puerto donde la media de edad de los pasajeros que embarcaron fue más alta
val portWithHighestAvgAge = titanicDF.groupBy("Embarked")
  .agg(avg("Age").alias("AvgAge"))
  .orderBy(desc("AvgAge")).show()



// Calcular el porcentaje de pasajeros que sobrevivieron si tenían hijos
val survivedWithChildrenPercentage = titanicDF.filter("Parch > 0")
  .groupBy()
  .agg((sum(when(col("Survived") === 1, 1).otherwise(0)) / count("*") * 100).alias("Percentage")).show()



// Calcular la tarifa media por clase y puerto de embarque
val avgFareByClassAndPort = titanicDF.groupBy("Pclass", "Embarked")
  .agg(avg("Fare").alias("AvgFare"))
  .orderBy("Pclass", "Embarked").show()



// Crear una matriz de correlación entre el sexo, embarked y la clase
val correlationMatrix = titanicDF.stat.crosstab("Sex", "Embarked")
  .join(titanicDF.stat.crosstab("Sex", "Pclass"))
  .drop("Sex_Embarked", "Sex_Pclass").show()



// Detener la sesión de Spark
spark.stop()
