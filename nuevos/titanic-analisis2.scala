
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._



// Leer datos
val titanicDF = spark.read.option("header", "true").option("inferSchema", "true").csv("titanic.csv")

// Mostrar el número de filas y columnas 
val numRows = titanicDF.count()
val numCols = titanicDF.columns.length
println(s"Number of rows in the DataFrame: $numRows")
println(s"Number of columns in the DataFrame: $numCols")

// cuantas filas con nulos
val nullRowsCount = titanicDF.filter(row => row.anyNull).count()
println(s"Number of rows with at least one null value: $nullRowsCount")

// nulos por filas
println("Number of nulls per row:")
titanicDF.select(titanicDF.columns.map(c => sum(when(col(c).isNull, 1).otherwise(0)).alias(c)): _*).show()

// Eliminar las filas con nulo en atributos
val cleanedDF = titanicDF.na.drop(Seq("Embarked", "Fare"))

// media de la columna age
val meanAge = cleanedDF.select(avg("Age")).head().getDouble(0)

// Completar los nulos de age con la media
val filledDF = cleanedDF.na.fill(meanAge, Seq("Age"))

// Agrupar por embarked y contar cuantos embarcaron por puerto
val embarkedCount = filledDF.groupBy("Embarked").count().orderBy("Embarked")
println("Number of passengers embarked at each port:")
embarkedCount.show()

// Contar  hombres y sobrevivieron
val maleSurvivedCount = filledDF.filter($"Sex" === "male" && $"Survived" === 1).count()
println(s"Number of male passengers who survived: $maleSurvivedCount")

// Valores diferentes columna
println("Different values that the 'Embarked' column can take:")
filledDF.select("Embarked").distinct().show()

// Detener la sesión de Spark
spark.stop()
