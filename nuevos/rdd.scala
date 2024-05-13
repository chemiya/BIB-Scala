// Importar la biblioteca necesaria
import org.apache.spark.sql.SparkSession



// Crear un RDD 
val rdd = spark.sparkContext.parallelize(Seq(
  (1, "apple", 5),
  (2, "banana", 15),
  (3, "orange", 10),
  (4, "grape", 20),
  (5, "orange", 20)
))

// Obtener el número de filas 
val numRows = rdd.count()
println(s"Number of rows in the RDD: $numRows")

// Imprimir
println("All elements in the RDD:")
rdd.collect().foreach(println)

// columna que convierta a mayúsculas otra columna
val uppercaseRDD = rdd.map { case (id, fruit, value) => (id, fruit.toUpperCase, value) }

// Filtrar las filas superior a 10
val filteredRDD = uppercaseRDD.filter { case (_, _, value) => value > 10 }

// Sumar los valores reduciendo por columna
val sumByFruitRDD = filteredRDD.map { case (_, fruit, value) => (fruit, value) }
  .reduceByKey(_ + _)

// Imprimir el resultado 
println("Sum of values by fruit:")
sumByFruitRDD.collect().foreach(println)

// Detener la sesión de Spark
spark.stop()
