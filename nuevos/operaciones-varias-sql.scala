// Crear un DataFrame
val df = spark.createDataFrame(Seq(
  (1, "apple", 5),
  (2, "banana", 15),
  (3, "orange", 10),
  (4, "grape", 20)
)).toDF("id", "fruit", "value")

// Registrar el DataFrame como tabla temporal
df.createOrReplaceTempView("fruits")

// consulta SQL con condición
val result = spark.sql("SELECT * FROM fruits WHERE value > 10")

// Mostrar el resultado
result.show()

// consulta SQL para calcular el valor medio con agrupación
val avgValueByFruit = spark.sql("SELECT fruit, AVG(value) AS avg_value FROM fruits GROUP BY fruit")

// Mostrar el resultado
avgValueByFruit.show()


// consulta SQL con la cláusula IN
val result = spark.sql("SELECT * FROM fruits WHERE fruit IN ('apple', 'banana')")

// Mostrar el resultado
result.show()

// Importar las funciones de ventana
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

// Definir la ventana
val windowSpec = Window.partitionBy("fruit").orderBy("value")

// consulta SQL utilizando una ventana
val result = spark.sql("SELECT *, ROW_NUMBER() OVER (PARTITION BY fruit ORDER BY value) AS row_number FROM fruits")

// Mostrar el resultado
result.show()

