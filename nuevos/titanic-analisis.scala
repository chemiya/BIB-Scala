

// Cargar el conjunto de datos 
val titanicDF = spark.read.option("header", "true").csv("titanic.csv")

// Imprimir el esquema
println("Schema of the Titanic dataset:")
titanicDF.printSchema()

// Imprimir los datos
println("Sample data from the Titanic dataset:")
titanicDF.show()

// tipo de variable
println("Type of the Titanic dataset:")
println(titanicDF.getClass)

// datos de dos columnas
println("Data from 'Cabin' and 'Embarked' columns:")
titanicDF.select("Cabin", "Embarked").show()

// tipos de las columnas
println("Data types of columns:")
titanicDF.dtypes.foreach(println)

// resumen de los datos
println("Summary of the Titanic dataset:")
titanicDF.describe().show()

// crear columnas
val titanicDFWithNewColumn = titanicDF.withColumn("Fare_Multiplied", col("Fare") * 0.21)

// renombrar columna
val titanicDFWithRenamedColumn = titanicDFWithNewColumn.withColumnRenamed("Fare_Multiplied", "New_Fare")

// mostrar datos
println("Data with the new column:")
titanicDFWithRenamedColumn.show()

// Eliminar columna
val titanicDFWithoutNewColumn = titanicDFWithRenamedColumn.drop("New_Fare")

// Mostrar otra vez
println("Data without the new column:")
titanicDFWithoutNewColumn.show()

// Detener la sesión de Spark
spark.stop()
