/* Master en Ingeniería Informática - Universidad de Valladolid
*
*  TECNICAS ESCLABLES DE ANÁLISIS DE DATOS EN ENTORNOS BIG DATA: CLASIFICADORES
*  Proyecto Software: Construcción y validación de un modelo de clasificación usando la metodología CRISP-DM y Spark
*
*  Script para la transformar el DataFrame
*
*  Grupo 2: Sergio Agudelo Bernal
*           Miguel Ángel Collado Alonso
*           José María Lozano Olmedo.
*/

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


object TransformDataframe {

def transformDataFrame(census_df: DataFrame): DataFrame = {

var attributeColumns = census_df.columns.toSeq.filter(_ != "income").toArray

var outputColumns = attributeColumns.map(_ + "-num").toArray

var siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

var simColumns = siColumns.fit(census_df)

var census_df_numeric = simColumns.transform(census_df).drop(attributeColumns:_*)


//census_df_numeric.show(10)

// Generamos los nombres de las nuevas columnas
var inputColumns = outputColumns
outputColumns = attributeColumns.map(_ + "-hot").toArray


// Creamos OneHotEncoder para transformar todos los atributos, salvo la clase
var hotColumns = new OneHotEncoder().setInputCols(inputColumns).setOutputCols(outputColumns)


//Creamos el OneHotEncoderModel
var hotmColumns = hotColumns.fit(census_df_numeric)


var censusDFhot = hotmColumns.transform(census_df_numeric).drop(inputColumns:_*)
    
// Lo examinamos
//censusDFhot.show(10)

 
//  Definimos columna Features con todos los atributos, menos la clase, con VectorAssempler
//var va = new VectorAssembler().setOutputCol("features").setInputCols(carDFhot.columns.diff(Array("clase")))
var va = new VectorAssembler().setOutputCol("features").setInputCols(outputColumns)


// Creamos el DataFrame carFeaturesClaseDF con columnas features y clase
var censusFeaturesClaseDF = va.transform(censusDFhot).select("features", "income")

// Lo examinamos
//censusFeaturesClaseDF.show(10)


// creamos el StringIndexer para la clase
var indiceClase= new StringIndexer().setInputCol("income").setOutputCol("label").setStringOrderType("alphabetDesc")

// Creamos el DataFrame carFeaturesLabelDF con columnas features y label
var censusFeaturesLabelDF = indiceClase.fit(censusFeaturesClaseDF).transform(censusFeaturesClaseDF).drop("income")
//censusFeaturesLabelDF.show(10)

censusFeaturesLabelDF
}



}
