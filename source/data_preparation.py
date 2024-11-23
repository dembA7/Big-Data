from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

# Creo mi sesión de Spark
spark = SparkSession.builder \
    .appName("Fashion MNIST Data Preparation") \
    .getOrCreate()

# Cargo los datos
train_data = spark.read.csv("data/fashion-mnist_train.csv", header=True, inferSchema=True)
test_data = spark.read.csv("data/fashion-mnist_test.csv", header=True, inferSchema=True)

# Preparo las columnas de características
feature_columns = [f"pixel{i}" for i in range(1, 785)]
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

# Transformo los datos
train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

# Selecciono las columnas relevantes
train_data = train_data.select("label", "features")
test_data = test_data.select("label", "features")

# Guardo los datos preparados en formato Parquet
train_data.write.mode('overwrite').parquet("data/prepared_train_data.parquet")
test_data.write.mode('overwrite').parquet("data/prepared_test_data.parquet")

# Muestro las primeras filas del DataFrame modificado
train_data.select("label", "features").show(5)

# Detengo la sesión de Spark
spark.stop()

