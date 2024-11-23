from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier

# Creo mi sesión de Spark
spark = SparkSession.builder \
    .appName("Fashion MNIST Model Training") \
    .config("spark.executor.memory", "10g") \
    .config("spark.driver.memory", "5g") \
    .getOrCreate()

# Cargo los datos preparados
train_data = spark.read.parquet("data/prepared_train_data.parquet").repartition(100)

# Entreno el modelo
rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=100)
model = rf.fit(train_data)

# Guardo el modelo
model.save("models/fashion_mnist_rf_model")
