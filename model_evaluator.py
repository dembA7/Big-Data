from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, concat_ws
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.sql.types import ArrayType, DoubleType

# Inicializo Spark
spark = SparkSession.builder \
    .appName("Fashion MNIST Model Evaluator") \
    .config("spark.executor.memory", "10g") \
    .config("spark.driver.memory", "5g") \
    .getOrCreate()

# Defino un UDF para convertir un Vector a un Array
def vector_to_array(vector):
    return vector.toArray().tolist() if isinstance(vector, Vector) else []

# Cargo datos de prueba
test_data = spark.read.parquet("data/prepared_test_data.parquet")

# Cargo el modelo entrenado
model = RandomForestClassificationModel.load("models/fashion_mnist_rf_model")

# Realizo predicciones
predictions = model.transform(test_data)

# Evalúo el modelo
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

# Imprimo las métricas del modelo
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score}")

# Guardo las métricas en un archivo de texto
with open("model_metrics.txt", "w") as metrics_file:
    metrics_file.write(f"Accuracy: {accuracy}\n")
    metrics_file.write(f"Precision: {precision}\n")
    metrics_file.write(f"Recall: {recall}\n")
    metrics_file.write(f"F1-Score: {f1_score}\n")

# Convierte 'features', 'rawPrediction', y 'probability' a arrays

vector_to_array_udf = udf(vector_to_array, ArrayType(DoubleType()))
predictions = predictions.withColumn("features", vector_to_array_udf(col("features"))) \
                         .withColumn("rawPrediction", vector_to_array_udf(col("rawPrediction"))) \
                         .withColumn("probability", vector_to_array_udf(col("probability")))

# Convertir columnas de arrays a cadenas separadas por comas
predictions = predictions.withColumn("features", concat_ws(",", col("features"))) \
                         .withColumn("rawPrediction", concat_ws(",", col("rawPrediction"))) \
                         .withColumn("probability", concat_ws(",", col("probability")))

# Guardo las predicciones en un archivo CSV
predictions.select("label", "prediction", "features", "rawPrediction", "probability").write.mode("overwrite").csv("predictions.csv", header=True)

# Cierro la sesión de Spark
spark.stop()
