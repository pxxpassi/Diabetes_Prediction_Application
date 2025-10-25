from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# Start Spark session
spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()

# Load data
df = spark.read.csv("processed_diabetes_data.csv", header=True, inferSchema=True)

# Define features and label
feature_cols = ['pregnancies', 'glucose_imp', 'blood_pressure_imp', 'skin_thickness_imp',
                'insulin_imp', 'bmi_imp', 'diabetes_pedigree', 'age', 'diabetes_risk_score']

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
rf = RandomForestClassifier(featuresCol="features", labelCol="diabetes", numTrees=100, maxDepth=10)

pipeline = Pipeline(stages=[assembler, rf])

# Train model
model = pipeline.fit(df)

# Save model
model.write().overwrite().save("spark_diabetes_model")
print("✅ PySpark model saved to 'spark_diabetes_model'")
