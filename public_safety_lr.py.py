# 1. Import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# 2. Start Spark session
spark = SparkSession.builder.appName("PublicSafetyLinearRegressionFinal").getOrCreate()

# 3. Load the CSV dataset
datapath = "safetydata.csv"  # use your CSV filename here
data = spark.read.csv(datapath, header=True, inferSchema=True)
data.show(5)
data.printSchema()

# 4. Data cleaning: drop nulls & remove zero response times
data = data.na.drop()
data = data.filter(col("responsetime") != 0)

# 5. See which features are most correlated to response time
for colname in data.columns:
    if colname != "responsetime":
        correlation = data.stat.corr(colname, "responsetime")
        print(colname, correlation)

# 6. Prepare features for model
featurecolumns = ["populationdensity", "numofficers", "avgincome", "crimerate", "distancetostation"]
assembler = VectorAssembler(inputCols=featurecolumns, outputCol="features")
dataassembled = assembler.transform(data)
datafinal = dataassembled.select(*featurecolumns, "responsetime", "features")

# 7. Split into train (70%) and test (30%)
traindata, testdata = datafinal.randomSplit([0.7, 0.3], seed=42)

# 8. Train Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="responsetime")
model = lr.fit(traindata)
print("Model Coefficients:", model.coefficients)
print("Model Intercept:", model.intercept)

# 9. Predict on test set
predictions = model.transform(testdata)
finaloutput = predictions.select(*featurecolumns, "responsetime", col("prediction").alias("predictedresponsetime"))
finaloutput.show(15)

# 10. Evaluate model
evaluatorrmse = RegressionEvaluator(labelCol="responsetime", predictionCol="predictedresponsetime", metricName="rmse")
evaluatorr2 = RegressionEvaluator(labelCol="responsetime", predictionCol="predictedresponsetime", metricName="r2")
rmse = evaluatorrmse.evaluate(finaloutput)
r2 = evaluatorr2.evaluate(finaloutput)
print(f"Model Evaluation Results\nRMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"R2 (Coefficient of Determination): {r2:.4f}")

# 11. Export results (optional, needs pandas library)
pdf = finaloutput.toPandas()
exportpath = "safetyresults.csv"
pdf.to_csv(exportpath, index=False)
print("Results Exported Successfully to", exportpath)
