
from pyspark.sql import SparkSession

# Create a Spark session
spark = (
    SparkSession.builder
    .appName("MasterThesis-Spark-Test")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

# Create a small dataset
data = [
    (1, "Srinidhi", 25),
    (2, "Rahul", 28),
    (3, "Ananya", 22),
    (4, "Priya", 30)
]

columns = ["id", "name", "age"]

# Convert Python data into a Spark DataFrame
df = spark.createDataFrame(data, columns)

print("\nOriginal DataFrame:")
df.show()

print("\nDataFrame Schema:")
df.printSchema()

print("\nPeople older than 25:")
df.filter(df.age > 25).show()

spark.stop()
