

from pyspark import SparkContext
from pyspark.sql import SparkSession


sc = SparkContext()

spark = SparkSession(sc).builder.getOrCreate()
