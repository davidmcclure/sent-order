

from pyspark import SparkContext
from pyspark.sql import SparkSession


def get_spark():
    """Build sc and spark.
    """
    sc = SparkContext()
    spark = SparkSession(sc).builder.getOrCreate()

    return sc, spark


def try_or_none(f):
    """Wrap a class method call in a try block. On error return None.
    """
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            return None
    return wrapper
