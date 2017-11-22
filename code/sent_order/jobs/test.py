

from sent_order.session import sc, spark


if __name__ == '__main__':
    sc.parallelize([(1,2), (3,4)]).toDF(('a', 'b')) \
        .write.parquet('test.parquet')
