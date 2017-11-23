

import click

from sent_order.session import sc, spark
from sent_order.models import Abstract


def count_ngrams(abstract, key, n, lower):
    for sent in abstract._sentences:
        for ng in sent.ngrams(key, n, lower):
            yield ng, 1


def most_freq_ngrams(abstracts, key, n, depth, lower=False):
    """Get the N most frequent ngrams.
    """
    counts = abstracts \
        .flatMap(lambda a: count_ngrams(a, key, n, lower)) \
        .reduceByKey(lambda a, b: a + b) \
        .toDF(('ngram', 'count')) \
        .orderBy('count', ascending=False) \
        .head(depth)

    return [r.ngram for r in counts]


@click.command()
@click.option('--src', default='/data/abstracts.parquet')
def main(src):
    """Count tokens.
    """
    df = spark.read.parquet(src)

    rdd = df.rdd.map(Abstract.from_row)

    print(most_freq_ngrams(rdd, 'text', 2, 100, True))


if __name__ == '__main__':
    main()
