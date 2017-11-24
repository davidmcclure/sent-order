

import click
import os

from pyspark.sql import Row

from sent_order.utils import get_spark
from sent_order.models import Abstract


def count_ngrams(abstract, key, n):
    """Generate (ngram, 1) pairs for an abstract.
    """
    for sent in abstract._sentences:
        for ngram in sent.ngrams(key, n):
            yield ngram, 1


def most_freq_ngrams(df, key, n, depth):
    """Get the most frequent ngrams of a given type.
    """
    counts = (
        df.rdd
        .map(Abstract.from_row)
        .flatMap(lambda a: count_ngrams(a, key, n))
        .reduceByKey(lambda a, b: a + b)
        .toDF(('ngram', 'count'))
        .orderBy('count', ascending=False)
        .head(depth)
    )

    return [r.ngram for r in counts]


def build_vocab(df):
    """Build the complete set of allowed ngram features.
    """
    vocab = []

    for n in (1, 2, 3):
        vocab += most_freq_ngrams(df, 'lemma', n, 2000)

    for key in ('pos', 'tag', 'dep', 'shape'):
        for n in (1, 2, 3):
            vocab += most_freq_ngrams(df, key, n, 10)

    return set(vocab)


@click.command()
@click.option('--src', default='/data/abstracts.parquet')
@click.option('--dest', default='/data')
def main(src, dest):
    """Extract features.
    """
    sc, spark = get_spark()

    df = spark.read.parquet(src)

    train = df.filter(df.split=='train')
    train.cache()

    # Build vocab on train.
    vocab = build_vocab(train)
    print(vocab)

    # Features for train/dev/test.
    for split in ('train', 'dev', 'test'):

        xy = (
            df.filter(df.split==split).rdd
            .map(Abstract.from_row)
            .flatMap(lambda a: list(a.xy(vocab)))
            .map(lambda r: Row(x=r[0], y=r[1]))
            .toDF()
        )

        path = os.path.join(dest, f'xy-{split}.json')

        xy.write.mode('overwrite').json(path)


if __name__ == '__main__':
    main()
