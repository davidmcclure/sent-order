

import click

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
        vocab += most_freq_ngrams(df, 'text', n, 2000)

    for key in ('pos', 'tag', 'dep'):
        for n in (1, 2, 3):
            vocab += most_freq_ngrams(df, key, n, 200)

    return set(vocab)


@click.command()
@click.option('--src', default='/data/abstracts.parquet')
@click.option('--split', default=None)
def main(src, split):
    """Count tokens.
    """
    sc, spark = get_spark()

    df = spark.read.parquet(src)

    if split:
        df = df.filter(df.split==split)

    vocab = build_vocab(df)

    print(vocab)


if __name__ == '__main__':
    main()
