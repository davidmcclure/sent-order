

import click

from twitter import fs
from twitter.utils import get_spark


@click.command()
@click.option('--src', default='/data/tweets.parquet')
def main(src):
    """Count tweets.
    """
    sc, spark = get_spark()

    tweets = spark.read.parquet(src)
    print(tweets.count())


if __name__ == '__main__':
    main()
