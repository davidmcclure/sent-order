

import click

from twitter.utils import get_spark


@click.command()
@click.option('--src', default='/data/tweets.parquet')
@click.option('--dest', default='/data/locations.txt')
@click.option('--fraction', default=0.1, type=float)
def main(src, dest, fraction):
    """Dump location field examples.
    """
    sc, spark = get_spark()

    tweets = spark.read.parquet(src)

    examples = tweets \
        .filter(tweets.actor.language=='en') \
        .filter(tweets.actor.location.isNotNull()) \
        .select(tweets.actor.location) \
        .sample(False, fraction)

    examples.write.mode('overwrite').text(dest)


if __name__ == '__main__':
    main()
