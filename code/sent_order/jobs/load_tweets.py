

import click
import gzip
import ujson

from twitter import fs
from twitter.utils import get_spark
from twitter.models import Tweet


def parse_segment(path):
    """Parse a raw minute file.
    """
    with gzip.open(fs.read(path)) as fh:
        for line in fh.readlines():

            try:

                raw = ujson.loads(line)

                if raw['verb'] == 'post':
                    yield Tweet.from_gnip_json(raw)

            except Exception as e:
                print(e)


@click.command()
@click.option('--src', default='/data/twitter-lsm')
@click.option('--dest', default='/data/tweets.parquet')
def main(src, dest):
    """Ingest tweets.
    """
    sc, spark = get_spark()

    paths = list(fs.scan(src, '\.json.gz'))

    paths = sc.parallelize(paths, len(paths))

    df = paths.flatMap(parse_segment).toDF(Tweet.schema)

    df.write.mode('overwrite').parquet(dest)


if __name__ == '__main__':
    main()
