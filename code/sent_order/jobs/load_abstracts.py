

import click

from itertools import islice

from sent_order.utils import get_spark
from sent_order.sources import Corpus
from sent_order.models import Abstract


def parse_lines(lines):
    """Parse abstract lines.
    """
    return Abstract.from_lines(lines)


@click.command()
@click.option('--src', default='/data/abstracts/test.txt')
@click.option('--dest', default='/data/abstracts.parquet')
@click.option('--partitions', default=10000)
def main(src, dest, partitions):
    """Ingest abstracts.
    """
    sc, spark = get_spark()

    corpus = Corpus(src)

    lines = corpus.abstract_lines()

    lines = sc.parallelize(lines, partitions)

    df = lines.map(parse_lines).toDF(Abstract.schema)

    df.write.mode('overwrite').parquet(dest)


if __name__ == '__main__':
    main()
