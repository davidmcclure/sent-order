

import click

from glob import glob
from os.path import basename, splitext
from itertools import islice
from functools import partial

from pyspark.sql import DataFrame

from sent_order.session import sc, spark
from sent_order.sources import Corpus
from sent_order.models import Abstract


def parse_lines(split, lines):
    """Parse abstract lines.
    """
    return Abstract.from_lines(lines, split)


@click.command()
@click.option('--src', default='/data/abstracts/*.txt')
@click.option('--dest', default='/data/abstracts.parquet')
def main(src, dest):
    """Ingest abstracts.
    """
    results = []

    for path in glob(src):

        corpus = Corpus(path)

        # Read lines, parallelize, partition.
        lines = list(corpus.abstract_lines())
        lines = sc.parallelize(lines, int(len(lines) / 1000))

        # Get split name, bind to worker.
        split = splitext(basename(path))[0]
        parse_split = partial(parse_lines, split)

        res = lines.map(parse_split)
        results.append(res)

    df = sc.union(results).toDF(Abstract.schema)

    df.write.mode('overwrite').parquet(dest)


if __name__ == '__main__':
    main()
