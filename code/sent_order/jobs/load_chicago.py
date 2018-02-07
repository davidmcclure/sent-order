

import click
import csv

from itertools import islice

from sent_order.utils import get_spark
from sent_order.models import Novel


def parse_novel(metadata, text_dir):
    return Novel.from_metadata(metadata, text_dir)


@click.command()
@click.option('--csv_path', default='/data/CHICAGO_CORPUS/CHICAGO_NOVEL_CORPUS_METADATA/CHICAGO_CORPUS_NOVELS2.csv')
@click.option('--text_dir', default='/data/CHICAGO_CORPUS/CHICAGO_NOVEL_CORPUS')
@click.option('--dest', default='/data/novels.parquet')
def main(csv_path, text_dir, dest):
    """Ingest novels.
    """
    sc, spark = get_spark()

    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        rows = sc.parallelize(islice(reader, 10))

    df = rows.map(lambda r: parse_novel(r, text_dir)).toDF(Novel.schema)

    df.write.mode('overwrite').parquet(dest)


if __name__ == '__main__':
    main()
