

import click

from sent_order.utils import get_spark
from sent_order.sources import Corpus
from sent_order.models import Abstract


@click.command()
@click.option('--src', default='/data/abstracts/test.txt')
@click.option('--dest', default='/data/test.json')
def main(src, dest):
    """Ingest abstracts.
    """
    sc, spark = get_spark()

    corpus = Corpus(src)

    # Read lines, partition.
    lines = list(corpus.abstract_lines())[:10]
    lines = sc.parallelize(lines)

    # Parse abstracts.
    df = lines.map(Abstract.from_lines).toDF(Abstract.schema)
    df.write.mode('overwrite').json(dest)


if __name__ == '__main__':
    main()
