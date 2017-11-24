

import spacy

from collections import namedtuple
from pyspark.sql import SparkSession, types as T
from inflection import singularize


nlp = spacy.load('en')


class ModelMeta(type):

    def __new__(meta, name, bases, dct):
        """Generate a namedtuple from the `schema` class attribute.
        """
        if isinstance(dct.get('schema'), T.StructType):

            Row = namedtuple(name, dct['schema'].names)

            # By default, default all fields to None.
            Row.__new__.__defaults__ = (None,) * len(Row._fields)

            bases = (Row,) + bases

        return super().__new__(meta, name, bases, dct)


class Model(metaclass=ModelMeta):

    @classmethod
    def from_row(cls, row):
        """Wrap a raw `Row` instance from an RDD as a model instance.

        Args:
            row (pyspark.sql.Row)

        Returns: Model
        """
        return cls(**row.asDict())

    def __getattr__(self, attr):
        """Automatically re-wrap nested collections.
        """
        if attr.startswith('_'):

            field = attr[1:]

            cls_name = singularize(field).capitalize()
            cls = globals()[cls_name]

            return self.wrap_children(field, cls)

    def wrap_children(self, field, cls):
        """Re-wrap a nested collection.
        """
        return list(map(cls.from_row, getattr(self, field)))


class Sentence(Model):

    schema = T.StructType([
        T.StructField('text', T.StringType()),
        T.StructField('token', T.ArrayType(T.StringType())),
        T.StructField('lemma', T.ArrayType(T.StringType())),
        T.StructField('pos1', T.ArrayType(T.StringType())),
        T.StructField('pos2', T.ArrayType(T.StringType())),
        T.StructField('dep', T.ArrayType(T.StringType())),
    ])

    @classmethod
    def from_text(cls, text):
        """Parse a sentence.
        """
        tokens = list(nlp(text))

        return cls(
            text=text,
            token=[t.text for t in tokens],
            lemma=[t.lemma_ for t in tokens],
            pos1=[t.pos_ for t in tokens],
            pos2=[t.tag_ for t in tokens],
            dep=[t.dep_ for t in tokens],
        )


class Abstract(Model):

    schema = T.StructType([
        T.StructField('id', T.StringType(), nullable=False),
        T.StructField('tags', T.ArrayType(T.StringType())),
        T.StructField('sentences', T.ArrayType(Sentence.schema)),
    ])

    @classmethod
    def from_lines(cls, lines):
        """Parse abstract lines.
        """
        sentences = list(map(Sentence.from_text, lines[2:]))

        return cls(
            id=lines[0],
            tags=lines[1].split(),
            sentences=sentences,
        )
