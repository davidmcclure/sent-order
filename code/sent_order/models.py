

import spacy
import re
import numpy as np

from collections import namedtuple, Counter
from pyspark.sql import SparkSession, types as T
from inflection import singularize
from boltons.iterutils import windowed


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


class Token(Model):

    schema = T.StructType([
        T.StructField('text', T.StringType()),
        T.StructField('lemma', T.StringType()),
        T.StructField('pos', T.StringType()),
        T.StructField('tag', T.StringType()),
        T.StructField('dep', T.StringType()),
        T.StructField('shape', T.StringType()),
    ])

    @classmethod
    def from_spacy_token(cls, token):
        """Map in token.
        """
        return cls(
            text=token.text,
            lemma=token.lemma_,
            pos=token.pos_,
            tag=token.tag_,
            dep=token.dep_,
            shape=token.shape_,
        )


class Novel(Model):

    schema = T.StructType([
        T.StructField('book_id', T.IntegerType()),
        T.StructField('filename', T.StringType()),
        T.StructField('title', T.StringType()),
        T.StructField('auth_last', T.StringType()),
        T.StructField('auth_first', T.StringType()),
        T.StructField('auth_id', T.StringType()),
        T.StructField('publ_date', T.IntegerType()),
        T.StructField('source', T.StringType()),
        T.StructField('clean', T.BooleanType()),
        T.StructField('text', T.StringType()),
        T.StructField('tokens', T.ArrayType(Token.schema)),
    ])
