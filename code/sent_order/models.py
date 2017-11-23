

import re
import spacy

from collections import namedtuple
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


class Sentence(Model):

    schema = T.StructType([
        T.StructField('text', T.StringType()),
        T.StructField('tokens', T.ArrayType(Token.schema)),
    ])

    @classmethod
    def from_text(cls, text):
        """Parse sentence.
        """
        doc = nlp(text)

        tokens = list(map(Token.from_spacy_token, doc))

        return cls(text, tokens)

    def ngrams(self, key, n, lower=False, sep='_'):
        """Generate ngrams from tokens.
        """
        texts = [t[key] for t in self.tokens]

        if lower:
            texts = [t.lower() for t in texts]

        for ng in windowed(texts, n):
            yield sep.join(ng)


class Abstract(Model):

    schema = T.StructType([
        T.StructField('id', T.StringType(), nullable=False),
        T.StructField('tags', T.ArrayType(T.StringType())),
        T.StructField('sentences', T.ArrayType(Sentence.schema)),
        T.StructField('split', T.StringType()),
    ])

    @classmethod
    def from_lines(cls, lines, split):
        """Parse abstract lines.
        """
        sentences = list(map(Sentence.from_text, lines[2:]))

        return cls(
            id=lines[0],
            tags=lines[1].split(),
            sentences=sentences,
            split=split,
        )
