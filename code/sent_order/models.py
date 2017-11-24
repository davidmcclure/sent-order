

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

    def token_seq(self, key):
        return [t[key] for t in self.tokens]

    def ngrams(self, key, n, sep='_'):
        """Generate ngrams from tokens.
        """
        texts = [t[key] for t in self.tokens]

        for ng in windowed(texts, n):
            yield f'_{key}{n}_{sep.join(ng)}'

    def ngram_counts(self, key, maxn=3):
        """Generate ngram counts.
        """
        for n in range(1, maxn+1):
            counts = Counter(self.ngrams(key, n))
            yield from counts.items()

    def ngram_features(self):
        """Generate un-filtered ngram features.
        """
        yield from self.ngram_counts('text')
        yield from self.ngram_counts('lemma')
        yield from self.ngram_counts('pos')
        yield from self.ngram_counts('tag')
        yield from self.ngram_counts('dep')
        yield from self.ngram_counts('shape')

    def word_count(self):
        return len(self.tokens)

    def char_count(self):
        return len(self.text)

    def avg_word_len(self):
        word_lens = [len(t.text) for t in self.tokens]
        return sum(word_lens) / len(word_lens)

    def features(self, vocab=None):
        """Generate feature k/v pairs.
        """
        for ngram, count in self.ngram_features():
            if not vocab or ngram in vocab:
                yield ngram, count

        yield 'word_count', self.word_count()
        yield 'char_count', self.char_count()
        yield 'avg_word_len', self.avg_word_len()

    def x(self, vocab=None):
        return dict(self.features(vocab))


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

    def xy(self, vocab=None):
        """Generate x/y pairs for sentences.
        """
        for i, sent in enumerate(self._sentences):
            x = sent.x(vocab)
            y = i / (len(self.sentences)-1)
            yield x, y


class FlatSentence(Model):

    schema = T.StructType([
        T.StructField('raw', T.StringType()),
        T.StructField('text', T.ArrayType(T.StringType())),
        T.StructField('lemma', T.ArrayType(T.StringType())),
        T.StructField('pos', T.ArrayType(T.StringType())),
        T.StructField('tag', T.ArrayType(T.StringType())),
        T.StructField('dep', T.ArrayType(T.StringType())),
        T.StructField('shape', T.ArrayType(T.StringType())),
    ])

    @classmethod
    def from_sentence(cls, sent):
        """Map in sentence.
        """
        return cls(
            raw=sent.text,
            text=sent.token_seq('text'),
            lemma=sent.token_seq('lemma'),
            pos=sent.token_seq('pos'),
            tag=sent.token_seq('tag'),
            dep=sent.token_seq('dep'),
            shape=sent.token_seq('shape'),
        )


class FlatAbstract(Model):

    schema = T.StructType([
        T.StructField('id', T.StringType(), nullable=False),
        T.StructField('tags', T.ArrayType(T.StringType())),
        T.StructField('sentences', T.ArrayType(FlatSentence.schema)),
    ])

    @classmethod
    def from_abstract(cls, abstract):
        """Map in abstract.
        """
        sentences = map(FlatSentence.from_sentence, abstract._sentences)

        return cls(
            id=abstract.id,
            tags=abstract.tags,
            sentences=list(sentences),
        )
