import utils
import logging
from gensim.corpora import TextCorpus, Dictionary
from gensim.models.doc2vec import LabeledSentence


class SeriesCorpus(TextCorpus):
    def __init__(self, series, tokenizer=utils.tokenize_stem_ngram,
                 vocab=None, stem=False, bigram=None, labels=True):
        """ Create a corpus that returns one row at a time out
            of a Pandas Series"""
        self.series = series
        self.metadata = False
        self.tokenizer = tokenizer
        if vocab is not None:
            vocab = set(vocab)
        self.vocab = vocab
        self.labels = labels
        self.kwargs = dict(stem=stem, bigram=bigram)
        logging.info("Building SeriesCorpus")
        self.dictionary = Dictionary()
        self.dictionary.add_documents(self.get_texts())

    def __iter__(self):
        if self.labels:
            for index, line in zip(self.series.index, self.series.values):
                label = ['SENT_%s' % str(index)]
                yield LabeledSentence(line.split(' '), label)
        else:
            for index, line in self.series.index, self.series.values:
                yield line.split(' ')

    def line_iter(self, line):
        if self.vocab is not None:
            for word in self.tokenizer([line], **self.kwargs)[0]:
                if word in self.vocab:
                    yield word
        else:
            for word in self.tokenizer([line], **self.kwargs)[0]:
                yield word

    def get_texts(self):
        # import pdb; pdb.set_trace()
        logging.info("Iterating SeriesCorpus")
        for lineno, line in enumerate(self.series.values):
            if self.metadata:
                yield self.line_iter(line), (lineno,)
            else:
                yield self.line_iter(line)
