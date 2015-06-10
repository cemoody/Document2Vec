import unittest
import random
import os.path
import pandas as pd
import numpy as np
from document2vec.document2vec import Document2Vec
from document2vec.corpora import SeriesCorpus
from gensim.models.doc2vec import LabeledLineSentence

w2v_file = os.path.realpath("models/small.w2v")


def _generate_corpus(model, n=10):
    docs = {}
    random.seed(0)
    max_nword = len(model.syn0)
    for j in range(n):
        idxs = [random.randint(0, max_nword) for _ in range(8)]
        words = (model.index2word[idx] for idx in idxs)
        sentence = ' '.join(words)
        docs[j] = sentence
    series = pd.Series(docs)
    corpus = SeriesCorpus(series)
    return corpus


class TestDoc2Vec(unittest.TestCase):
    @unittest.skipIf(not os.path.exists(w2v_file),
                     "Need the file %s to continue" % w2v_file)
    def test_init(self):
        m = Document2Vec()
        assert 'train_lbls' in dir(m)

    def test_load_from_w2v(self):
        model = Document2Vec(w2v_file)
        self.assertIsNot(type(model), None)
        self.assertIs(type(model), Document2Vec)
        self.assertIn('jacket', model.index2word)

    @unittest.skipIf(not os.path.exists(w2v_file),
                     "Need the file %s to continue" % w2v_file)
    def test_get_vector(self):
        model = Document2Vec(w2v_file)
        v = model.get_vector('the')
        self.assertIs(type(v), np.ndarray)

    @unittest.skipIf(not os.path.exists(w2v_file),
                     "Need the file %s to continue" % w2v_file)
    def test_word_similarity(self):
        model = Document2Vec(w2v_file)
        sim = model.similarity('blue', 'gold')
        self.assertGreater(sim, 0.3)

    @unittest.skipIf(not os.path.exists(w2v_file),
                     "Need the file %s to continue" % w2v_file)
    def test_checkpoint(self):
        model = Document2Vec(w2v_file)
        checksum = model.syn0.sum()
        model._build_checkpoint()
        model.syn0 *= 2.0
        new_checksum = model.syn0.sum()
        self.assertNotEqual(new_checksum, checksum)
        model._reset_to_checkpoint()
        new_checksum = model.syn0.sum()
        self.assertEqual(new_checksum, checksum)

    @unittest.skipIf(not os.path.exists(w2v_file),
                     "Need the file %s to continue" % w2v_file)
    def test_expand_model(self, n=10):
        model = Document2Vec(w2v_file)
        corpus = _generate_corpus(model, n=n)
        shape_before = model.syn0.shape
        model._expand_from(corpus)
        self.assertEqual(shape_before[0] + n, model.syn0.shape[0])
        self.assertIn('SENT_0', model.index2word)

    @unittest.skipIf(not os.path.exists(w2v_file),
                     "Need the file %s to continue" % w2v_file)
    def test_labeledlinesentence(self):
        model = Document2Vec(w2v_file)
        model.workers = 1
        corpus = _generate_corpus(model)
        fn = '/tmp/tmp_corpus'
        with open(fn, 'w') as fh:
            for line in corpus:
                text = ' '.join([w for w in line.words])
                fh.write(text + '\n')
        corpus = LabeledLineSentence(fn)
        # vectors = model.fit_transform(corpus)
        # Get the first word in the corpus
        model.transform(corpus)
        word = next(next(corpus))
        sim = model.similarity('SENT_0', word)
        self.assertGreater(sim, 0.3)

    @unittest.skipIf(not os.path.exists(w2v_file),
                     "Need the file %s to continue" % w2v_file)
    def test_transform(self):
        """ Test that training the model brings the document vector
            closer to the vectors for words in the sentence"""
        model = Document2Vec(w2v_file)
        model.workers = 1
        corpus = _generate_corpus(model)
        # vectors = model.fit_transform(corpus)
        # Get the first word in the corpus
        model.transform(corpus)
        word = next(next(corpus))
        sim = model.similarity('SENT_0', word)
        self.assertGreater(sim, 0.3)
