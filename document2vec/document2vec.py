import gensim
import math
import copy
import numpy as np
from gensim.models import Doc2Vec, Word2Vec


class Document2Vec(Doc2Vec):
    def __init__(self, filename=None, min_count=1):
        Doc2Vec.__init__(self)
        if filename is not None:
            self.load_from_pickle(filename)
        self.checkpoint = {}
        self.filename = filename
        self.min_count = min_count
        assert 'train_lbls' in dir(self)

    def load_from_pickle(self, filename):
        """
        This loads a pretrained Word2Vec file into this Doc2Vec class.
        """
        model_w2v = Doc2Vec.load(filename)
        for attr in dir(model_w2v):
            if attr == '__dict__':
                continue
            # Skip methods that we already have in this class
            if attr in dir(self) and callable(getattr(model_w2v, attr)):
                continue
            try:
                setattr(self, attr, getattr(model_w2v, attr))
            except AttributeError:
                continue

    def load_from_w2v(self, filename):
        """
        This loads a pretrained Word2Vec file into this Doc2Vec class.
        """
        model_w2v = Doc2Vec.load_word2vec_format(filename, binary=False)
        self._vocab_from = Word2Vec._vocab_from
        self._prepare_sentences = model_w2v._prepare_sentences
        for attr in dir(model_w2v):
            if attr == '__dict__':
                continue
            if attr in dir(self) and callable(getattr(model_w2v, attr)):
                continue
            try:
                setattr(self, attr, getattr(model_w2v, attr))
            except AttributeError:
                continue

    def get_vector(self, word):
        """Return the vector for a word"""
        return self.syn0[self.vocab[word].index]

    def _build_checkpoint(self):
        """Save the current state of the vectors such that
           we can revert training progress."""
        vars = {}
        variables = ['syn0', 'index2word', 'vocab', 'syn1']
        for name in variables:
            var = getattr(self, name, None)
            if var is not None:
                vars[name] = copy.deepcopy(var)
        self.checkpoint = vars

    def _reset_to_checkpoint(self):
        vars = self.checkpoint
        for name, var in vars.items():
            setattr(self, name, var)

    @staticmethod
    def _make_label(prefix, suffix):
        label = '%s_%s' % (prefix, suffix)
        return label

    def _expand_from(self, corpus, prefix=None, labels=None):
        """
        Pass through the dataset once to add the new labels to the model.
        These labels stand in one for each document/sentence and not
        for new vocabulary.
        """
        if prefix is None:
            prefix = 'SENT'
        num_lines = sum(1 for _ in corpus)
        # Expand syn0
        shape = (self.syn0.shape[0] + num_lines, self.syn0.shape[1])
        syn0 = (np.random.random(shape).astype(self.syn0.dtype) - 0.5)
        syn0 /= self.layer1_size
        syn0[:self.syn0.shape[0]] = self.syn0
        self.syn0 = syn0
        index2word_start = len(self.index2word)
        for j, line_no in enumerate(range(num_lines)):
            # Expand vocab
            newvocab = gensim.models.doc2vec.Vocab()
            newvocab.index = len(self.index2word)
            newvocab.sample_probability = 1.0
            # We insert each sentence at the root of the
            # Huffman tree. It's a hack.
            newvocab.code = [1, ] * int(math.log(line_no + 1, 2) + 1)
            label = Document2Vec._make_label(prefix, str(j))
            self.vocab[label] = newvocab
            # Expand index2word
            self.index2word.append(label)
            assert len(self.index2word) == newvocab.index + 1
        return index2word_start

    @staticmethod
    def _calc_alpha(num_iters, i, initial=0.025):
        return initial * (num_iters - i) / num_iters + 0.0001 * i / num_iters

    def _fit(self, corpus, num_iters=20):
        """
        Given a gensim corpus, train the word2vec model on it.
        """
        self.train_word = False
        self.train_lbls = True
        self.index2word_start = self._expand_from(corpus)
        for i in range(0, num_iters):
            self.alpha = Document2Vec._calc_alpha(num_iters, i)
            self.min_alpha = self.alpha
            self.train(corpus)
        start = self.index2word_start
        return self.syn0[start:]

    def fit(self, *args, **kwargs):
        self._fit(*args, **kwargs)

    def fit_transform(self, *args, **kwargs):
        return self._fit(*args, **kwargs)

    def transform(self, corpus, **kwargs):
        self._build_checkpoint()
        vectors = self.fit_transform(corpus, **kwargs)
        self._reset_to_checkpoint()
        return vectors
