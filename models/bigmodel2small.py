import gensim
import traceback
import sys
import math
import numpy as np
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import LabeledLineSentence, LabeledSentence


model_w2v = Word2Vec.load("/home/moody/projects/Parachute/data/data-all-02.py2")
words = set([l.strip() for l in open('../data/small.list').readlines()[1:]])


indices = []
for w in words:
    v = model_w2v.vocab.get(w, None)
    if v is None: continue
    indices.append(v.index)
indices = np.array(indices, dtype=np.int)


syn0 = model_w2v.syn0[indices]
syn1 = model_w2v.syn1[indices]
index2word = list(np.array(model_w2v.index2word)[indices])
vocab = {k:v for k, v in model_w2v.vocab.items() if k in words}

for w, v in model_w2v.vocab.items():
    v.index = model_w2v.index2word.find(w)

model_w2v.syn0 = syn0
model_w2v.syn1 = syn1
model_w2v.vocab =vocab
model_w2v.index2word = index2word

model_w2v.save("../data/data-all-02.py2.small")
