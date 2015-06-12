# Document2Vec
Finding document vectors from pre-trained word2vec word vectors

![Build Status](https://api.travis-ci.org/cemoody/Document2Vec.svg)
![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)

# How to install
Simply install from the git repo like so:

```bash
pip install -e git+git://github.com/cemoody/Document2Vec.git#egg=Package
# on a shared machine without system-python access add --user
```

# How to use
The word2vec file must be a trained gensim Word2Vec file and cannot be Mikolov's
pre-trained vectors. This is because training a new document vector requires
the syn1 layer which the C version of word2vec throws away.

Initialize Document2Vec with pre-trained word vectors from a pre-existing
word2vec training run like so:

```python    
from document2vec.document2vec import Document2Vec
from document2vec.corpora import SeriesCorpus
import pandas as pd
# This must be a gensim Word2Vec or Doc2Vec pickle
d2v = Document2Vec("/home/moody/projects/Parachute/data/data-all-02.py2")
sentences = pd.Series(['i love jackets', 'blue is my favorite color'])
corpus = SeriesCorpus(sentences)
doc_vectors = d2v.transform(corpus)
```

And then semantic similarities can be evaluated directly:

```python
from scipy.spatial.distance import cosine
# vector for 'i love jackets'
v0 = doc_vectors[0, :] 
# vector the word 'jackets'
v1 = d2v['jackets']
similarity = 1 - cosine(v0, v1)
print(similarity) # 0.320
# Of course, the similarity with a word that is literally
# in the sentence is going to be quite high
# What if we try something similar, like coats?
v2 = d2v['coats']
similarity = 1 - cosine(v0, v2)
print(similarity) # 0.265
# And then if we try a very something very dissimilar from the sentece
# like the city of New York we get low similarity:
v3 = d2v['new_york']
similarity = 1 - cosine(v0, v3)
print(similarity) # 0.02
```

# Monitoring training

It can be useful to monitor the training over many iteration
to make sure doc2vec is at (least locally) doing what it should be doing:

```python
from scipy.spatial.distance import cosine
import numpy as np
def monitor(model):
    print model.alpha,
    for word in ['jackets', 'jacket', 'coats', 'dog']:
        print word,': ', 1.0 - cosine(model['SENT_0'], model[word]),
    print " "
d2v.monitor = monitor
doc_vectors = d2v.transform(corpus)
```

Will print something similar to the following:

```
0.25000 jackets :  0.347975713494 jacket :  0.150385576332 coats : 0.305263268479 dog :  0.121432161320
0.20002 jackets :  0.301431248517 jacket :  0.113824911821 coats : 0.272647329817 dog :  0.125565730551
0.15004 jackets :  0.296385793196 jacket :  0.108801409463 coats : 0.267922727947 dog :  0.126922837909
0.10006 jackets :  0.293973052240 jacket :  0.106190931536 coats : 0.265730524733 dog :  0.126504370045
0.05008 jackets :  0.293425048701 jacket :  0.105495592420 coats : 0.264931351959 dog :  0.125495564005
```
 
