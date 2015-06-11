#!/usr/bin/env python

from distutils.core import setup

setup(name='Document2Vec',
      version='0.1',
      description='Finding document vectors from pre-trained word vectors',
      author='Christopher Erick Moody',
      author_email='chrisemoody@gmail.com',
      install_requires=['pandas', 'numpy', 'gensim'],
      url='https://github.com/cemoody/Document2Vec')
