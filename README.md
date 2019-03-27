# broader-metaphor

Code and data for "Learning Outside the Box: Discourse-level Features Improve Metaphor Identification", NAACL 2019

## Getting started

## Reproduction

### GloVe

Classification with GloVe vectors requires spaCy's `en_vectors_web_lg` model,
which can be downloaded via

    python -m spacy download en_vectors_web_lg

Then run

    python extract_glove.py

### doc2vec

Unfortunately, classification with doc2vec depends on jhlau's [pretrained doc2vec models](https://github.com/jhlau/doc2vec), which requires python 2.7 and a custom fork of gensim. Instructions are as follows:

- In a python 2.7 virtual environment, install [jhlau's gensim fork](https://github.com/jhlau/gensim):
    - `pip install git+https://github.com/jhlau/gensim`
- Download the English Wikipedia DBOW from [jhlau/doc2vec](https://github.com/jhlau/doc2vec) and uncompress in the `models` directory (so that the path is `models/enwiki_dbow/doc2vec.bin`)

Then run

    python extract_doc2vec.py

(will take a while)
