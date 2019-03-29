# broader-metaphor

Code and data for "Learning Outside the Box: Discourse-level Features Improve Metaphor Identification", NAACL 2019

## Getting started

## Reproduction

In general, to reproduce the results in the paper, you do the following:

1. Run an `extract_*.py` script to generate a numpy features file in
   `features/` which contains embeddings for lemmas, arguments, contexts, and
   gold labels;
2. Run `python classify.py PATH_TO_FEATURES` which does classification with
   XGBoost and prints overall and per-genre performance to `stdout`,
   significance tests comparing LAC vs LA and LA vs A models, and saves
   predictions (with the columns `y_pred_l`, `y_pred_la`, and `y_pred_lac`) to
   the `analysis/` folder.

Specific instructions and requirements for each model follow.

### GloVe

Classification with GloVe vectors requires spaCy's `en_vectors_web_lg` model,
which can be downloaded via

    python -m spacy download en_vectors_web_lg

Then run

    python extract_glove.py

to produce `features/glove.npz`.

### doc2vec

Unfortunately, classification with doc2vec depends on jhlau's [pretrained doc2vec models](https://github.com/jhlau/doc2vec), which requires python 2.7 and a custom fork of gensim. Instructions are as follows:

- In a python 2.7 virtual environment, install [jhlau's gensim fork](https://github.com/jhlau/gensim):
    - `pip install git+https://github.com/jhlau/gensim`
- Download the English Wikipedia DBOW from [jhlau/doc2vec](https://github.com/jhlau/doc2vec) and uncompress in the `models` directory (so that the path is `models/enwiki_dbow/doc2vec.bin`)

Then run

    python extract_doc2vec.py

to produce `features/doc2vec.npz`.

### skip-thought

This also depends on `python2.7`. Clone the skip-thoughts submodule:

    git submodule init && git submodule update

Download the pretrained skip-thoughts models from
[ryankiros/skip-thoughts](https://github.com/ryankiros/skip-thoughts):

    cd models
    mkdir skipthoughts && cd skipthoughts
    wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
    wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
    wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
    wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
    wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
    wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
    wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl

Then run

    python extract_skipthought.py

(will take a while) to produce `features/skipthought.npz`.

### ELMo

This depends on `allennlp`: `pip install allennlp`.

Then run

    python extract_elmo.py --cuda 0

to extract ELMo embeddings (will download and cache the ELMo model) and save
them to `features/elmo.npz`. A GPU and the `--cuda DEVICE_ID` flag is strongly recommended to speed up processing from several hours to an hour.

## Citation

If this code is useful to you, please cite

```
@inproceedings{mu2019learning,
  author    = {Jesse Mu, Noah Goodman, Helen Yannakoudakis, and Ekaterina Shutova},
  title     = {Learning Outside the Box: Discourse-level Features Improve
  Metaphor Identification},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
  year      = {2019}
  }
```
