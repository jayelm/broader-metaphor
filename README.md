# broader-metaphor

Code and data for "Learning Outside the Box: Discourse-level Features Improve Metaphor Identification", NAACL 2019

## Getting started

- `analysis/` contains dev set predictions made by the models in the paper, as
    well as an RMarkdown script which investigates the ELMo model specifically.
- `data/` contains the original `VUAMC.xml` file, and `vuamc.csv`,a  processed
  version with additional context + arguments. For details on how to reproduce
  `VUAMC.csv`, see `data/README.md`.
- `features/` is empty and will contain the extracted model features generated
    by the `extract_*.py` scripts. Contact me if you're lazy and just want to
    run classification with the features outright.
- `models/` is empty and will contain pretrained doc2vec and skip-thought
    models if you want to reproduce those results. (Pretrained embeddings for
    GloVe and ELMo are handled by spaCy and allennlp, respectively; see below)
- `skip-thoughts/` just links to
    [rkiros/skipthoughts](https://github.com/ryankiros/skip-thoughts).
- `classify.py` is the main XGBoost classification script.
- `extract_*.py` are the scripts used to generate classification features.

Each folder has a corresponding README with more details.

## Dependencies

Most of this codebase was tested with Python 3.6 and the following
dependencies:

- `xgboost==0.80`
- `tqdm==4.28.1`
- `sklearn==0.19.1`
- `scipy==1.1.0`
- `numpy==1.14.3`
- `pandas==0.23.4`
- `spacy==2.0.18`
- `allennlp==0.7.1` (for `extract_elmo.py` only)

for reproducing `vuamc.csv` with `data/parse_vuamc.py`:

- `beautifulsoup4==4.6.3`
- `pycorenlp==0.3.0`
- `gensim==3.5.0`
- and the Java Stanford CoreNLP version `3.9.2` (2018-10-05).

Unfortunately, some of the codebase (to generate doc2vec and skipthought
vectors) requires Python 2.7 due to pretrained models and code depending on
2.7. They also may depend on the corresponding version 2.xx packages. See below
for details on how to reproduce each result, and open a GitHub issue if you
have trouble.

Finally, to reproduce the ELMo dev analysis located in `analysis/elmo_dev_analysis.Rmd`, I recommend using RStudio to knit the RMarkdown file. Dependencies are `tidyverse` and `knitr`.

## Reproduction

In general, to reproduce the results in the paper, do the following:

1. Run `python extract_*.py` to generate a numpy features file in
   `features/` which contains embeddings for lemmas, arguments, contexts, and
   gold labels;
2. Run `python classify.py PATH_TO_FEATURES` which does classification with
   XGBoost and prints overall and per-genre performance to `stdout` and
   significance tests comparing LAC vs LA and LA vs A models.
3. To examine a subset of the predictions made by the model, run `classify.py`
   with `--n_dev 500`. This will save a subset of `vuamc.csv` with model
   predictions (with the columns `y_pred_l`, `y_pred_la`, and `y_pred_lac`) to
   the `analysis/` folder. Due to seeding (`-seed 0`) the sampled examples
   should be the same as the ones already contained in `analysis/` (though
   the predictions may not be). Reported performance will also differ slightly.

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

This requires `python2.7`, since
[rkiros/skipthoughts](https://github.com/ryankiros/skip-thoughts) requires 2.7,
and also depends on the dependencies of that module (see rkiros' `README.md`)
Clone the skip-thoughts submodule:

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

Update `path_to_models` and `path_to_tables` in `skip-thoughts/skipthoughts.py`
to `./models/skipthoughts/`

Then run

    python extract_skipthought.py

to produce `features/skipthought.npz`.

### ELMo

With `allennlp` installed, run

    python extract_elmo.py --cuda 0

to extract ELMo embeddings (will download and cache the ELMo model) and save
them to `features/elmo.npz`. A GPU and the `--cuda DEVICE_ID` flag is strongly
recommended to speed up processing from several hours to an hour-ish.

## Citation

If this code is useful to you, please cite

```
@inproceedings{mu2019learning,
  author    = {Jesse Mu, Helen Yannakoudakis, and Ekaterina Shutova},
  title     = {Learning Outside the Box: Discourse-level Features Improve Metaphor Identification},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
  year      = {2019}
}
```
