"""
Extract GloVe embeddings for metaphor identification
"""

import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import re
import numpy as np
from tqdm import tqdm


HIDDEN_DIM_SIZE = 300


def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Extract GloVe vectors with Spacy',
        formatter_class=ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

    nlp = spacy.load('en_vectors_web_lg')
    nlp.tokenizer = custom_tokenizer(nlp)

    vuamc = pd.read_csv('./data/vuamc.csv',
                        keep_default_na=False)

    unique_ctx = vuamc.min_context.unique()
    ctx_embs = np.stack([nlp(ctx).vector for ctx in unique_ctx])
    ctx_to_idx = {ctx: i for i, ctx in enumerate(unique_ctx)}

    v_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)
    s_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)
    o_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)
    vuamc_rows_to_idxs = np.zeros(vuamc.shape[0], dtype=np.int32)

    for i, row in tqdm(vuamc.iterrows(), total=vuamc.shape[0]):
        ctx_idx = ctx_to_idx[row.min_context]
        vuamc_rows_to_idxs[i] = ctx_idx

        v_emb = nlp(row.verb_lemma).vector if row.verb_lemma else np.zeros(HIDDEN_DIM_SIZE, dtype=np.float32)
        s_emb = nlp(row.subject).vector if row.subject else np.zeros(HIDDEN_DIM_SIZE, dtype=np.float32)
        o_emb = nlp(row.object).vector if row.object else np.zeros(HIDDEN_DIM_SIZE, dtype=np.float32)

        v_embs[i] = v_emb
        s_embs[i] = s_emb
        o_embs[i] = o_emb


    np.savez('./features/glove.npz',
             ctx_embs=ctx_embs, v_embs=v_embs, s_embs=s_embs, o_embs=o_embs,
             ctx_idxs=vuamc_rows_to_idxs,
             y=np.array(vuamc.y.values, dtype=np.uint8),
             partition=vuamc.partition.values, genre=vuamc.genre.values, id=vuamc.id.values)
