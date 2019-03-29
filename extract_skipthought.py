"""
Extract skip-thought embeddings for metaphor identification
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

sys.path.append('./skip-thoughts/')
import skipthoughts

HIDDEN_DIM_SIZE = 4800


def infer_vector_skipthought(m, documents):
    nonzeros = [i for i, d in enumerate(documents) if d]
    nonzero_docs = [d for d in documents if d]
    return_arr = np.zeros((len(documents), 4800), dtype=np.float32)
    encoded_docs = m.encode(nonzero_docs)
    for idx, encoded in zip(nonzeros, encoded_docs):
        return_arr[idx] = encoded
    return return_arr


@np.vectorize
def decode(x):
    return x.decode('utf8')


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Extract GloVe vectors with Spacy',
        formatter_class=ArgumentDefaultsHelpFormatter)

    m = skipthoughts.Encoder(skipthoughts.load_model())

    args = parser.parse_args()

    vuamc = pd.read_csv('./data/vuamc.csv',
                        keep_default_na=False)
    vuamc.min_context = decode(vuamc.min_context)

    unique_ctx = vuamc.min_context.unique()
    ctx_embs = infer_vector_skipthought(m, unique_ctx)

    ctx_to_idx = {ctx: i for i, ctx in enumerate(unique_ctx)}

    v_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)
    s_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)
    o_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)
    vuamc_rows_to_idxs = np.zeros(vuamc.shape[0], dtype=np.int32)


    v_embs = infer_vector_skipthought(m, decode(vuamc.verb_lemma))
    s_embs = infer_vector_skipthought(m, decode(vuamc.subject))
    o_embs = infer_vector_skipthought(m, decode(vuamc.object))

    for i, row in tqdm(vuamc.iterrows(), total=vuamc.shape[0]):
        ctx_idx = ctx_to_idx[row.min_context]
        vuamc_rows_to_idxs[i] = ctx_idx

    np.savez('./features/skipthought.npz',
             ctx_embs=ctx_embs, v_embs=v_embs, s_embs=s_embs, o_embs=o_embs,
             ctx_idxs=vuamc_rows_to_idxs,
             y=np.array(vuamc.y.values, dtype=np.uint8),
             partition=vuamc.partition.values, genre=vuamc.genre.values, id=vuamc.id.values)
