"""
Extract doc2vec metaphors for metaphor identification
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.utils import simple_preprocess

import gensim.models as g

HIDDEN_DIM_SIZE = 300


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Extract GloVe vectors with Spacy',
        formatter_class=ArgumentDefaultsHelpFormatter)


    m = g.Doc2Vec.load('models/enwiki_dbow/doc2vec.bin')

    args = parser.parse_args()

    vuamc = pd.read_csv('./data/vuamc.csv',
                        keep_default_na=False)

    unique_ctx = vuamc.min_context.unique()
    ctx_embs = np.stack([
        m.infer_vector(simple_preprocess(ctx), alpha=0.01, steps=1000)
        for ctx in tqdm(unique_ctx, desc='Context vectors')])

    ctx_to_idx = {ctx: i for i, ctx in enumerate(unique_ctx)}

    v_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)
    s_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)
    o_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)
    vuamc_rows_to_idxs = np.zeros(vuamc.shape[0], dtype=np.int32)

    for i, row in tqdm(vuamc.iterrows(), total=vuamc.shape[0], desc='Lemmas + Args'):
        ctx_idx = ctx_to_idx[row.min_context]
        vuamc_rows_to_idxs[i] = ctx_idx

        v_emb = m.infer_vector([row.verb_lemma], alpha=0.01, steps=1000)
        s_emb = m.infer_vector([row.subject], alpha=0.01, steps=1000) if row.subject else np.zeros(HIDDEN_DIM_SIZE, dtype=np.float32)
        o_emb = m.infer_vector([row.object], alpha=0.01, steps=1000) if row.object else np.zeros(HIDDEN_DIM_SIZE, dtype=np.float32)

        v_embs[i] = v_emb
        s_embs[i] = s_emb
        o_embs[i] = o_emb


    np.savez('./features/doc2vec.npz',
             ctx_embs=ctx_embs, v_embs=v_embs, s_embs=s_embs, o_embs=o_embs,
             ctx_idxs=vuamc_rows_to_idxs,
             y=np.array(vuamc.y.values, dtype=np.uint8),
             partition=vuamc.partition.values, genre=vuamc.genre.values, id=vuamc.id.values)
