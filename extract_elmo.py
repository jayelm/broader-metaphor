"""
Extract ELMo embeddings for metaphor identification
"""

from allennlp.commands.elmo import ElmoEmbedder, DEFAULT_BATCH_SIZE
from allennlp.common.util import lazy_groups_of
import pandas as pd
from tqdm import tqdm
import numpy as np
import re

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex


def quote_merger(doc):
    # this will be called on the Doc object in the pipeline
    matched_spans = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        matched_spans.append(span)
    for span in matched_spans:  # merge into one token after collecting all matches
        span.merge()
    return doc

BATCH_SIZE = 2
HIDDEN_DIM_SIZE = 1024


class ParagraphElmoEmbedder(ElmoEmbedder):
    def embed_batch_last_layer(self, batch):
        elmo_embeddings = self.embed_batch(batch)
        # Keep last layer only
        return [emb[2] for emb in elmo_embeddings]

    def embed_sentences_last_layer(self, sentences, batch_size=DEFAULT_BATCH_SIZE):
        for batch in lazy_groups_of(iter(sentences), batch_size):
            yield from self.embed_batch_last_layer(batch)


def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)


def batches(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def uw_average(emb, *args):
    return np.mean(emb, 0)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='description',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cuda', type=int, default=-1)

    args = parser.parse_args()

    elmo = ParagraphElmoEmbedder(cuda_device=args.cuda)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
    nlp.tokenizer = custom_tokenizer(nlp)

    vuamc = pd.read_csv('./data/vuamc.csv',
                        keep_default_na=False)
    # Keep track of indices
    vuamc['i'] = list(range(vuamc.shape[0]))
    # Extract unique contexts
    unique_ctx = vuamc.min_context.unique()
    print("{} unique contexts".format(len(unique_ctx)))
    # Contexts to indices
    ctx_to_idx = {ctx: i for i, ctx in enumerate(unique_ctx)}

    print("Extracting verb/arg embeddings")
    # Setup argument/verb/average context embeddings
    vuamc_rows_to_idxs = np.zeros(vuamc.shape[0], dtype=np.int32)
    ctx_embs = np.zeros((len(unique_ctx), HIDDEN_DIM_SIZE),
                        dtype=np.float32)
    v_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)
    s_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)
    o_embs = np.zeros((vuamc.shape[0], HIDDEN_DIM_SIZE), dtype=np.float32)

    for ctx_batch in tqdm(batches(unique_ctx, BATCH_SIZE), desc='Processing ctx',
                          total=len(range(0, len(unique_ctx), BATCH_SIZE))):

        ctx_batch_idx = [ctx_to_idx[ctx] for ctx in ctx_batch]
        ctx_batch_nlp = [nlp(ctx) for ctx in ctx_batch]
        ctx_batch_offsets = [[t.idx for t in toks] for toks in ctx_batch_nlp]
        ctx_batch_tok = [[t.text for t in toks] for toks in ctx_batch_nlp]

        ctx_batch_embs = list(elmo.embed_sentences_last_layer(ctx_batch_tok, BATCH_SIZE))

        # Compute average and store in unique ctx vectors (these don't change)
        ctx_batch_embs_avg = [uw_average(emb, nlptok) for emb, nlptok in zip(ctx_batch_embs, ctx_batch_nlp)]
        for avg_emb, ctx_idx in zip(ctx_batch_embs_avg, ctx_batch_idx):
            ctx_embs[ctx_idx] = avg_emb

        # Now loop through each context
        for ctx, ctx_idx, ctx_emb, offsets, context_tokens in zip(ctx_batch, ctx_batch_idx, ctx_batch_embs,
                                                                  ctx_batch_offsets, ctx_batch_tok):
            vuamc_ctx = vuamc[vuamc.min_context == ctx]
            for _, row in vuamc_ctx.iterrows():
                # In ith vuamc row, assign the correct index into the unique context embs
                vuamc_rows_to_idxs[row.i] = ctx_idx

                v_emb = list(elmo.embed_sentences_last_layer([[row.verb_lemma]], BATCH_SIZE))[0][0]
                if row.subject:
                    s_emb = list(elmo.embed_sentences_last_layer([[row.subject]], BATCH_SIZE))[0][0]
                else:
                    s_emb = np.zeros_like(ctx_emb[0])
                if row.object:
                    o_emb = list(elmo.embed_sentences_last_layer([[row.object]], BATCH_SIZE))[0][0]
                else:
                    o_emb = np.zeros_like(ctx_emb[0])

                v_embs[row.i] = v_emb
                s_embs[row.i] = s_emb
                o_embs[row.i] = o_emb

    np.savez('./features/elmo.npz',
             ctx_embs=ctx_embs, v_embs=v_embs, s_embs=s_embs, o_embs=o_embs,
             ctx_idxs=vuamc_rows_to_idxs,
             y=np.array(vuamc.y.values, dtype=np.uint8),
             partition=vuamc.partition.values, genre=vuamc.genre.values, id=vuamc.id.values)
