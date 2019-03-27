"""
Extract metaphorical/non-metaphorical verbs, their arguments, and contexts
(sentences) from the VU Amsterdam corpus
"""

import json
import os

from tqdm import tqdm
tqdm.pandas()

from bs4 import BeautifulSoup
import pandas as pd
from gensim.utils import simple_preprocess
from pycorenlp import StanfordCoreNLP

nlp = None

GENRE_MAP = {
    'ACPROSE': 'academic',
    'NEWS': 'news',
    'FICTION': 'fiction',
    'CONVRSN': 'conversation'
}


def normalize_whitespace(x):
    return ' '.join(x.strip().split())


def load_vuamc(filepath):
    with open(filepath, 'r') as vuamc_f:
        vuamc = BeautifulSoup(vuamc_f.read(), 'lxml')
    return vuamc.find('text')


def load_lemmas(jsonlines_f):
    verb_lemmas = []
    ids = []
    with open(jsonlines_f, 'r') as jf:
        for line in jf:
            lemma_json = json.loads(line)
            verb_lemmas.append(lemma_json['x']['U-lemmaByPOS'])
            ids.append(lemma_json['id'])
    return pd.DataFrame({'verb_lemma': verb_lemmas, 'id': ids})


def split_text_segment_id(text_segment_id):
    """
    Get the bnc part and fragment id given a text segment id.
    """
    bnc_part, frg_n = text_segment_id.split('-')
    assert len(bnc_part) == 3, text_segment_id
    assert frg_n.startswith('fragment')
    return bnc_part.upper(), int(frg_n[8:])


def load_ets_annot(corpus_fname, index_fname):
    """
    Load ETS training data.

    Corpus_fname contains ids and gold labels; index_fname contains ids and
    genres. We need to consolidate both.
    """
    ids = []
    ys = []
    with open(corpus_fname, 'r') as fin:
        for line in fin:
            ljson = json.loads(line)
            ids.append(ljson['id'])
            ys.append(ljson['y'])

    test_labels = pd.DataFrame({'id': ids, 'y': ys})

    with open(index_fname, 'r') as fin:
        vals = pd.read_csv(fin)

    colnames = [('text_segment_id', str), ('sentence_number',
                                           str), ('sentence_offset', int),
                ('word_offset', int), ('subword_offset', int), ('verb', str)]
    for (colname,
         coltype), series in zip(colnames,
                                 zip(*vals.id.apply(lambda x: x.split('_')))):
        vals[colname] = pd.Series(series).astype(coltype)

    # Also get BNC data
    vals['bnc_file'], vals['bnc_file_n'] = \
        zip(*vals.text_segment_id.apply(split_text_segment_id))

    vals = vals.merge(test_labels, on='id')
    return vals


def load_ets_test(test_fname):
    """
    Load ETS test data.
    """
    vals = pd.read_csv(test_fname, names=['id', 'y'])
    colnames = [('text_segment_id', str), ('sentence_number', str),
                ('sentence_offset', int), ('word_offset',
                                           int), ('subword_offset', int)]
    for (colname,
         coltype), series in zip(colnames,
                                 zip(*vals.id.apply(lambda x: x.split('_')))):
        vals[colname] = pd.Series(series).astype(coltype)

    vals['bnc_file'], vals['bnc_file_n'] = \
        zip(*vals.text_segment_id.apply(split_text_segment_id))

    return vals


def load_fragment(soup, fragment, cache):
    """
    Load and cache a BNC fragment, given the id `fragment`.
    """
    if fragment in cache:
        return cache[fragment]
    fragment_xml = soup.find('text', {'xml:id': fragment})
    cache[fragment] = fragment_xml
    return fragment_xml


def get_sentence(soup, ets_series, cache, get_verb=False):
    """
    Given an ETS example `ets_series`, find the corresponding fragment, and
    retrieve the sentence corresponding to the ETS example.
    """
    frg = load_fragment(soup, ets_series.text_segment_id, cache)
    sentence = frg.find('s', {'n': ets_series.sentence_number})
    if get_verb:
        tokenized, raw_tokens = tokenize_vuamc(sentence, raw=True)
        # Offset starts from 1
        verb = raw_tokens[ets_series['word_offset'] - 1].lower()
        return tokenized, raw_tokens, verb
    tokenized, raw_tokens = tokenize_vuamc(sentence, raw=True)
    return tokenized, raw_tokens


def load_bncf(ets_series, bnc_xml_folder):
    """
    Load BNC file and convert to BeautifulSoup.
    """
    bncf = ets_series.bnc_file
    path_to_bncf = os.path.join(bnc_xml_folder, bncf[0], bncf[0:2],
                                '{}.xml'.format(bncf))
    with open(path_to_bncf, 'r') as bnc_fin:
        return BeautifulSoup(bnc_fin.read(), 'lxml')


def is_div_level(maybe_div, level):
    """
    Check if this BNC div exists and is of the same level as `level` (e.g. 4,
    3, 2)
    """
    if maybe_div is None:
        return False
    return (maybe_div.name == 'div' and 'level' in maybe_div.attrs
            and maybe_div.attrs['level'] == str(level))


def extract_lemmas(sentence):
    """
    Extract lemmas from a BNC XML sentence.
    """
    # Get all word tags - some may be nested in multiword <mw> tags
    children = sentence.find_all('w')
    lemmas = []
    for child in children:
        lemmas.append(child['hw'])
    return lemmas


def get_document(ets_series, cache, frg_sent_cache, lemmatize, bnc_xml_folder):
    """
    Retrieve entire document (i.e. fragment) from bnc_xml.
    """
    frg_sent_key = '{}_{}'.format(ets_series.text_segment_id,
                                  ets_series.sentence_number)
    if frg_sent_key in frg_sent_cache:
        return frg_sent_cache[frg_sent_key]

    if ets_series.bnc_file not in cache:
        cache[ets_series.bnc_file] = load_bncf(ets_series, bnc_xml_folder)
    bnc_xml = cache[ets_series.bnc_file]
    bnc_n = ets_series.bnc_file_n
    # Fragments are divs with level 1
    fragments = bnc_xml.find_all('div', {'level': '1'})
    if not fragments:
        # No levels in this xml file (e.g. KDB.xml)
        fragments = bnc_xml.find_all('div')
    frg = fragments[bnc_n - 1]  # BNC is 1-indexed
    # Get sentence by number
    if ets_series.sentence_number == '675a':
        # Some kind of BNC glitch - just get original 675
        sentence_number = '675'
    else:
        sentence_number = ets_series.sentence_number
    sentence = frg.find('s', {'n': sentence_number})

    smallest_context = None
    smallest_context_len = 1e10

    # Get sentence's parent paragraph or div
    paragraph = sentence.parent
    while paragraph is not None and paragraph.name != 'p':
        paragraph = paragraph.parent
    if paragraph is not None:
        paragraph = paragraph.text.strip()
        if len(paragraph) < smallest_context_len:
            smallest_context_len = len(paragraph)
            smallest_context = paragraph

    # For spoken text, wrapped in utterances
    utterance = sentence.parent
    while utterance is not None and utterance.name != 'u':
        utterance = utterance.parent
    if utterance is not None:
        utterance = utterance.text.strip()
        if len(utterance) < smallest_context_len:
            smallest_context_len = len(utterance)
            smallest_context = utterance

    # Get div4, div3, div2, div1 if they exist
    ret_vals = [None, paragraph, utterance]
    for i in (4, 3, 2):
        div = sentence.parent
        while div is not None and not is_div_level(div, i):
            div = div.parent
        if div is None:
            ret_vals.append(None)
        else:
            div = div.text.strip()
            if len(div) < smallest_context_len:
                smallest_context_len = len(div)
                smallest_context = div
            ret_vals.append(div)

    # div1 is the already-found fragment (could have no level e.g. KBD.xml)
    frg = frg.text.strip()
    ret_vals.append(frg)
    if len(frg) < smallest_context_len:
        smallest_context_len = len(frg)
        smallest_context = frg

    if smallest_context is None:
        print("No contexts found for {}".format(ets_series.id))
        import ipdb
        ipdb.set_trace()
    ret_vals[0] = normalize_whitespace(smallest_context)

    # Get index of sentence in min context
    sentence_text = normalize_whitespace(sentence.text)
    sentence_start_idx = int(ret_vals[0].find(sentence_text))
    if sentence_start_idx < 0:
        import pdb; pdb.set_trace()
    sentence_end_idx = int(sentence_start_idx + len(sentence_text))

    ret_vals.append(sentence_start_idx)
    ret_vals.append(sentence_end_idx)

    # Finally, get genre
    text_units = bnc_xml.find_all('wtext')
    if not text_units:
        text_units = bnc_xml.find_all('stext')
    assert len(text_units) == 1
    genre = text_units[0].attrs['type']
    # Fix genre
    genre = GENRE_MAP.get(genre, 'unknown')
    ret_vals.append(genre)

    # Cache for like sentences (will always be in the same paragraph/div4321)
    frg_sent_cache[frg_sent_key] = ret_vals

    return ret_vals


def pprint_deps(basicdeps):
    """
    Print dependencies according to StanfordNLP.
    """
    for dep in basicdeps:
        print('{}({}-{}, {}-{})'.format(
            dep['dep'],
            dep['governorGloss'],
            dep['governor'],
            dep['dependentGloss'],
            dep['dependent'],
        ))


def extract_arguments(row):
    """
    Extract arguments for the given VUAMC row by dependency parsing via
    Stanford CoreNLP.

    Many verbs are hidden as nouns, modifiers, etc. and do not have
    clean subjects/objects. We treat case modifiers as objects as a special
    case, but do no further preprocessing.

    When there is no subject/object, we return an empty string, rather than
    None, to preserve feather format capabilities.
    """
    global nlp
    if nlp is None:
        nlp = StanfordCoreNLP('http://localhost:9000')
    raw_str = ' '.join(row.raw_sentence)
    output = nlp.annotate(
        raw_str, properties={
            'annotators': 'depparse,lemma',
            'outputFormat': 'json'
        })['sentences'][0]
    idx_to_lemma = dict(enumerate(
        [x['lemma'] for x in output['tokens']], start=1))
    # Loop through dependencies, find verb
    verb_deps = [
        x for x in output['basicDependencies']
        if x['governor'] == row.word_offset
    ]
    if not verb_deps:
        # Check if gloss is in dependents, with case modifier - if so, put that
        # as object
        verb_govs = [
            x for x in output['basicDependencies']
            if x['dependent'] == row.word_offset and x['dep'] == 'case'
        ]
        if not verb_govs:
            return '', '', '', ''
        if len(verb_govs) > 1:
            print("Multiple cases for verb {} id {}".format(row.verb, row.id))
            pprint_deps(verb_govs)
        # Just take the first one
        return ('', verb_govs[0]['governorGloss'].lower(),
                '', idx_to_lemma[verb_govs[0]['governor']].lower())
    subject = ''
    object = ''
    subject_lemma = ''
    object_lemma = ''
    for dep_obj in verb_deps:
        if dep_obj['dep'] == 'nsubj':
            # Found subject
            subject = dep_obj['dependentGloss'].lower()
            subject_lemma = idx_to_lemma[dep_obj['dependent']].lower()
        elif dep_obj['dep'] == 'dobj':
            object = dep_obj['dependentGloss'].lower()
            object_lemma = idx_to_lemma[dep_obj['dependent']].lower()
    if not subject and not object:
        print("No subject/objects found for verb {} id {}".format(
            row.verb, row.id))
        pprint_deps(verb_deps)
    return subject, object, subject_lemma, object_lemma


def fix_word_offset(row):
    """
    There are some off-by-one errors given a verb and its offset into the
    sentence. This tries to fix it by looking at the raw sentence, the word
    index, and the actual verb.
    """
    try:
        if row.raw_sentence[row.word_offset - 1].lower() != row.verb:
            # Check if it's off-by-one
            if row.raw_sentence[row.word_offset].lower() == row.verb:
                print("Fixing word offset {}".format(row.id))
                return row.word_offset + 1
            if row.raw_sentence[row.word_offset + 1].lower() == row.verb:
                print("Fixing word offset {}".format(row.id))
                return row.word_offset + 2
            # Just return the first index of the verb itself
            lower_sentence = [x.lower() for x in row.raw_sentence]
            if row.verb not in lower_sentence:
                # Just return the index and hope it's correct -
                # some subtle problems e.g. British vs American spelling
                return row.word_offset
            else:
                return lower_sentence.index(row.verb) + 1
        # Fine, keep word offset
        return row.word_offset
    except IndexError:
        # End/beginning-of-sentence issues, just trust word index
        return row.word_offset


def tokenize_vuamc(sentence, raw=False):
    """
    Given a sentence as an XML object, tokenize it
    """
    tokens = []
    for el in sentence.findChildren():
        # If there are children, skip
        if len(el.findChildren()) > 0:
            continue
        tokens.append(el.text)
    joined = ' '.join(tokens)
    if raw:
        return simple_preprocess(joined), joined.split()
    return simple_preprocess(joined)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='VU Amsterdam Parser',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--bnc_xml_folder',
        default='/u/nlp/data/bnc_xml/',
        help='Path to uncompressed XML version of BNC (should have ABCDEFGHJK toplevel dirs)')
    parser.add_argument(
        '--lemmatize',
        action='store_true',
        help='Lemmatize VUAMC')
    parser.add_argument(
        '--out', default='./vuamc.csv',
        help='Output CSV file'
    )

    args = parser.parse_args()

    # Load ETS dataset with links back to VUAMC locations
    ets_df = load_ets_annot('./ets/train.jsonlines',
                            './ets/train.csv')
    vuamc = load_vuamc('./VUAMC.xml')

    # Extract sentences (available in VUAMC)
    frg_cache = {}
    print("Extracting sentences and verbs from VUAMC: train")
    ets_df['sentence'], ets_df['raw_sentence'] = zip(*ets_df.progress_apply(
        lambda row: get_sentence(vuamc, row, frg_cache), axis=1))

    # Extract paragraphs and documents (available in BNC)
    print("Extracting broader contexts from original BNC XML: train")
    doc_cache = {}
    fs_cache = {}
    contexts = zip(*ets_df.progress_apply(
        lambda row: get_document(row, doc_cache, fs_cache, args.lemmatize, args.bnc_xml_folder), axis=1))
    context_names = [
        'min_context', 'paragraph', 'utterance', 'div_4', 'div_3', 'div_2',
        'div_1', 'sentence_start_idx', 'sentence_end_idx'  # IGNORE GENRE HERE, already given
    ]
    del doc_cache, fs_cache
    for cname, c in zip(context_names, contexts):
        ets_df[cname] = c

    # Do the same for test
    ets_df_test = load_ets_test('./ets/test.csv')
    ets_df_test['word_offset'] = ets_df_test['sentence_offset'].copy()
    del ets_df_test['sentence_offset']
    print("Extracting sentences and verbs from VUAMC: test")
    ets_df_test['sentence'], ets_df_test['raw_sentence'], ets_df_test[
        'verb'] = zip(*ets_df_test.progress_apply(
            lambda row: get_sentence(vuamc, row, frg_cache, get_verb=True),
            axis=1))
    del frg_cache
    print("Extracting broader contexts from original BNC XML: test")
    doc_cache = {}
    fs_cache = {}
    contexts = zip(*ets_df_test.progress_apply(
        lambda row: get_document(row, doc_cache, fs_cache, args.lemmatize, args.bnc_xml_folder), axis=1))
    context_names = [
        'min_context', 'paragraph', 'utterance', 'div_4', 'div_3', 'div_2',
        'div_1', 'sentence_start_idx', 'sentence_end_idx', 'genre'
    ]
    del doc_cache, fs_cache
    for cname, c in zip(context_names, contexts):
        ets_df_test[cname] = c
    ets_df_test['partition'] = 'test'
    # Some dummy default values
    ets_df_test['sentence_offset'] = 0
    ets_df_test['subword_offset'] = 0
    ets_df_test['fold_no'] = 0

    ets_df = pd.concat([ets_df, ets_df_test])

    # Fix word offsets
    print("Fixing word offsets")
    ets_df['word_offset'] = ets_df.apply(fix_word_offset, axis=1)

    # Dependency parse sentences to extract arguments
    print("Dependency parsing")
    ets_df['subject'], ets_df['object'], ets_df['subject_lemma'], ets_df['object_lemma'] = zip(*ets_df.progress_apply(extract_arguments, axis=1))

    # Get verb lemma. Do this separately because min_contexts/sources of
    # lemmatized verbs are terribly messed up
    lemmas = pd.concat([
        load_lemmas('./ets/train_lemmas.jsonlines'),
        load_lemmas('./ets/test_lemmas.jsonlines')
    ])

    ets_df = ets_df.merge(lemmas, on=['id'])

    # Keep minimum context only, convert back to string
    ets_df = ets_df.drop(['div_1', 'div_2', 'div_3', 'div_4', 'sentence', 'paragraph', 'utterance'],
                         axis=1)
    ets_df['sentence'] = ets_df['raw_sentence'].apply(lambda tokens: ' '.join(tokens))
    del ets_df['raw_sentence']

    print("Saving to {}".format(args.out))
    ets_df.to_csv(args.out, index=False)
