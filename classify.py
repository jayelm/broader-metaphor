"""
Perform VUAMC metaphor classification
"""

import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support as prfs
from scipy.stats import binom
import pandas as pd


def preprocess(all_features, features, n_dev=0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    train_idx = all_features['partition'] == 'train'
    test_idx = all_features['partition'] == 'test'

    # Sample dev values
    dev_idx_ints = np.random.choice(np.where(train_idx)[0], size=n_dev, replace=False)
    dev_idx = np.zeros_like(train_idx).astype(np.bool)
    dev_idx[dev_idx_ints] = True
    train_idx[dev_idx_ints] = False

    n_train = train_idx.sum()
    n_dev = dev_idx.sum()
    n_test = test_idx.sum()

    assert n_train + n_dev + n_test == all_features['partition'].shape[0]
    assert not np.any(np.logical_and(np.logical_and(train_idx, dev_idx), test_idx))

    # Get size of all features
    total_f_size = sum(all_features[f_name].shape[1] for f_name in features)

    X_train = np.zeros((n_train, total_f_size), dtype=np.float32)
    X_dev = np.zeros((n_dev, total_f_size), dtype=np.float32)
    X_test = np.zeros((n_test, total_f_size), dtype=np.float32)

    # Fill X arrays
    f_start_num = 0
    for f_name in features:
        if f_name == 'ctx_embs':
            ctx_idxs = all_features['ctx_idxs']
            this_train_idx = ctx_idxs[train_idx]
            this_dev_idx = ctx_idxs[dev_idx]
            this_test_idx = ctx_idxs[test_idx]
            # Need to index via unique ctx embs
        else:
            # Index using boolean array as normal
            this_train_idx = train_idx
            this_dev_idx = dev_idx
            this_test_idx = test_idx
        this_feature = all_features[f_name]
        this_feature_size = this_feature.shape[1]

        train_feats = this_feature[this_train_idx]
        dev_feats = this_feature[this_dev_idx]
        test_feats = this_feature[this_test_idx]

        X_train[:, f_start_num:f_start_num+this_feature_size] = train_feats
        X_dev[:, f_start_num:f_start_num+this_feature_size] = dev_feats
        X_test[:, f_start_num:f_start_num+this_feature_size] = test_feats

        f_start_num += this_feature_size

    assert f_start_num == total_f_size

    y_train = all_features['y'][train_idx]
    y_dev = all_features['y'][dev_idx]
    y_test = all_features['y'][test_idx]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_dev': X_dev,
        'y_dev': y_dev,
        'X_test': X_test,
        'y_test': y_test,
        'train_idx': train_idx,
        'dev_idx': dev_idx,
        'test_idx': test_idx,
        'test_genre': all_features['genre'][test_idx]
    }


def classify_vuamc(all_features, features_to_use, classifier=xgb.XGBClassifier,
                   n_dev=0, seed=0,
                   classifier_args=None):
    if classifier_args is None:
        classifier_args = {}
    print("Extracting features {}".format(', '.join(features_to_use)))
    F = preprocess(all_features, features_to_use, n_dev=n_dev, seed=seed)
    print("{} total features".format(F['X_train'].shape[1]))
    print("{} train, {} dev, {} test".format(F['X_train'].shape[0], F['X_dev'].shape[0], F['X_test'].shape[0]))

    print("Training {}".format(classifier))
    clf = classifier(scale_pos_weight=sum(1 - F['y_train']) / sum(F['y_train']), **classifier_args)

    clf.fit(F['X_train'], F['y_train'])
    print("Scoring")
    y_pred_test = clf.predict(F['X_test'])
    y_pred_dev = clf.predict(F['X_dev'])
    p, r, f1, s = prfs(F['y_test'], y_pred_test, average='binary')
    print("Precision:", p)
    print("Recall:", r)
    print("F1 Score:", f1)
    acc = (F['y_test'] == y_pred_test).mean()
    print("Accuracy:", acc)
    n_features = F['X_train'].shape[1]
    # By genre
    genres = np.unique(F['test_genre'])
    for genre in genres:
        print("Genre: {}".format(genre))
        genre_mask = F['test_genre'] == genre
        y_pred_genre = clf.predict(F['X_test'][genre_mask])
        pg, rg, fg, sg = prfs(F['y_test'][genre_mask], y_pred_genre, average='binary')
        print("Precision:", pg)
        print("Recall:", rg)
        print("F1 Score:", fg)
        accg = (F['y_test'][genre_mask] == y_pred_genre).mean()
        print("Accuracy:", accg)
        print()
    stats = {
        'precision': p,
        'recall': r,
        'f1': f1,
        'accuracy': acc,
        'n_features': n_features
    }
    F['y_pred_test'] = y_pred_test
    F['y_pred_dev'] = y_pred_dev
    return F, stats


def mcnemar_midp(b, c):
    """
    Compute McNemar's test using the "mid-p" variant suggested by:

    M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for
    binary matched-pairs data: Mid-p and asymptotic are better than exact
    conditional. BMC Medical Research Methodology 13: 91.

    `b` is the number of observations correctly labeled by the first---but
    not the second---system; `c` is the number of observations correctly
    labeled by the second---but not the first---system.
    """
    n = b + c
    x = min(b, c)
    dist = binom(n, .5)
    p = 2. * dist.cdf(x)
    midp = p - dist.pmf(x)
    return midp


def test_signif(y_pred_1, y_pred_2, y_test):
    m1_correct_only = sum((y_pred_1 == y_test) & (y_pred_1 != y_pred_2))
    print("# m1 correct not m2 correct", m1_correct_only)
    m2_correct_only = sum((y_pred_2 == y_test) & (y_pred_2 != y_pred_1))
    print("# m2 correct not m1 correct", m2_correct_only)
    return mcnemar_midp(m1_correct_only, m2_correct_only)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Classify metaphors with features',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('features_file', help='Path to .npz features file')
    parser.add_argument('--n_jobs', default=4, type=int, help='How many workers to use')
    parser.add_argument('--n_dev', default=0, type=int, help='How many dev examples to sample')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()

    features = np.load(args.features_file)

    print("====L (LEMMA)====")
    F, stats = classify_vuamc(features, ['v_embs'],
                              n_dev=args.n_dev,
                              classifier_args={'n_jobs': args.n_jobs},
                              seed=args.seed)
    train_idx = F['train_idx']
    dev_idx = F['dev_idx']
    y_pred_test_l = F['y_pred_test']
    y_pred_dev_l = F['y_pred_dev']

    print("\n====LA (LEMMA + ARGS)====")
    F, stats = classify_vuamc(features,
                              ['v_embs', 's_embs', 'o_embs'],
                              n_dev=args.n_dev,
                              classifier_args={'n_jobs': args.n_jobs},
                              seed=args.seed)
    assert np.all(train_idx == F['train_idx'])
    assert np.all(dev_idx == F['dev_idx'])
    y_pred_test_la = F['y_pred_test']
    y_pred_dev_la = F['y_pred_dev']
    print("\nSignifcance test")
    print(test_signif(y_pred_test_l, y_pred_test_la, F['y_test']))

    print("\n====LAC (LEMMA + ARGS + CONTEXT)====")
    F, stats = classify_vuamc(features, ['v_embs', 's_embs', 'o_embs', 'ctx_embs'],
                              n_dev=args.n_dev,
                              classifier_args={'n_jobs': args.n_jobs},
                              seed=args.seed)
    y_pred_test_lac = F['y_pred_test']
    y_pred_dev_lac = F['y_pred_dev']
    print("\nSignificance test")
    print(test_signif(y_pred_test_la, y_pred_test_lac, F['y_test']))

    # Save dev set predictions
    if args.n_dev:
        vuamc = pd.read_csv('./data/vuamc.csv')
        vuamc_dev = vuamc.iloc[F['dev_idx']].copy()
        vuamc_dev['y_pred_l'] = y_pred_dev_l
        vuamc_dev['y_pred_la'] = y_pred_dev_la
        vuamc_dev['y_pred_lac'] = y_pred_dev_lac

        dev_perf_fname = os.path.join('analysis', '{}_dev_predictions.csv'.format(
            os.path.splitext(os.path.basename(args.features_file))[0]))
        print("Saving dev predictions to {}".format(dev_perf_fname))
        vuamc_dev.to_csv(dev_perf_fname, index=False)
    else:
        print("No dev set specified, not saving dev predictions...")
