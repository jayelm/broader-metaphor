# 2018 NAACL Shared Task Split

This is the Educational Testing Service train/test split of the VU Amsterdam
Metaphor corpus for the 2018 NAACL shared task on verbs detection, with some
files concatenated and simplified.

To recreate these files, download `verbs-datasets.zip` and `naacl_flp_test_gold_labels.zip` from [https://github.com/EducationalTestingService/metaphor/releases](https://github.com/EducationalTestingService/metaphor/releases).
Directions for recreating these files are below their descriptions.

## File descriptions

- `train.csv` contains VUAMC XML ids, verbs, and genres for every verb in the **training set**.
    - From `verbs-datasets.zip`: renamed copy of `/datasets/inputs/traintest_verbs_all/all_genres/train/index.csv`
- `train.jsonlines` contains JSON objects, each with a VUAMC XML id and a gold label, for each verb in the **training set**.
    - From `verbs-datasets.zip`: renamed copy of `/datasets/inputs/traintest_verbs_all/all_genres/train/corpus.jsonlines`
- `test.csv` contains VUAMC XML ids and gold labels each verb in the **test set**.
    - From `naacl_flp_test_gold_labels.zip`: renamed copy of `verb_tokens.csv`
- **Lemmas**:
    - `train_lemmas.jsonlines` contains JSON objects, each with a VUAMC XML id, precomputed lemma, and gold label, for each verb in the **training set**.
        - From `verbs-datasets.zip`: renamed copy of `datasets/inputs/traintest_verbs_all/all_genres/train/U-lemmaByPOS.jsonlines`
    - `test_lemmas.jsonlines` is the same as `train_lemmas.jsonlines`, but has verbs in the **test set**.
        - From `verbs-datasets.zip`: concatenation of `datasets/inputs/traintest_verbs/{academic,conversation,fiction,news}/test/U-lemmaByPOS.jsonlines`
