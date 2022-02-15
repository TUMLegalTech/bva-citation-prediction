"""
Majority vote baseline

Usage: python3 majority_vote.py
"""
import tqdm
import math
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from tqdm import tqdm

import util
from dataset_vocab import *


def evaluate_majority_vote(bva_ids):
    """Evaluate majority vote prediction on a set of BVA documents"""
    result_list = []

    for bva_id in tqdm(bva_ids):
        citation_list = util.get_citations(bva_id, vocab)

        for i in range(1, len(citation_list)):
            true_cits = citation_list[i]
            if not true_cits:
                continue

            pred = majority_class
            label = vocab.citation_source_class_by_index(true_cits[0])
            if label == CLASS_UNK:
                continue

            hit_1, hit_5, hit_20 = 0, 0, 0
            true_cits = set(true_cits)

            idx = 0
            for idx, prediction in enumerate(predictions):
                if prediction in true_cits:
                    if idx < 1:
                        hit_1 += 1
                    if idx < 5:
                        hit_5 += 1
                    hit_20 += 1

            recall_1 = hit_1 / len(true_cits)
            recall_5 = hit_5 / len(true_cits)
            recall_20 = hit_20 / len(true_cits)

            result_list.append((pred, label, recall_1, recall_5, recall_20))

    small_class_pred, small_class_label, recall_1_list, \
        recall_5_list, recall_20_list = zip(*result_list)

    return {
        'small_class': classification_report(y_true=small_class_label,
                                             y_pred=small_class_pred,
                                             digits=4, output_dict=True),
        'full': {
            'recall@1': np.mean(recall_1_list),
            'recall@5': np.mean(recall_5_list),
            'recall@20': np.mean(recall_20_list),
        }
    }


def save_evaluation(evaluation, i, small_class_df_all, full_df_all):
    """Merge the evaluation results into a dataframe"""
    fold_id = 'fold_{}'.format(i+1)

    small_class_df = pd.DataFrame(evaluation['small_class']) \
        .melt(value_vars=['case', 'code', 'reg'], value_name=fold_id, ignore_index=False) \
        .reset_index().groupby(['variable', 'index']).sum()
    small_class_df.loc[('all', 'f1-macro'), :] = small_class_df[
        small_class_df.index.get_level_values(1) == 'f1-score'].mean()
    small_class_df_all = pd.concat([small_class_df_all, small_class_df], axis=1)

    full_df = pd.DataFrame(evaluation['full'], index=[fold_id]).transpose()
    full_df_all = pd.concat([full_df_all, full_df], axis=1)

    return small_class_df_all, full_df_all


if __name__ == '__main__':
    # Find the 20 most popular citations
    print('Loading vocab and dataset...')
    vocab = util.get_vocab()
    n_recommendations = 20
    majority_class = CLASS_REG
    majority_citations = sorted(list(vocab.citation_counts.items()),
                                key=lambda v: -v[1])[:n_recommendations]
    predictions = [vocab.citation_index(cit) for cit, _ in majority_citations]

    # Evaluate majority vote prediction on six folds of the test set
    print('Starting prediction...')
    small_class_df_all = pd.DataFrame()
    full_df_all = pd.DataFrame()

    n_folds = 6
    test_ids = util.get_dataset('test')
    fold_size = math.ceil(len(test_ids) / n_folds)

    for i in range(n_folds):
        evaluation = evaluate_majority_vote(
            test_ids[i*fold_size: (i+1)*fold_size])
        small_class_df_all, full_df_all = save_evaluation(
            evaluation, i, small_class_df_all, full_df_all)

    for df in [small_class_df_all, full_df_all]:
        df['mean'] = df.mean(axis=1)
        df['std'] = df.std(axis=1)

    small_class_df_all.to_csv('results/small_class_majority.csv')
    full_df_all.to_csv('results/full_majority.csv')
    print('Evaluation results saved in results/[small_class|full]_majority.csv', file=sys.stderr)

