"""
Implementation of the Collaborative Filtering model
"""
import collections
import math
import os
import random
import sys
import numpy as np
import pandas as pd
import pickle

from multiprocessing import Queue, Process
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from tqdm import tqdm

import util
from dataset_vocab import *


class CollabFilter(object):

    INFO = 2
    WARN = 1
    ERROR = 0

    def __init__(self,
                 relevant_doc_limit=50,
                 recommendation_limit=50,
                 scoring='tf',
                 metadata_names=['year', 'issarea', 'vlj'],
                 svm_model_folder='.',
                 eval_result_folder='.',
                 run_id=None,
                 verbose='error'):

        assert scoring in ['binary', 'tf']
        assert verbose in ['info', 'warn', 'error']
        assert set(metadata_names) <= {'base', 'year', 'issarea', 'vlj'}

        self.relevant_doc_limit = relevant_doc_limit
        self.recommendation_limit = recommendation_limit
        self.scoring = scoring
        self.metadata_names = metadata_names

        self.run_id_suffix = '' if run_id is None else '_{}'.format(run_id)
        self.X = None

        if not os.path.exists(svm_model_folder):
            os.makedirs(svm_model_folder)
        if not os.path.exists(eval_result_folder):
            os.makedirs(eval_result_folder)

        setting = '_'.join(metadata_names) or 'orig'
        self.svm_sample_file = os.path.join(
            svm_model_folder, 'svm_samples{}.csv'.format(self.run_id_suffix))
        self.svm_model_file = os.path.join(
            svm_model_folder, 'svm_coef_{}{}.txt'.format(setting, self.run_id_suffix))
        self.eval_result_file = os.path.join(
            eval_result_folder, '{}_' + '{}{}.csv'.format(setting, self.run_id_suffix))

        self.verbose = {'info': self.INFO,
                        'warn': self.WARN,
                        'error': self.ERROR}[verbose]

        if self.verbose >= self.INFO:
            print('Loading vocab and data structures...', file=sys.stderr)
        self.vocab = util.get_vocab()
        self.inv_lists = util.get_inverted_list()
        self.fwd_idxes = util.get_forward_index()
        self.metadata = util.get_metadata()
        self.metadata_features = util.get_metadata_features(metadata_names)

        self.train_ids = util.get_dataset('train')
        self.test_ids = util.get_dataset('test')

    def make_pairwise_data(self, n_documents=100):
        """Make pairwise training data from the randomly sampled documents"""

        def _sampling_worker_process(self, queue, return_queue):
            """A worker's routine to sample training data from the documents it receives"""

            for serialized_args in iter(queue.get, None):
                citation_list, bva_id = pickle.loads(serialized_args)
                all_rows = []

                for i in range(1, len(citation_list)):
                    # get citation recommendations
                    recommendations = self.find_recommendations(citation_list[:i])
                    if recommendations is None:
                        continue

                    true_cits = citation_list[i]
                    if not recommendations.index.isin(true_cits).any():
                        continue

                    # get metadata features
                    recommendations[self.metadata_names] = recommendations.apply(
                        lambda row: self.get_metadata_features(bva_id, row.name),
                        axis=1, result_type='expand')

                    # normalize the features
                    recommendations = self._normalize(recommendations)

                    # generate pairwise data
                    rows = self.generate_rows(recommendations, citation_list[i])
                    all_rows.extend(rows)

                return_queue.put(all_rows)

        # no need to sample training data if no metadata is required
        if not self.metadata_names:
            return

        # assign the documents to workers
        if self.verbose >= self.INFO:
            print('Preparing to make pairwise data...', file=sys.stderr)

        list_workers, queue, return_queue = self._init_queue_and_workers(
            _sampling_worker_process)
        task_cnt = 0

        random.seed(42)

        while task_cnt < n_documents:
            bva_id = random.sample(self.train_ids, 1)[0]
            citation_list = util.get_citations(bva_id, self.vocab)
            if not citation_list:
                continue

            serialized_args = pickle.dumps((citation_list, bva_id),
                                           protocol=pickle.HIGHEST_PROTOCOL)
            queue.put(serialized_args)
            task_cnt += 1

        for _ in list_workers:
            queue.put(None)  # end of queue

        # collect results from the workers
        all_rows = []
        for _ in tqdm(range(task_cnt)):
            all_rows.extend(return_queue.get())
        for worker in list_workers:
            worker.join()

        self.X = pd.DataFrame(
            all_rows, columns=['score'] + self.metadata_names)
        self.X['label'] = 1

        # flip half of the signs to make two classes balanced
        sign_flips = np.array([(-1)**k for k in range(len(self.X))])
        self.X = self.X.multiply(sign_flips, axis=0)

        if self.verbose >= self.INFO:
            print('Number of training samples:', len(self.X), file=sys.stderr)
            print('Samples saved in ' + self.svm_sample_file, file=sys.stderr)

        self.X.to_csv(self.svm_sample_file, index=False)

    def get_metadata_features(self, bva_id, cit_idx):
        """Get metadata features for a citation in a BVA document"""
        doc_metadata = self.metadata.loc[bva_id]
        feature_values = []

        for name in self.metadata_names:
            if name == 'base':
                feature_value = self.metadata_features[name][cit_idx]
            else:
                feature_value = self.metadata_features[name][doc_metadata[name]][cit_idx]
            feature_values.append(feature_value)
        return feature_values

    def generate_rows(self, recommendations, true_cits):
        """Generate pairwise data from true citation and other citations"""
        true_hit = recommendations.index.isin(true_cits)
        good_cit_features = recommendations.loc[true_hit]
        bad_cit_features = recommendations.loc[~true_hit]

        rows = []
        for i in range(good_cit_features.shape[0]):
            for j in range(bad_cit_features.shape[0]):
                rows.append(good_cit_features.iloc[i].subtract(
                    bad_cit_features.iloc[j]))
        return rows

    def _normalize(self, X):
        X_col_min, X_col_max = X.min(axis=0), X.max(axis=0)
        X = (X - X_col_min) / (X_col_max - X_col_min)
        X = X.fillna(0.0)
        return X

    def train_svm(self, C):
        """Train an SVM model based on the pairwise data"""
        if not self.metadata_names:
            self.coef = np.array([1])
            return

        if self.verbose >= self.INFO:
            print('Start training SVM...', file=sys.stderr)

        if self.X is None:
            self.X = pd.read_csv(self.svm_sample_file)

        self.Y = self.X.label.to_numpy()
        self.X = self.X[['score'] + self.metadata_names].to_numpy()

        clf = LinearSVC(C=C, dual=False, fit_intercept=False)
        clf.fit(self.X, self.Y)
        self.coef = clf.coef_.ravel() / np.linalg.norm(clf.coef_)
        self.X = self.Y = None

        coef_weights = [f"collab_filter: {self.coef[0]:.4f}"]
        for i, _ in enumerate(self.metadata_names):
            coef_weights.append(
                f"{self.metadata_names[i]}: {self.coef[i+1]:.4f}")
        if self.verbose >= self.INFO:
            print("Feature Weights: " + " | ".join(coef_weights), file=sys.stderr)

        np.savetxt(self.svm_model_file, self.coef)
        if self.verbose >= self.INFO:
            print('Coefficients saved in {}'.format(
                self.svm_model_file), file=sys.stderr)

    def predict(self, n_test_docs=None, n_folds=1):
        """Perform prediction on the test documents"""
        small_class_df_all = pd.DataFrame()
        full_df_all = pd.DataFrame()

        if n_test_docs is None:  # use the whole test set
            n_test_docs = len(self.test_ids)

        fold_size = math.ceil(n_test_docs / n_folds)

        # evaluate each fold and combine their results
        for i in range(n_folds):
            evaluation = self.evaluate(
                self.test_ids[i*fold_size: (i+1)*fold_size])
            fold_id = 'fold_{}'.format(i+1)

            small_class_df = pd.DataFrame(evaluation['small_class']) \
                .melt(value_vars=['case', 'code', 'reg'], value_name=fold_id, ignore_index=False) \
                .reset_index().groupby(['variable', 'index']).sum()
            small_class_df.loc[('all', 'f1-macro'), :] = small_class_df[
                small_class_df.index.get_level_values(1) == 'f1-score'].mean()
            small_class_df_all = pd.concat(
                [small_class_df_all, small_class_df], axis=1)

            full_df = pd.DataFrame(evaluation['full'], index=[
                                   fold_id]).transpose()
            full_df_all = pd.concat([full_df_all, full_df], axis=1)

        for df in [small_class_df_all, full_df_all]:
            df['mean'] = df.mean(axis=1)
            df['std'] = df.std(axis=1)

        small_class_df_all.to_csv(self.eval_result_file.format('small_class'))
        full_df_all.to_csv(self.eval_result_file.format('full'))

        if self.verbose >= self.INFO:
            print('Evaluation results saved in ' + self.eval_result_file.format(
                '[small_class|full]'), file=sys.stderr)

        return small_class_df_all, full_df_all

    def evaluate(self, bva_ids):
        """Evaluate the model on a set of BVA documents"""

        def _evaluation_worker_process(self, queue, return_queue):
            """A worker's routine to evaluate on the documents it receives"""

            for serialized_args in iter(queue.get, None):
                citation_list, bva_id = pickle.loads(serialized_args)
                results = []

                # evaluate for each target citation
                for i in range(1, len(citation_list)):

                    recommendations = self.find_recommendations(
                        citation_list[:i])
                    true_cits = citation_list[i]
                    if (recommendations is None) or (recommendations.empty) or (not true_cits):
                        continue

                    # get citation recommendation list
                    recommendations[self.metadata_names] = recommendations.apply(
                        lambda row: self.get_metadata_features(
                            bva_id, row.name),
                        axis=1, result_type='expand')

                    # normalize the citation features
                    X = self._normalize(recommendations).to_numpy()

                    # rerank the citations using the learned SVM
                    recommendations['rerank_score'] = np.matmul(X, self.coef)
                    predictions = recommendations.rerank_score.sort_values(
                        ascending=False)

                    # perform small class prediction
                    pred = self.vocab.citation_source_class_by_index(
                        predictions.index[0])
                    label = self.vocab.citation_source_class_by_index(
                        true_cits[0])
                    if pred == CLASS_UNK or label == CLASS_UNK:
                        continue

                    # perform full citation prediction
                    hit_1, hit_5, hit_20 = 0, 0, 0
                    true_cits = set(true_cits)

                    for idx, prediction in enumerate(predictions.index[:20]):
                        if prediction in true_cits:
                            if idx < 1:
                                hit_1 += 1
                            if idx < 5:
                                hit_5 += 1
                            hit_20 += 1
                    recall_1 = hit_1 / len(true_cits)
                    recall_5 = hit_5 / len(true_cits)
                    recall_20 = hit_20 / len(true_cits)

                    result = (pred, label, recall_1, recall_5, recall_20)
                    results.append(result)

                return_queue.put(results)

        # assign the documents to workers
        if self.verbose >= self.INFO:
            print(
                'Start predictions: assigning tasks to worker processes...', file=sys.stderr)

        list_workers, queue, return_queue = self._init_queue_and_workers(
            _evaluation_worker_process)
        task_cnt = 0

        for bva_id in bva_ids:
            citation_list = util.get_citations(bva_id, self.vocab)
            if not citation_list:
                continue

            serialized_args = pickle.dumps((citation_list, bva_id),
                                           protocol=pickle.HIGHEST_PROTOCOL)
            queue.put(serialized_args)
            task_cnt += 1

        for _ in list_workers:
            queue.put(None)  # end of queue

        # collect results from the workers
        if self.verbose >= self.INFO:
            print('Collecting worker results...', file=sys.stderr)

        results = []
        for _ in tqdm(range(task_cnt)):
            results.extend(return_queue.get())
        for worker in list_workers:
            worker.join()

        small_class_pred, small_class_label, recall_1_list, \
            recall_5_list, recall_20_list = zip(*results)

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

    def _init_queue_and_workers(self, worker_fn):
        queue = Queue()
        return_queue = Queue()
        num_workers = os.cpu_count()
        list_workers = []

        for _ in range(num_workers):
            process = Process(target=worker_fn,
                              args=(self, queue, return_queue))
            list_workers.append(process)
            process.start()
        return list_workers, queue, return_queue

    def _get_score(self, counter):
        if self.scoring == 'binary':
            for key in counter:
                counter[key] = 1
        return counter

    def _flatten(self, input_citations):
        input_dict = {}
        for citation in input_citations:
            assert type(citation) is list
            for cit in citation:
                cit = str(cit)
                input_dict[cit] = input_dict.get(cit, 0) + 1
        return input_dict

    def _scale(self, counter, weight):
        for key in counter:
            counter[key] *= weight
        return counter

    def _get_structured_cit_name(self, idx):
        name = self.vocab.citation_str_by_index(idx).split(', ')
        try:
            name[0] = name[0].split(': ')[1]
        except:
            pass
        if len(name) > 1:
            name = [name[1], name[0]]
        return pd.Series(name)

    def find_candidate_docs(self, input_dict):
        """Find candidate relevant documents given the input citations"""
        candidate_docs = collections.Counter()

        # compute dot product for candidate docs
        for cit in input_dict:
            inv_list = self.inv_lists.get(cit, {})
            docs = self._get_score(inv_list)
            if input_dict[cit] != 1:
                docs = self._scale(docs, input_dict[cit])
            candidate_docs.update(docs)

        problem_docs = set()

        # divide by L2-norm of the vectors
        for doc in candidate_docs:
            if self.scoring == 'tf':
                doc_vec_norm = self.fwd_idxes[doc]['tf_vec_norm']
            else:
                assert self.scoring == 'binary'
                doc_vec_norm = math.sqrt(len(self.fwd_idxes[doc]['counter']))

            if doc_vec_norm == 0:
                problem_docs.add(doc)
                continue
            candidate_docs[doc] /= doc_vec_norm

        for doc in problem_docs:
            del candidate_docs[doc]

        return candidate_docs

    def find_candidate_citations(self, relevant_docs):
        """Find candidate citations based on what the relevant documents have cited"""
        candidate_citations = collections.Counter()
        weight_sum = sum(doc[1] for doc in relevant_docs)

        for bva_id, weight in relevant_docs:
            fwd_idx = self.fwd_idxes.get(bva_id, {}).get('counter', {})
            fwd_idx = self._get_score(fwd_idx)
            candidate_citations.update(
                self._scale(fwd_idx, weight / weight_sum))

        return candidate_citations

    def pick_top_citations(self, candidate_citations, input_dict={}, readable=False):
        """Return the top citation recommendations"""
        # Filter out those already in the input list
        # Sort candidate recommendations by score in descending order
        # Take the top <recommendation_limit> recommendations
        for cit in input_dict:
            candidate_citations.pop(cit, None)
        top_citations = sorted(candidate_citations.items(),
                               key=lambda x: -x[1])[:self.recommendation_limit]
        top_citations_df = pd.DataFrame(top_citations, columns=['idx', 'score']) \
            .astype({'idx': 'int32'}).set_index('idx')

        # if readable, convert citation indices into their string representations
        if readable:
            top_citations_df = pd.concat([
                top_citations_df,
                top_citations_df.apply(
                    lambda row: self._get_structured_cit_name(row.name), axis=1,
                ).rename({0: 'name', 1: 'caption'}, axis=1)
            ], axis=1).fillna('N/A')

        return top_citations_df

    def find_recommendations(self, input_citations, readable=False):
        """Find citation recommendations for a list of input citations"""
        try:
            input_dict = self._get_score(self._flatten(input_citations))

            candidate_docs = self.find_candidate_docs(input_dict)

            relevant_docs = sorted(candidate_docs.items(),
                                   key=lambda x: -x[1])[:self.relevant_doc_limit]

            candidate_citations = self.find_candidate_citations(relevant_docs)

            top_citations_df = self.pick_top_citations(
                candidate_citations, input_dict, readable)

            return top_citations_df
        except:
            return None
