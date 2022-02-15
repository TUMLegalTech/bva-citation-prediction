"""
Collaborative-filtering model main routine
"""
import argparse
import pandas as pd
import os

from model import CollabFilter


def parse_args():
    parser = argparse.ArgumentParser()

    # Routine related arguments
    parser.add_argument("-m", "--metadata", nargs='+', default=[],
                        help='Which metadata to add. Must be a subset of {year, issarea, vlj}.')
    parser.add_argument("--sample_only", action='store_true',
                        help='Only do generating pairwise data from the documents.')
    parser.add_argument("--train_eval_only", action='store_true',
                        help='Only do training and evaluation, with pre-selected samples.')
    parser.add_argument("-r", "--run_id", default=None,
                        help='Suffix for the saved files to tell different runs apart.')
    parser.add_argument("-v", "--verbose", default='info', choices=['info', 'warn', 'error'],
                        help='Verbosity level. Should be one of info, warn, and error.')

    # Evaluation related arguments
    parser.add_argument("-t", "--n_test_docs", type=int, default=None,
                        help='How many documents to evaluate on. If None, use the whole test set.')
    parser.add_argument("-n", "--n_folds", type=int, default=1,
                        help='How many folds to divide the set of test documents into.')
    parser.add_argument("--eval_result_folder", default='results',
                        help='Path where evaluation results are saved.')

    # Training related arguments
    parser.add_argument("-d", "--n_sample_docs", type=int, default=100,
                        help='How many documents to sample pairwise training data from.')
    parser.add_argument("-c", "--svm_c_param", type=float, default=1.0,
                        help='Penalty parameter C for soft-margin SVM.')
    parser.add_argument("--svm_model_folder", default='results',
                        help='Path where the SVM weights are saved.')

    # Recommender related arguments
    parser.add_argument("--relevant_doc_limit", type=int, default=50,
                        help='How many relevant documents to find for the input citations.')
    parser.add_argument("--recommendation_limit", type=int, default=50,
                        help='How many citations to put into the recommendation list.')
    parser.add_argument("--scoring", default='binary', choices=['binary', 'tf'],
                        help='The scoring function in the user-item matrix. Either binary or tf.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # parse the command line arguments
    args = parse_args()
    print(args)

    # create a collaborative filtering model object
    model = CollabFilter(relevant_doc_limit=args.relevant_doc_limit,
                         recommendation_limit=args.recommendation_limit,
                         scoring=args.scoring,
                         metadata_names=args.metadata,
                         svm_model_folder=args.svm_model_folder,
                         eval_result_folder=args.eval_result_folder,
                         run_id=args.run_id,
                         verbose=args.verbose)

    do_complete_routine = not args.sample_only and not args.train_eval_only

    # generate pairwise training data
    if args.sample_only or do_complete_routine:
        model.make_pairwise_data(n_documents=args.n_sample_docs)

    # train and evaluate the model
    if args.train_eval_only or do_complete_routine:
        model.train_svm(C=args.svm_c_param)
        small_df, full_df = model.predict(
            n_test_docs=args.n_test_docs, n_folds=args.n_folds)

