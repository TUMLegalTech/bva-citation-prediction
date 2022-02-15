from model import TfidfModel
from tokenizer import TfidfTokenizer
from utils import *
import os
from datetime import datetime
import pickle
import random
import argparse
from sklearn.model_selection import train_test_split
from filepaths import *

def main(model, h, train_ids, dev_ids, test_ids, K):
    print(h)
    if debug:
        h["n_svm"] = 100

    # Set hyperparameters
    model.set_model(h["model"])
    model.set_features(h["features"])
    model.set_pred_ids(dev_ids)
    model.train_model(h["n_svm"])

    # K Fold Testing
    for i in range(K):
        _, pred_ids = load_kfold_ids(test_ids, K, i)
        print(f"Test set: {len(pred_ids)}")
        model.set_pred_ids(pred_ids)
        model.set_partition(K, i)
        metrics = model.predict_cases()
        model.log_metrics(metrics, h)
                        

if __name__ == '__main__':
    global debug
    parser = argparse.ArgumentParser(description='Tfidf Program')
    parser.add_argument('-d', '--debug', default=False, action='store_true')
    parser.add_argument('-params')
    args = vars(parser.parse_args())
    debug = args['debug']
    hyperparams_fpath = args['params']


    # Load train, dev, test ids
    train_ids, dev_ids, test_ids = load_case_ids()
    global_params, model_params = load_hyperparams(hyperparams_fpath)

    # Debug mode
    random.seed(42)
    if debug:
        train_ids = random.sample(train_ids, 2000)
        dev_ids = random.sample(dev_ids, 1000)
        test_ids = random.sample(test_ids, 200)
        print(f"Dev set: {len(set(dev_ids) - set(train_ids))}")

    # Init Model
    model = TfidfModel(encoded_txt_fpath,
                       tokenizer_fpath,
                       text_sim_dir,
                       log_dir,
                       metadata_fpath,
                       debug=debug)
    model.load_data_to_memory()
    model.load_tokenizer()
    model.load_metadata()

    # Pretrain
    model.set_text_span(global_params["text_span"])
    model.set_target(global_params["target"])
    model.set_cit_as_vocab(global_params["cit_as_vocab"])
    model.set_vocab_size(global_params["vocab_size"])
    model.set_klist(global_params["klist"])
    model.set_max_docs(global_params["max_docs"])
    model.set_train_ids(train_ids)
    model.set_dev_ids(dev_ids)
    model.set_max_cit_count(global_params["max_cit_count"])
    model.compute_tfidf_stage2()

    # Run Program
    for i, h in enumerate(model_params):
        main(model, h, train_ids, dev_ids, test_ids, K=6)

    # Close
    model.clean_up()
