import dataset_build as db
import dataset_vocab as dv
import pickle
import copy
import functools
import argparse
import importlib as imp


cv_raw_fpath = '../../../data/bva/vocab/vocab_raw_v4.pkl'
cv_norm_fpath = '../../../data/bva/vocab/vocab_norm_v4.pkl'
cv_norm_min20_fpath = '../../../data/bva/vocab/vocab_norm_min20_v4.pkl'


def save_vocab(cv, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(cv, f)

def load_vocab(fpath):
    with open(fpath, 'rb') as f:
        cv = pickle.load(f)
    return cv


def build_vocab():
    print('building raw vocabulary')
    cv = db.make_raw_citation_vocab(db.preprocessed_dir, db.train_ids_fpath)
    print(cv.vocab_report())
    save_vocab(cv, cv_raw_fpath)
    print('normalizing vocabulary')
    cv_norm = db.normalize_vocabulary(cv, db.citation_dict_fpaths)
    print(cv_norm.vocab_report())
    save_vocab(cv_norm, cv_norm_fpath)
    print('thresholding vocabulary')
    cv_norm_min20 = copy.deepcopy(cv_norm)
    cv_norm_min20.reduce_sparse_to_unknown(20)
    save_vocab(cv_norm_min20, cv_norm_min20_fpath)
    print(cv_norm_min20.vocab_report())
    return cv, cv_norm, cv_norm_min20


def load_vocabs():
    print('loading raw vocabulary')
    cv = load_vocab(cv_raw_fpath)
    print('loading normalized vocabulary')
    cv_norm = load_vocab(cv_norm_fpath)
    print('loading normalized, thresholded vocabulary')
    cv_norm_min20 = load_vocab(cv_norm_min20_fpath)
    return cv, cv_norm, cv_norm_min20


def make_cache(vocab):
    print('caching preprocessed decisions')
    db.parallelize_cache_citation_indices('../../../data/bva/utils/updated_ids_all.txt',
            db.preprocessed_dir,
            '../../../data/bva/preprocessed-cached-v4/',
            vocab)
