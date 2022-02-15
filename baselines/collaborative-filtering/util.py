import pickle
import pandas as pd
import json
import re

"""
Data file paths
"""
# TODO Change path to your bva-data
ROOT_PATH = '<path>/data/bva-data/'

VOCAB_PATH = ROOT_PATH + 'vocab_norm_min20_v4.pkl'
INV_LIST_PATH = ROOT_PATH + 'cf_inverted_list.json'
FWD_INDEX_PATH = ROOT_PATH + 'cf_forward_index.json'

METADATA_PATH = ROOT_PATH + 'metadata.csv'
METADATA_FEATURE_PATH = ROOT_PATH + 'feature_{}_cit_idx.pkl'

PROCESSED_CASES = ROOT_PATH + 'preprocessed-cached-v4/'

DATA_SPLIT = {
    'train': 'data_split/train_without_dev_ids.txt',
    'dev': 'data_split/dev_data_ids.txt',
    'test': 'data_split/test_data_ids.txt',
}

"""
Utility functions
"""
def load_case_ids_from_file(partition_ids_fpath):
    '''load a list of case ids from a text file with one id per line'''
    partition_ids = set()
    with open(partition_ids_fpath) as f:
        for line in f:
            partition_ids.add(line.strip())
    return partition_ids

# returns the vocabulary object
def get_vocab():
    vocab = pickle.load(open(VOCAB_PATH, 'rb'))
    return vocab

# returns a list of BVA document IDs in the dataset
def get_dataset(dataset_name):
    assert dataset_name in ['train', 'dev', 'test']
    dataset_file = DATA_SPLIT[dataset_name]

    with open(dataset_file, 'r') as id_file:
        bva_ids = [bva_id.strip() for bva_id in id_file.readlines()]
    return bva_ids

# returns the inverted index as a dict
# structure: citation_idx -> {bva_id -> term_frequency}
def get_inverted_list():
    with open(INV_LIST_PATH, 'r') as f:
        inv_list = json.load(f)
    return inv_list

# returns the forward index as a dict
# structure: bva_id -> {
#     citations -> citation_list,
#     counter -> tf_vector,
#     tf_vec_norm -> l2_norm_of_tf_vector
# }
def get_forward_index():
    with open(FWD_INDEX_PATH, 'r') as f:
        fwd_index = json.load(f)
    return fwd_index

# returns document metadata as a pandas dataframe
def get_metadata():
    return pd.read_csv(METADATA_PATH, dtype={'bva_id': str}).set_index('bva_id')

# returns metadata features as a dict
def get_metadata_features(metadata_names):
    assert set(metadata_names) <= {'year', 'issarea', 'vlj'}
    metadata_features = {}
    for name in metadata_names:
        metadata_features[name] = pickle.load(
            open(METADATA_FEATURE_PATH.format(name), 'rb'))
    return metadata_features

# returns the list of citations in a document
# example: [[10], [25, 28, 30], [134]]
def get_citations(bva_id, vocab):
    case_path = PROCESSED_CASES + '{}.json'.format(bva_id)
    try:
        with open(case_path, 'r') as f:
            case_dict = json.load(f)
    except:
        return None

    return case_dict['citation_indices']
