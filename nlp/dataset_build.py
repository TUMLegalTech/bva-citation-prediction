import dataset_vocab as dv
from preprocessing import DataPreprocessor
from typing import List, Tuple, Dict, Any, Optional, Callable, Set
import re
import csv
import util
import pickle
import importlib as imp
import os
import json
import tqdm
import functools
from functools import lru_cache
import multiprocessing
import random


# paths to caselaw access metadata. can be found in shared Google Drive
citation_dict_fpaths = [
    '../../../data/bva/utils/cla_vet_app_metadata.jsonl',
    '../../../data/bva/utils/f3d.jsonl'
]
metadata_fpath = '../../../data/bva/utils/appeals_meta_wscraped.csv'
# path to folder with full dataset as txt files
case_coll_dir = '../../../data/all-bva-decisions/'
# folder to be used to store all preprocessed cases
preprocessed_dir = '../../../data/bva/preprocessed/'
# text file with training set ids. one id per line
train_ids_fpath = '../../../data/bva/utils/updated_train_ids.txt'
dev_ids_fpath = '../../../data/bva/utils/updated_dev_ids.txt'
train_plus_dev_ids_fpath = '../../../data/bva/utils/updated_train_plus_dev_ids.txt'
transform_log_fpath = '../../../data/bva/utils/vocab_transform_log'
# like above, but smaller for debugging
#train_ids_small_fpath = '../utils/train_data_ids_small.txt'


def preprocess_cases(case_coll_dir, preprocessed_dir, case_ids_fpath):
    '''preprocesses all cases whose ID is in the id file from the raw txt folder
    to a json in the preprocessed folder. If no ID file is passed, then all text
    files in the folder are preprocessed'''
    pp = DataPreprocessor()
    case_ids = util.load_case_ids_from_file(case_ids_fpath)
    print(f'loaded {len(case_ids)} case IDs')
    pp.process_partition_txt_files(case_coll_dir, preprocessed_dir, case_ids)


def make_raw_citation_vocab(preprocessed_dir, case_ids_fpath):
    '''builds a raw citation vocabulary object'''
    cv = dv.CitationVocabulary()
    #cv.load_from_case_dir(preprocessed_dir, case_id_fpath=train_id_fpath)
    cv.load_from_case_dir(preprocessed_dir, case_id_fpath=case_ids_fpath)
    return cv


def make_cla_processor(citation_dict_fpaths):
    cpr = dv.CLACitationProcessor()
    for cpath in citation_dict_fpaths:
        cpr.load_cla_citation_dictionary(cpath)
    return cpr


def normalize_vocabulary(cv, citation_dict_fpaths=None, cla_processor=None):
    '''produces a smaller citation vocabulary object based on a deep
    copy of the first one by applying normalization reports reduction
    statistics to the console'''
    if cla_processor:
        cpr = cla_processor
    else:
        cpr = dv.CLACitationProcessor()
        for cpath in citation_dict_fpaths:
            cpr.load_cla_citation_dictionary(cpath)
    citation_transform_fns = [dv.remove_trailing_parenth_years,
                              dv.remove_trailing_punctuation,
                              cpr.resolve_case_citation,
                              dv.split_statutory_citations]
    cvx = cv.duplicate()
    cvx.set_transform_fns(citation_transform_fns)
    cvx.cla_processor = cpr
    pre_size = len(cvx)
    cvx.transform_vocabulary(log_save_path=transform_log_fpath)
    post_size = len(cvx)
    print(f'reduced vocabulary from {pre_size} to {post_size}')
    return cvx


class CitationTransformLog:

    def __init__(self, log_pkl_path: str):
        with open(log_pkl_path, 'rb') as f:
            self.transforms = pickle.load(f)

    def filter_raw_by_regex_search(self, regex: str):
        return {k: v for k, v in self.transforms.items()
                if re.search(regex, k)}

    def filter_raw_by_regex_match(self, regex: str):
        return {k: v for k, v in self.transforms.items()
                if re.match(regex, k)}

    def filter_transformed_by_regex_search(self, regex: str) -> dict:
        filtered = {}
        for k, v in self.transforms.items():
            for vi in v:
                if re.search(regex, vi):
                    filtered[k] = v
        return filtered

    def filter_transformed_by_regex_match(self, regex: str) -> dict:
        filtered = {}
        for k, v in self.transforms.items():
            for vi in v:
                if re.match(regex, vi):
                    filtered[k] = v
        return filtered

    def export_transform_txt(self, fpath: str, transforms: dict = None):
        if not transforms:
            transforms = self.transforms
        with open(fpath, 'w') as f:
            for k, v in transforms.items():
                f.write(f'{k} => '+' / '.join(v)+'\n')


def vocab_freq_threshold_stats(cv, thresholds):
    stats = []
    for t in thresholds:
        cvx = cv.duplicate()
        cvx.reduce_sparse_to_unknown(t)
        stats.append(cvx.vocab_report())
    return stats


def cache_citation_indices(input_case_dir, 
                           output_case_dir, 
                           vocab, 
                           total_cit_count,
                           indices_cache,
                           num_cache_lookups,
                           fname):
    '''cache the citation_indices for a single file'''
    if not fname.endswith('.json'):
        fname += '.json'
    with open(os.path.join(input_case_dir, fname)) as file:
        data = json.load(file)
    citation_indices = []
    for ct in data['citation_texts']:
        total_cit_count.value += 1
        if indices_cache is not None: # careful: empty dict evals to False
            cached = indices_cache.get(ct, None)
            if cached:
                citation_indices.append(cached)
                num_cache_lookups.value += 1
            else:
                cis = vocab.citation_indices_from_raw(ct)
                citation_indices.append(cis)
                indices_cache[ct] = cis
        else:
            citation_indices.append(vocab.citation_indices_from_raw(ct))
    data['citation_indices'] = citation_indices
    with open(os.path.join(output_case_dir, fname), 'w') as file:
        file.write(json.dumps(data))


def parallelize_cache_citation_indices(case_ids_fpath,
                                       input_case_dir,
                                       output_case_dir,
                                       vocab,
                                       use_cache=True,
                                       num_threads=4,
                                       block_size=100,
                                       max_n=None):
    '''cache the citation_indices in parallel for files in input_case_dir whose
    id is in the provided list'''
    with open(case_ids_fpath, 'r') as file:
        case_ids = file.read().splitlines()
        if max_n:
            case_ids = case_ids[:max_n]
    print(f'found {len(case_ids)} case ids')
    manager = multiprocessing.Manager()
    if use_cache:
        indices_cache = manager.dict()
        print(f'setting up cache: {indices_cache}')
        num_cache_lookups = manager.Value('i', 0)
    else:
        indices_cache = None
        num_cache_lookups = None
    total_cit_count = manager.Value('i', 0)
    if indices_cache is None:
        print('cache not set up')
    else:
        print(f'cache set up')
    partial = functools.partial(cache_citation_indices,
                                input_case_dir,
                                output_case_dir,
                                vocab,
                                total_cit_count,
                                indices_cache,
                                num_cache_lookups)
    pool = multiprocessing.Pool(num_threads)
    list(tqdm.tqdm(pool.imap_unordered(partial, case_ids, block_size),
                   total=len(case_ids)))
    print('done caching')
    print(f'total citation count: {total_cit_count.value}')
    if indices_cache:
        print(f'# elements in cache: {len(indices_cache)}')
        print(f'# cache lookups: {num_cache_lookups.value}')
    else:
        print('cache not used or empty')
    
    
def cache_citation_indices_single_thread(case_ids_fpath,
                                         input_case_dir,
                                         output_case_dir,
                                         vocab):
    with open(case_ids_fpath, 'r') as file:
        case_ids = file.read().splitlines()
    print(f'found {len(case_ids)} case ids')
    fnames = [cid+'.json' for cid in case_ids]
    for fname in tqdm.tqdm(fnames):
        with open(os.path.join(input_case_dir, fname)) as file:
            data = json.load(file)
        citation_indices = []
        for ct in data['citation_texts']:
            citation_indices.append(vocab.citation_indices_from_raw(ct))
        data['citation_indices'] = citation_indices
        with open(os.path.join(output_case_dir, fname), 'w') as file:
            file.write(json.dumps(data))
            
            
def cache_normalization_check(cache_dir, case_ids_fpath, n, vocab, random_sample=True):
    with open(case_ids_fpath, 'r') as f:
        case_ids = f.read().splitlines()
    if random_sample:
        sample = random.sample(case_ids, n)
    else:
        sample = case_ids[:n]
    for case_id in sample:
        print(f'CASE: {case_id}')
        with open(os.path.join(cache_dir, case_id+'.json'), 'r') as f:
            data = json.load(f)
            cit_texts = data['citation_texts']
            for i in range(len(cit_texts)):
                print(cit_texts[i])
                normed = [vocab.citation_str_by_index(ci)
                          for ci in data['citation_indices'][i]]
                for norm in normed:
                    print(f'=> {norm}')
                print()
    


def citation_transformation_log_analysis(log_fpath, raw_vocab=None):
    '''analyse citation transformation analysis pickle file for proportions
    of usc, cfr, and case citations. quantify numbers and counts of case
    citations whose parties do not seem to match the parties in the CLA
    citation they have been normalized to.'''
    with open(log_fpath, 'rb') as f:
        log = pickle.load(f)
    print(f'total raw: {len(log)}')
    cfr = {}
    usc = {}
    case_presumed = {}
    for raw, transformed in log.items():
        if dv.usc_re.search(raw):
            usc[raw] = transformed
        elif dv.cfr_re.search(raw):
            cfr[raw] = transformed
        else:
            case_presumed[raw] = transformed
    case_no_party = {}
    case_w_party = {}
    for raw, transformed in case_presumed.items():
        if re.search(r'^\d', raw):
            case_no_party[raw] = transformed
        else:
            case_w_party[raw] = transformed
    case_matching_party = {}
    case_mismatch_party = {}
    party_re = re.compile(r'^(see|compare|citing|in|see,? e\.g\.,|under)?\s*(?P<parties>[^,\d]+),?\s+\d+\s+(vet|f\.?3r?d)')
    for raw, transformed in case_w_party.items():
        raw_lc = raw.lower()
        m = party_re.search(raw_lc)
        if m:
            parties = m.group('parties')
        else:
            parties = raw.split(',')[0]
        parties_clean = re.sub(r'\W', '', parties.lower())
        norm_clean = re.sub(r'\W', '', transformed[0].lower())
        if norm_clean.find(parties_clean) != -1:
            case_matching_party[raw] = transformed
        else:
            case_mismatch_party[raw] = transformed
    if raw_vocab:
        mismatch_counts = [raw_vocab.citation_counts[raw]
                           for raw in case_mismatch_party]
        case_mismatch_party_count = sum(mismatch_counts)
        print(f'- - raw/cla parties mismatch: {len(case_mismatch_party)}')
    print(f'usc by regex: {len(usc)}')
    print(f'cfr by regex: {len(cfr)}')
    print(f'presumed case by regex: {len(case_presumed)}')
    print(f'- presumed case without parties: {len(case_no_party)}')
    print(f'- presumed case with parties: {len(case_w_party)}')
    print(f'- - raw/cla parties match: {len(case_matching_party)}')
    print(f'- - raw/cla parties mismatch: {len(case_mismatch_party)}')
    if raw_vocab:
        print(f'- - raw/cla parties mismatch: {case_mismatch_party_count}')
        print('- - raw/cla parties proportion: '
              +str(case_mismatch_party_count / raw_vocab.total_count()))
    return {'usc': usc,
            'cfr': cfr,
            'case_presumed': case_presumed,
            'case_no_party': case_no_party,
            'case_w_party': case_w_party,
            'case_matching_party': case_matching_party,
            'case_mismatch_party': case_mismatch_party}



# cache the citation_indices in parallel for all files in input_case_dir
def assemble_broken_citation_data(case_ids_fpath,
                                  input_case_dir,
                                  metadata_fpath,
                                  broken_raw_norm_dict):
    broken_citations = []
    vlj_broken_counts = {}
    vlj_case_n = {}
    sa_broken_counts = {}
    sa_case_n = {}
    case_vlj_sa = {}
    print('loading metadata')
    with open(metadata_fpath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_vlj_sa[row['tiread2']] = (row['vlj_name'], row['sa_name'])
    print('gathering broken citations')
    with open(case_ids_fpath, 'r') as f:
        case_ids = f.read().splitlines()
    n = 0
    for case_id in tqdm.tqdm(case_ids):
        fname = case_id+'.json'
        with open(os.path.join(input_case_dir, fname)) as file:
            data = json.load(file)
        citations = data['citation_texts']
        vlj = case_vlj_sa[case_id][0]
        sa = case_vlj_sa[case_id][1]
        vlj_case_n[vlj] = 1 + vlj_case_n.get(vlj, 0)
        sa_case_n[sa] = 1 + sa_case_n.get(sa, 0)
        for cit in citations:
            n += 1
            cla_norm = broken_raw_norm_dict.get(cit, None)
            if cla_norm:
                vlj_broken_counts[vlj] = 1 + vlj_broken_counts.get(vlj, 0)
                sa_broken_counts[sa] = 1 + sa_broken_counts.get(sa, 0)
                broken_citations.append({'case_id': case_id,
                                         'raw': cit,
                                         'cla': cla_norm[0],
                                         'vlj': vlj,
                                         'sa': sa})
    print(f'{n} citations processed')
    return {'broken_citations': broken_citations,
            'vlj_broken_counts': vlj_broken_counts,
            'sa_broken_counts': sa_broken_counts,
            'vlj_case_n': vlj_case_n,
            'sa_case_n': sa_case_n}
            

def export_broken_citation_stats(broken_citation_data,
                                 export_dir = '../../../data/bva/utils',
                                 sep='|'):
    with open(os.path.join(export_dir, 'broken_citations_train.csv'), 'w') as f:
        f.write(sep.join(['case_id', 'raw', 'cla', 'vlj', 'sa'])+'\n')
        for b in broken_citation_data['broken_citations']:
            row = [b['case_id'], b['raw'], b['cla'], b['vlj'], b['sa']]
            f.write(sep.join(row)+'\n')
    with open(os.path.join(export_dir, 'broken_citations_train_vlj.csv'), 'w') as f:
        f.write(sep.join(['vlj', '#broken', '#cases', 'errors_per_case'])+'\n')
        for vlj, count in broken_citation_data['vlj_broken_counts'].items():
            case_n = broken_citation_data['vlj_case_n'][vlj]
            errors_per_case = count / case_n
            row = [vlj, str(count), str(case_n), str(errors_per_case)]
            f.write(sep.join(row) +'\n') 
    with open(os.path.join(export_dir, 'broken_citations_train_sa.csv'), 'w') as f:
        f.write(sep.join(['sa', '#broken', '#cases', 'errors_per_case'])+'\n')
        for sa, count in broken_citation_data['sa_broken_counts'].items():
            case_n = broken_citation_data['sa_case_n'][sa]
            errors_per_case = count / case_n
            row = [sa, str(count), str(case_n), str(errors_per_case)]
            f.write(sep.join(row) +'\n') 

    

