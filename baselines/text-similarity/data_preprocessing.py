import sys
import dataset_vocab as dv
from preprocessing import DataPreprocessor
import pickle
import importlib as imp
import os
import json
from multiprocessing import Pool
from utils import connect_db, get_issarea, load_case_ids, Timer, get_issarea
from db_config import db_details
from tqdm import tqdm
import pandas as pd
from typing import List
from filepaths import *

UNKNOWN_CITATION_TOKEN = 'UNKCIT'

def preprocess_cases(case_coll_dir, preprocessed_dir, case_ids):
    '''preprocesses all cases whose ID is in the id file from the raw txt folder
    to a json in the preprocessed folder'''
    timer = Timer()
    pp = DataPreprocessor()
    pp.process_partition_txt_files(case_coll_dir, preprocessed_dir, case_ids)
    print(f"Finished preprocessing. | {timer.end()}")


def make_raw_citation_vocab(preprocessed_dir, case_ids):
    '''builds a raw citation vocabulary object'''
    timer = Timer()
    cv = dv.CitationVocabulary()
    cv.load_from_case_dir(preprocessed_dir, case_ids)
    print(f"Made raw vocab. | {timer.end()}")
    return cv


def normalize_vocabulary(cv, citation_dict_fpaths=None, cla_processor=None):
    '''produces a smaller citation vocabulary object based on a deep
    copy of the first one by applying normalization reports reduction
    statistics to the console'''
    timer = Timer()
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
    pre_size = len(cvx)
    cvx.transform_vocabulary(log_save_path='transform_log.pkl')
    post_size = len(cvx)
    print(f'reduced vocabulary from {pre_size} to {post_size} | {timer.end()}')
    cvx.vocab_report()
    return cvx


class CitationReplacer:

    def __init__(self, cit_vocabulary, preprocessed_dir,
                 postprocessed_dir, citation_place_holder):
        self.cit_vocabulary = cit_vocabulary
        self.preprocessed_dir = preprocessed_dir
        self.postprocessed_dir = postprocessed_dir
        self.citation_place_holder = citation_place_holder
        self.citation_list = {k:i for i,k in 
                            enumerate(cit_vocabulary.citation_counts.keys())}

    def citation_index(self, cit: str) -> int:
        '''Drop-in replacement for citation_index'''
        if cit in self.cit_vocabulary.citation_counts:
            return self.citation_list[cit]
        else:
            return self.citation_list[UNKNOWN_CITATION_TOKEN]

    def citation_indices_from_raw(self, cit: str) -> List[int]:
        '''Drop-in replacement for citation_indices_from_raw'''
        resolved = self.cit_vocabulary.transform_citation(cit)
        return [self.citation_index(r) for r in resolved]

    # For use by write_processed_case_text
    def replace_cit_and_write(self, fname):
        read_fpath = os.path.join(self.preprocessed_dir, fname)
        with open(read_fpath) as f:
            data = json.load(f)
        citation_indices = [self.citation_indices_from_raw(cit) 
                for cit in data['citation_texts']]
        txt = data['txt']
        for cis in citation_indices:
            cit_string = ' '.join([f'@cit{ci}@' for ci in cis])
            txt = txt.replace(self.citation_place_holder, cit_string, 1)
        write_fpath = os.path.join(self.postprocessed_dir, fname)
        with open(write_fpath, "w") as f:
            f.write(txt)

    # With the CitationVocabulary object, replace all occurences of
    # citations with the @citN@ token
    def write_processed_case_text(self):
        timer = Timer()
        fnames = [fn for fn in os.listdir(preprocessed_dir)
                if fn.lower().endswith('.json')] 
        pool = Pool(os.cpu_count())
        pool.map(self.replace_cit_and_write, fnames)
        print(f"Finished writing {len(fnames)} docs. | {timer.end()}")


class MetadataBuilder():

    def __init__(self, db, metadata_fpath, case_ids):
        self.db = db
        self.metadata_fpath = metadata_fpath
        self.case_ids = case_ids
        self.metadata_dict = {}

    def list_collections(self):
        for name in self.db.list_collection_names():
            print(f"{name}: {self.db[name].estimated_document_count():,}")

    def convert_to_int(self, col):
        return [int(x) if x != "NA" else -1 for x in col]

    def get_diag_code(self, isslev2):
        return [int(str(x)[0]) if x != "NA" else -1 for x in isslev2]

    # Convert VLJ names to index in descending freq order
    # those with 5 or less cases will be grouped together
    def get_vlj_code(self, vlj):
        vlj_counts = vlj.value_counts()
        vlj_counts = vlj_counts[vlj_counts > 5]
        vlj_dict = {k:i for i,k in enumerate(vlj_counts.keys())}
        unk = len(vlj_dict)
        return [vlj_dict[v] if v in vlj_dict else unk for v in vlj]

    def build_metadata(self):
        print("Creating Metadata dictionary...")
        timer = Timer()
        df_meta = pd.DataFrame(list(self.db.appeals_meta_wscraped.find(
                               {"tiread2": {"$in": self.case_ids}},
                               {"_id": 0})))
        df_meta["isscode"] = self.convert_to_int(df_meta["isscode"])
        df_meta["issprog"] = self.convert_to_int(df_meta["issprog"])
        df_meta["isslev1"] = self.convert_to_int(df_meta["isslev1"])
        df_meta["vlj"] = self.get_vlj_code(df_meta["vlj_name"])
        df_meta["diagcode"] = self.get_diag_code(df_meta["isslev2"])
        df_meta["year"] = df_meta["imgadtm"].str[0:4].astype(int)
        df_meta["bva_id"] = df_meta["tiread2"]
        df_meta["issarea"] = df_meta.apply(lambda r: 
                            get_issarea(r["issprog"], r["isscode"],
                                r["isslev1"], r["diagcode"]), axis=1)
        df_meta = df_meta[["bva_id", "year", "issarea", "vlj"]]
        df_meta.drop_duplicates(subset="bva_id", keep='first', inplace=True)
        df_meta.to_csv(self.metadata_fpath, index=False)
        print(f"Saved {df_meta.shape[0]} rows of metadata.")
        print(f"Data from {min(df_meta['year'])} to {max(df_meta['year'])}.")
        print(timer.end())


if __name__ == "__main__":

    # Preprocess cases
    train_ids, dev_ids, test_ids = load_case_ids()
    preprocess_cases(case_coll_dir, preprocessed_dir,
                        train_ids + dev_ids + test_ids)
    
    # Make Raw Vocab and pickle it
    cv = make_raw_citation_vocab(preprocessed_dir, train_ids+dev_ids)
    with open(raw_vocab_fpath, 'wb') as f:
        pickle.dump(cv, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(raw_vocab_fpath, 'rb') as f:
        cv = pickle.load(f)
    
    # Make Reduced Vocab and pickle it
    cvx = normalize_vocabulary(cv, citation_dict_fpaths) 
    with open(reduced_vocab_fpath, 'wb') as f:
        pickle.dump(cvx, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(reduced_vocab_fpath, 'rb') as f:
        cvx = pickle.load(f)

    # Threshold
    cvx.reduce_sparse_to_unknown(20)
    with open(thresholded_vocab_fpath, "wb") as f:
        pickle.dump(cvx, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(thresholded_vocab_fpath, 'rb') as f:
        cvx = pickle.load(f)
    
    # Process texts by replacing citations
    cit_replacer = CitationReplacer(cvx, preprocessed_dir,
                     postprocessed_dir, '@cit@')
    cit_replacer.write_processed_case_text()

    # Prepare metadata
    db = connect_db(db_details)
    metadata_builder = MetadataBuilder(db, metadata_fpath,
                                       train_ids+dev_ids+test_ids)
    metadata_builder.list_collections()
    metadata_builder.build_metadata()

    # Store list of case_ids that can be found in metadata
    metadata_ids = list(db.appeals_meta_wscraped.find(
        {},
        {"_id": 0, "tiread2": 1}))
    metadata_ids = [d['tiread2'] for d in metadata_ids]
    with open(metadata_ids_fpath, "w") as f:
        for case_id in metadata_ids:
            f.write(case_id + "\n")



