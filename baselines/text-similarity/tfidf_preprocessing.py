from tokenizer import TfidfTokenizer
from model import TfidfModel
from utils import load_case_ids, load_kfold_ids, get_datetime
import os
import random
from filepaths import *
import pickle

if __name__ == "__main__":
    train_ids, dev_ids, test_ids = load_case_ids()

    # Train the tokenizer
    with open(thresholded_vocab_fpath, "rb") as f:
        cvx = pickle.load(f)
        print(f"Loaded vocab of {len(cvx)} citations.")
    tokenizer = TfidfTokenizer(postprocessed_dir, 
                               tokenizer_fpath, 
                               encoded_txt_fpath,
                               cvx, 
                               train_ids,
                               dev_ids,
                               test_ids)
    tokenizer.save_tfidf(min_df=10)
    tokenizer.save_vocab()
    tokenizer.save()
    tokenizer.open_h5file()
    tokenizer.encode_files()
    tokenizer.close_h5file()

    # Make metadata features
    model = TfidfModel(encoded_txt_fpath,
                       tokenizer_fpath,
                       text_sim_dir,
                       log_dir,
                       metadata_fpath)

    model.set_train_ids(train_ids)
    model.load_tokenizer()
    model.load_data_to_memory()
    
    for target in ["cit_idx", "cit_class"]:
        model.set_target(target)
        model.load_metadata()
        model.compute_cit_locations()
        model.make_feature("base")
        model.make_feature("issarea")
        model.make_feature("year")
        model.make_feature("vlj")

