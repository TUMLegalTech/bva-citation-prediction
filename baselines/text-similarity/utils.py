from datetime import datetime
import pickle
import time
from functools import reduce
import os
import pymongo
import glob
import pandas as pd
import math
from filepaths import *

# Load hyperparameters for experiments
def load_hyperparams(fpath):
    global_params = {}
    model_params = []
    i=0
    with open(fpath, "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        if line.startswith("#") or len(line) == 0:
            continue

        p = line.split(",")
        if i == 0:
            global_params["vocab_size"] = int(p[0])
            global_params["cit_as_vocab"] = bool(int(p[1]))
            global_params["target"] = p[2]
            global_params["text_span"] = int(p[3])
            global_params["forecast_window"] = int(p[4])
            global_params["klist"] = [int(k) for k in p[5].split("|")]
            global_params["max_cit_count"] = int(p[6])
            global_params["max_docs"] = int(p[7])

        else:
            params = {}
            params["model"] = p[0]
            params["features"] = p[1].split("|")
            params["n_svm"] = int(p[2])
            model_params.append(params)
        i+=1
    return global_params, model_params


# Load case ids
def load_case_ids():
    with open(metadata_ids_fpath, "r") as f:
        meta_ids = set(f.read().splitlines())

    id_set = set()
    l = []
    for fname in case_id_fpaths:
        with open(fname, "r") as f:
            case_ids = f.read().splitlines()
        new_ids = set(case_ids) - id_set
        case_ids = list(meta_ids.intersection(new_ids))
        l.append(case_ids)
        id_set.update(case_ids)

    print(f"Loaded {', '.join([str(len(x)) for x in l])} case ids.")
    return l


def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


# k: number of folds
# i: which fold we are in
def load_kfold_ids(case_ids, k, i):
    n = len(case_ids)
    start_idx = math.floor((n/k)*i)
    if i == k-1:
        end_idx = n
    else:
        end_idx = math.floor((n/k)*(i+1))
    test_ids = case_ids[start_idx:end_idx]
    print(f"Partition {i+1}-{k} | Test IDs from {start_idx} to {end_idx}")
    train_ids = list(set(case_ids) - set(test_ids))
    return train_ids, test_ids


def load_cit_vocab(fpath):
    with open(fpath, "rb") as f:
        return pickle.load(f)


def get_datetime():
    return datetime.today().strftime('%Y-%m-%d-%H-%M')


class Timer:
    def __init__(self):
        self.start = time.time()
        self.checkpoints = 0
    
    def check(self):
        self.checkpoints += 1
        current_time = time.time()
        avg_time_elapsed = (current_time - self.start) / self.checkpoints
        return(f"Average time elapsed: {avg_time_elapsed:.2f}s")

    def end(self):
        time_elapsed = time.time() - self.start
        return(f"Total time elapsed: {time_elapsed:.2f}s")


def clear_dir(directory):
    files = glob.glob(os.path.join(directory, "*"))
    for f in files:
        os.remove(f)


# Connect to luima mongodb
def connect_db(db_details):
    db = pymongo.MongoClient(host=db_details["DB_HOST"], 
                             port=db_details["DB_PORT"])[db_details["DB_DATABASE"]]
    db.authenticate(db_details["DB_USER"], db_details["DB_PASSWORD"])
    return db


# Utility code to encode metadata codes
# Requires access to luima mongodb
class Dictionary():
    def __init__(self, db):
        self.issue_dict = db.code_dictionary.find_one({"name": "issue"})['dict']
        self.issdc_dict = db.code_dictionary.find_one({"name": "issdc"})['dict']
        self.cvdisp_dict = db.code_dictionary.find_one({"name": "cvdisp"})['dict']
        self.issue_levels = ["issprog", "isscode", "isslev1", "isslev2", "isslev3"]

    # Given issue codes, return the relevant label
    def label_issue(self, **kwargs):
        list_conditions = []
        try:
            for var in self.issue_levels:
                if var in kwargs:
                    list_conditions.append(var)
                    list_conditions.append(str(kwargs[var]))
            label = reduce(dict.__getitem__, list_conditions + ["label"], 
                            self.issue_dict)
        except:
            label = ''
        return label

    # Given issdc code, return label
    def label_issdc(self, issdc):
        try:
            return self.issdc_dict["issdc"][str(issdc)]['label']
        except:
            return ''

    # Given cvdisp code, return label
    def label_cvdisp(self, cvdisp):
        try:
            return self.cvdisp_dict["cvdisp"][str(cvdisp)]['label']
        except:
            return ''

# Get the issarea label
def get_issarea(progcode, isscode, isslev1, diagcode):
    if progcode != 2:
        return 0
    else:
        if isscode == 8:
            return 2
        elif isscode == 9:
            return 3
        elif isscode == 17:
            return 4
        elif isscode == 12:
            if isslev1 != 4:
                return 11
            elif isslev1 == 4:
                if diagcode == 5:
                    return 12
                elif diagcode == 6:
                    return 13
                elif diagcode == 7:
                    return 14
                elif diagcode == 8:
                    return 15
                elif diagcode == 9:
                    return 16
        elif isscode == 15:
            if isslev1 != 3:
                return 5
            elif isslev1 == 3:
                if diagcode == 5:
                    return 6
                if diagcode == 6:
                    return 7
                if diagcode == 7:
                    return 8
                if diagcode == 8:
                    return 9
                if diagcode == 9:
                    return 10
        else:
            return 1
    assert False, "Failed to assign metadata class to this example"

