# Insert bva-capstone/nlp folder in sys.path to allow relative imports
import sys
sys.path.insert(1, './../../nlp')

from pdb import set_trace
import os
import json
import glob
import pickle
import torch
import numpy as np
np.seterr(all='raise')
from datetime import datetime
import h5py
import time
import random
from tqdm import tqdm
from scipy import sparse
from utils import *
from multiprocessing import SimpleQueue, Process
from sklearn import svm
import pandas as pd

normal_case_prefix = 'NJ: '
normal_code_prefix = 'NC: '
normal_reg_prefix = 'NR: '

################################################################
# Main Class for Tfidf
################################################################
class TfidfModel:

    def __init__(self,
                 encoded_txt_fpath, 
                 tokenizer_fpath, 
                 text_sim_dir,
                 log_dir,
                 metadata_fpath,
                 max_docs=10,
                 max_cit_count=10,
                 text_span=100,
                 vocab_size=5000,
                 debug=False):
        self.encoded_txt_fpath = encoded_txt_fpath
        self.tokenizer_fpath = tokenizer_fpath
        self.text_sim_dir = text_sim_dir
        self.log_fpath = os.path.join(log_dir, f"{get_datetime()}_logs.txt")
        self.metadata_fpath = metadata_fpath
        self.max_docs = max_docs
        self.max_cit_count = max_cit_count
        self.text_span = text_span
        self.vocab_size = vocab_size
        self.train_data = np.zeros(1)
        self.train_idx = np.zeros(1)
        self.train_lengths = np.zeros(1)
        self.debug = debug

    ################################################################
    # PREPROCESSING
    ################################################################

    # Load all the text data into memory (~5Gb)
    def load_data_to_memory(self):
        self.open_encoded_texts()
        self.all_data = self.encoded_texts

    def clean_up(self):
        self.close_encoded_texts()


    # Make features used by the svmRank algorithm later
    # If col_name = "base", it just does a simple count of cit proportions.
    # Otherwise, it counts cit proportions conditioned on that particular
    # column (e.g. based on "year").
    def make_feature(self, col_name):

        def _worker_process(self, queue, return_queue):
            for serialized_txt in iter(queue.get, None):
                encoded_txt, case_id = pickle.loads(serialized_txt)
                if col_name == "base":
                    v = 0
                else:
                    v = self.metadata_dict[case_id][col_name]
                cit_indices = np.where((encoded_txt < self.num_cits) &
                                       (encoded_txt > 0))[0]
                for ci in cit_indices:
                    cit = encoded_txt[ci]
                    cit = self.convert_citation_by_target(cit)
                    self.d1[v][cit] += 1
                    self.d2[v] += 1
            return_queue.put((self.d1, self.d2))

        def _combine_metrics(self, d1, d2, class_values):
            for v in class_values:
                self.d2[v] += d2[v]
                for cit in self.cit_targets:
                    self.d1[v][cit] += d1[v][cit]

        def _clear_metrics(self):
            self.d1 = None
            self.d2 = None

        def _init(self, class_values):
            d1 = {} # Store counts at class+cit level
            d2 = {} # Store counts at class level
            for v in class_values:
                d1[v] = {}
                d2[v] = 0
                for cit in self.cit_targets:
                    d1[v][cit] = 0
            return (d1, d2)

        # Init
        print(f"Making Feature {col_name}...")
        class_values = self.get_unique_values(col_name)
        n = len(class_values)
        self.d1, self.d2 = _init(self, class_values)

        list_workers, queue, return_queue =\
                self.init_queue_and_workers(_worker_process)
        
        # Count
        for train_id in tqdm(self.train_ids):
            encoded_txt = self.all_data[train_id]
            serialized_txt = pickle.dumps((encoded_txt, train_id), 
                              protocol=pickle.HIGHEST_PROTOCOL)
            queue.put(serialized_txt)
        for _ in list_workers: 
            queue.put(None)
        dlist = []
        for w in list_workers:
            dlist.append(return_queue.get())
        for w in list_workers:
            w.join()

        while len(dlist) > 0:
            d1, d2 = dlist.pop()
            _combine_metrics(self, d1, d2, class_values)
        
        # Avoid zeros
        for v in class_values:
            self.d2[v] += 1
            for cit in self.cit_targets:
                self.d1[v][cit] += round(1/n, 3)

        # Normalize
        for v in class_values:
            for cit in self.cit_targets:
                self.d1[v][cit] /= self.d2[v]

        if col_name == "base":
            self.d1 = self.d1[0]

        # Write
        feature_name = f"feature_{col_name}_{self.target}.pkl"
        feature_fpath = os.path.join(self.text_sim_dir, feature_name)
        with open(feature_fpath, "wb") as f:
            pickle.dump(self.d1, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.close_encoded_texts()

    ################################################################
    # HELPERS
    ################################################################

    def load_tokenizer(self):
        with open(self.tokenizer_fpath, "rb") as f:
            self.tokenizer = pickle.load(f)
        self.num_cits = self.tokenizer.num_cits
        self.num_vocab = len(self.tokenizer.words)
        self.init_citation_converter()

    def open_encoded_texts(self):
        self.encoded_texts = h5py.File(self.encoded_txt_fpath, "r")

    def close_encoded_texts(self):
        self.encoded_texts.close()

    def compute_tfidf_from_context(self, context):
        # Adjust context
        context = context[(context >= self.vocab_start) &
                          (context < self.vocab_end)]
        context -= self.vocab_adjust

        tf = np.bincount(context)
        tf = np.pad(tf, (0, self.vocab_size-len(tf)),
                mode="constant")
        tfidf = tf * self.idf
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm
            return sparse.csr_matrix(tfidf)
        else:
            return None

    def load_metadata(self):                            
        dtypes = {"bva_id": str}
        df = pd.read_csv(self.metadata_fpath, dtype=dtypes)
        self.metadata_dict = df.set_index('bva_id').T.to_dict()

    def get_unique_values(self, col_name):
        # Base feature just counts proportion
        if col_name == "base":
            return [0]         
        else:
            dtypes = {"bva_id": str}
            df = pd.read_csv(self.metadata_fpath, dtype=dtypes)
            values = sorted(df[col_name].unique().tolist())
            return values

    # Helper to start a Queue
    def init_queue_and_workers(self, worker_fn):
        queue = SimpleQueue()
        return_queue = SimpleQueue()
        num_workers = os.cpu_count()
        list_workers = []
        for _ in range(num_workers):
            process = Process(target=worker_fn, 
                              args=(self, queue, return_queue))
            list_workers.append(process)
            process.start()
        return (list_workers, queue, return_queue)

    ################################################################
    # SETTERS
    ################################################################

    # target: either "cit_idx", "cit_class" or "binary"
    def set_target(self, target):
        self.target = target
        if self.target == "cit_idx":
            self.cit_targets = range(1, self.num_cits)
        elif self.target == "cit_class":
            self.cit_targets = range(2, 5)

        # Set path to citation location file
        self.citloc_fpath = os.path.join(self.text_sim_dir,
                            f"citloc_{self.target}.json")

    def set_max_docs(self, max_docs):
        self.max_docs = max_docs

    def set_max_cit_count(self, max_cit_count):
        self.max_cit_count = max_cit_count

    def set_text_span(self, text_span):
        self.text_span = text_span

    def set_klist(self, klist):
        self.klist = klist

    # Hnadling the Vocabulary is a little tricky. 
    # Here is what each index in the encoded_txt represents:
    #   0:            <unk> token
    #   1-20607:      Citations
    #   20608 onward: Normal vocabulary, ordered from most 
    #                 frequently occurring to least
    # If vocab_size is set, the normal vocabulary is chopped off based
    # on the specified vocab_size.
    # If vocab_size = -1, then the maximum vocabulary size is used.
    #
    # The parameters set here are used in compute_tfidf_from_context:
    #   self.vocab_start:  Which index to start using as vocab.
    #   self.vocab_end:    Which index to stop using as vocab.
    #   self.vocab_adjust: How much to subtract so that the vocab
    #                      will start from 0.
    def set_vocab_size(self, vocab_size):
        if vocab_size <= 0:
            self.vocab_size = self.num_vocab
        else:
            self.vocab_size = vocab_size

        if self.cit_as_vocab:
            # start inclusive, end exclusive
            self.vocab_start = 1 # 0 is <unk>
            self.vocab_end = self.num_cits + self.vocab_size
            self.vocab_size += self.num_cits-1
            self.vocab_adjust = 1
        else:
            self.vocab_start = self.num_cits
            self.vocab_end = self.num_cits + self.vocab_size
            self.vocab_adjust = self.num_cits

        # idf does not have <unk>, so -1
        self.idf = self.tokenizer.idf[(self.vocab_start-1):(self.vocab_end-1)]

    def set_pred_ids(self, pred_ids):
        self.pred_ids = pred_ids 

    def set_train_ids(self, train_ids):
        self.train_ids = train_ids 

    def set_dev_ids(self, dev_ids):
        self.dev_ids = dev_ids 

    def set_cit_as_vocab(self, cit_as_vocab):
        self.cit_as_vocab = cit_as_vocab

    def set_model(self, model_name):
        self.model_name = model_name

    def set_features(self, features):
        self.features = features

    def set_partition(self, K, i):
        self.partition = f"{i+1}-{K}"

    ################################################################
    # CITATION CONVERTERS
    ################################################################

    # Load a dictionary to perform conversion of cit_idx to
    # cit_class
    def init_citation_converter(self):
        cit_list = list(self.tokenizer.cit_vocabulary\
                             .citation_counts)
        self.cit_to_class = {}
        for cit in range(self.num_cits):
            cit_str = cit_list[cit]
            if cit_str.startswith(normal_code_prefix):
                self.cit_to_class[cit] = 3 # code
            elif cit_str.startswith(normal_reg_prefix):
                self.cit_to_class[cit] = 4 # reg
            elif cit_str.startswith(normal_case_prefix):
                self.cit_to_class[cit] = 2 # case
            else:
                self.cit_to_class[cit] = 1 # none


    # Convert the cit_idx into cit_class if necessary
    def convert_citation_by_target(self, cit):
        if self.target == "cit_idx":
            return cit
        elif self.target == "cit_class":
            return self.cit_to_class[cit]

    ################################################################
    # STAGE 1
    ################################################################

    # Compute an inverted index of the form 
    # { cit: 
    #    { docid : [locations]}
    # }
    # so that we can look up each citation and find the context
    # in which it occurred during the second stage.
    # Note:
    #   If self.target="cit_class", then cit = {1,2,3,4}
    #   If self.target="cit_idx", then cit = {1,...,20607}
    def compute_cit_locations(self):
        print("Starting to compute citation locations...")

        # Function to add new entry to cit_dict
        def _add_to_cit_dict(self, cit, docid, locations):
            if cit in self.cit_dict:
                self.cit_dict[cit][docid] = locations
            else:
                self.cit_dict[cit] = {docid : locations}

        # Function to flush data, by ascending order of cit
        def _flush_data(self):
            citloc_fpath = os.path.join(self.text_sim_dir, 
                                f"citloc_{self.target}.json")
            with open(citloc_fpath, "w") as f:
                for cit in sorted(self.cit_dict.keys()):
                    f.write(json.dumps({cit : self.cit_dict[cit]}) + "\n")
            self.cit_dict = {} # clear cache

        timer = Timer()
        self.cit_dict = {} # Citation dict with document: locations
        self.open_encoded_texts()

        case_ids = list(self.all_data.keys())
        for case_id in tqdm(case_ids):

            # Load One Case
            # Identify Citation Locations
            encoded_txt = self.all_data[case_id][:]
            cit_indices = np.where((encoded_txt < self.num_cits) &
                                   (encoded_txt > 0))[0]
            cit_list = np.unique(encoded_txt[cit_indices]).tolist()
            for cit in cit_list:
                locations = np.where(encoded_txt == cit)[0].tolist()
                cit = self.convert_citation_by_target(cit)
                _add_to_cit_dict(self, cit, case_id, locations)

        # Flush data 
        _flush_data(self)
        timer.end()
        self.close_encoded_texts()


    ################################################################
    # STAGE 2
    ################################################################
    # Second stage computes and stores tfidf contexts for each 
    # citation. This is used for the predict_cases step later.
    def compute_tfidf_stage2(self):

        # Given a list of sparse arrays, each of shape (n_i, m), 
        # where n_i can differ (hence called "ragged").
        # Stack them on top of each other to get stacked array 
        # of shape (sum n_i, m). Also keep track of the start index
        # of each array and the length of each array.
        # Adapted from: https://tonysyu.github.io/ragged-arrays.html
        def _stack_ragged(array_list, axis=0):
            lengths = [np.shape(a)[axis] for a in array_list]
            idx = np.cumsum(lengths[:-1])
            idx = np.insert(idx, 0, 0)
            stacked = sparse.vstack(array_list)
            lengths = np.expand_dims(np.array(lengths), 1)
            return stacked, idx, lengths


        # Load the tfidf_arrays into a stacked array for use during
        # prediction.
        # self.available_cits is an array storing the available citations
        # in the train_data, which means that we can only predict these
        # citations (and not non-occurring ones).
        def _load_tfidf_array(self):
            larray = []
            cits = []
            while len(self.tfidf_arrays) > 0:
                cit, data = self.tfidf_arrays.popitem()
                larray.append(data)
                cits.append(cit)
            self.train_data, self.train_idx, self.train_lengths = \
                _stack_ragged(larray, axis=0)
            larray = None
            self.available_cits = np.array(cits, dtype=np.int32)
            print(f"Loaded {len(cits)} tfidf arrays with total" +
                  f" of {self.train_data.shape[0]} rows...")


        # When loading is done for each citation, push the contexts
        # for this citation into tfidf_arrays
        def _push_to_array(self, cit):
            tfidf_list = self.tfidfs.pop(cit)
            self.tfidf_arrays[cit] = sparse.vstack(tfidf_list)
            self.cit_count.pop(cit)


        # Store each tfidf context in the tfidfs dict
        def _add_to_tfidf(self, cit, tfidf): 
            if cit in self.tfidfs:
                self.tfidfs[cit].append(tfidf)
            else:
                self.tfidfs[cit] = [tfidf]


        # Given a citation, encoded_txt, [locations]
        # Retrieve the corresponding context and compute the tfidf
        # and store it.
        def _add_cits_by_locations(self, cit, encoded_txt, locations):
            for ci in locations:
                text_span_start = max(ci-self.text_span, 0)
                context = encoded_txt[text_span_start:ci]
                tfidf = self.compute_tfidf_from_context(context)
                if not tfidf is None:
                    _add_to_tfidf(self, cit, tfidf)
                    self.cit_count[cit] = self.cit_count.get(cit)-1


        # Given a citation and the {document: [locations]} 
        # dictionary where it can be found, look up a random doc
        # and load the contexts for all citation occurrences in doc.
        def _load_cit_line(self, cit, docs):
            while len(docs) > 0:
                docid = random.sample(docs.keys(), 1)[0]
                locations = docs.pop(docid)

                if not docid in self.train_ids_set:
                    continue

                encoded_txt = self.all_data[docid][:]
                _add_cits_by_locations(self, cit, encoded_txt, locations)

                # End when loaded max_cit_count number of contexts
                if self.cit_count[cit] <= 0:
                    _push_to_array(self, cit)
                    return

            # End if all documents exhausted
            if cit in self.tfidfs:
                _push_to_array(self, cit)


        # Initialize
        print("Starting Stage 2...")
        timer = Timer()
        self.tfidfs = {}
        self.tfidf_arrays = {}
        self.cit_count = {}
        self.train_ids_set = set(self.train_ids)
        for cit in range(self.num_cits):
            self.cit_count[cit] = self.max_cit_count

        # For full runs, load up to max_cit_count per citation.
        if not self.debug:
            with open(self.citloc_fpath, "r") as f:
                lines = f.read().splitlines()
            for line in tqdm(lines):
                cit, docs = json.loads(line).popitem()
                cit = int(cit)
                if self.target == "cit_class" and cit == 1:
                    continue
                _load_cit_line(self, cit, docs)

        # For debug mode:
        # Load all citation occurrences in each document 
        # (since the number of documents is small).
        if self.debug:
            train_ids = random.sample(self.train_ids, self.max_docs)
            for train_id in tqdm(train_ids):
                encoded_txt = self.all_data[train_id][:]
                locs = np.where((encoded_txt < self.num_cits) &
                                (encoded_txt > 0))[0]
                for ci in locs:
                    cit = self.convert_citation_by_target(encoded_txt[ci])
                    if self.target == "cit_class" and cit == 1:
                        continue
                    _add_cits_by_locations(self, cit, encoded_txt, [ci])

            # Push lists of tfidfs into tfidf_arrays
            while len(self.tfidfs) > 0:
                cit, tfidf_list = self.tfidfs.popitem()
                self.tfidf_arrays[cit] = sparse.vstack(tfidf_list)

        # Load train data
        _load_tfidf_array(self)

        print(f"Finished stage 2. | {timer.end()}")


    ################################################################
    # MODEL TRAINING
    ################################################################
    # Train the svmRank model if model_name == "letor"
    def train_model(self, n_svm=100):

        if self.model_name == "simple":
            self.model = PredictionModelSimple(self)
        elif self.model_name == "letor":
            self.model = PredictionModelLetor(self, n_svm)
        print("Start model training...")
        self.model.train()


    ################################################################
    # PREDICT CASES
    ################################################################
    # Use the stored train_data from stage2 to predict for new cases
    # Predict for all citation locations in self.pred_ids
    def predict_cases(self, n_cases=-1):

        # Function to convert an array of citation positions
        # into the citation target it represents
        def _cit_idx_to_cit(self, cit_idx):
            return self.available_cits[cit_idx]

        # Predictions for a single test document
        def _predict_for_doc(self, encoded_txt, case_id):
            cit_indices = np.where((encoded_txt < self.num_cits) &
                                   (encoded_txt > 0))[0]
            if isinstance(self.model, PredictionModelLetor):
                list_features = self.model.extract_features(case_id)
            else:
                list_features = None

            # Gather data for each citation location
            l = []
            true_cits = []
            for ci in cit_indices:
                cit = encoded_txt[ci]
                cit = self.convert_citation_by_target(cit)
                if self.target == "cit_class" and cit == 1:
                    continue

                text_span_start = max(ci-self.text_span, 0)
                context = encoded_txt[text_span_start : ci]
                test_tfidf = self.compute_tfidf_from_context(context)
                if not test_tfidf is None:
                    l.append(test_tfidf)
                    true_cits.append(cit)

            if len(true_cits) == 0:
                return
            
            # model.forward computes a dot product
            #   train_array: n_contexts x n_vocab
            #   test_array: n_vocab x n_pos
            #   x : n_contexts x n_pos
            #   y : n_cits x n_pos
            test_array = sparse.vstack(l)
            y = self.model.forward(test_array, list_features)

            # Sort based on score
            # sorted_ind: Indices of citations sorted by score
            sorted_ind = np.argsort(-y, axis=0)
            _add_to_metrics(self, true_cits, sorted_ind)


        def _init_metrics(self):
            self.metrics = {"n": 0}

            if self.target == "cit_idx":
                for k in self.klist:
                    self.metrics[f"recall_{k}"] = 0

            elif self.target == "cit_class":
                for target in self.cit_targets:
                    self.metrics[f"cit_{target}"] = \
                        {"predict": 0,
                         "actual": 0, 
                         "correct": 0}


        def _add_to_metrics(self, true_cits, sorted_ind):
            self.metrics["n"] += len(true_cits)

            if self.target == "cit_idx":
                for k in self.klist:
                    ind = sorted_ind[0:k, :]
                    for position, true_cit in enumerate(true_cits):
                        predicted_cits = _cit_idx_to_cit(self, ind[:, position])
                        if true_cit in predicted_cits:
                            self.metrics[f"recall_{k}"] += 1

            elif self.target == "cit_class":
                for position, true_cit in enumerate(true_cits):
                    top_predict = _cit_idx_to_cit(self, sorted_ind[0, position])
                    top_predict = int(top_predict)
                    self.metrics[f"cit_{top_predict}"]["predict"] += 1
                    self.metrics[f"cit_{true_cit}"]["actual"] += 1
                    if true_cit == top_predict:
                        self.metrics[f"cit_{true_cit}"]["correct"] += 1


        def _combine_metrics(self, d):
            self.metrics["n"] += d["n"]
            if self.target == "cit_idx":
                for k in self.klist:
                    self.metrics[f"recall_{k}"] += d[f"recall_{k}"]

            elif self.target == "cit_class":
                for target in self.cit_targets:
                    temp = d[f"cit_{target}"]
                    for k in temp.keys():
                        self.metrics[f"cit_{target}"][k] += temp[k]


        def _worker_process(self, queue, return_queue):
            doc_count = 0
            for serialized_txt in iter(queue.get, None):
                encoded_txt, case_id = pickle.loads(serialized_txt)
                _predict_for_doc(self, encoded_txt, case_id)
                doc_count += 1
            return_queue.put(self.metrics)


        print("Start predicting...")
        timer = Timer()

        # Initialize
        self.metrics = None
        _init_metrics(self)

        list_workers, queue, return_queue =\
                self.init_queue_and_workers(_worker_process)

        # 1. Put case_ids onto queue
        # 2. End Workers
        # 3. Retrieve Values
        if n_cases > 0:
            pred_ids = random.sample(self.pred_ids, n_cases)
        else:
            pred_ids = self.pred_ids
            
        for pred_id in tqdm(pred_ids):
            encoded_txt = self.all_data[pred_id][:]
            serialized_txt = pickle.dumps((encoded_txt, pred_id), 
                                          protocol=pickle.HIGHEST_PROTOCOL)
            queue.put(serialized_txt)
        for _ in list_workers:
            queue.put(None)
        dicts = []
        for w in list_workers:
            dicts.append(return_queue.get())
        for w in list_workers:
            w.join()

        # Combine metrics
        while len(dicts) > 0:
            d = dicts.pop()
            _combine_metrics(self, d)

        # Close up
        print(f"Finished Predicting {len(self.pred_ids)} documents. | {timer.end()}")
        return self.metrics


    def log_metrics(self, metrics, h):

        def _log_cit_idx(self, metrics):
            n = metrics["n"]
            return [metrics[f"recall_{k}"]/n*100 for k in self.klist]

        def _log_cit_class(self, metrics):
            n = metrics.pop("n")
            k = len(metrics)
            h_cols = []; h_vals = []
            macro_f1 = 0
            for cit_name, d in metrics.items():
                p = calc_precision(d)
                r = calc_recall(d)
                f1 = calc_f1(p, r)
                macro_f1 += f1
                h_cols.extend([f"{cit_name}-{k}" for k in ["F1", "P", "R"]])
                h_vals.extend([f1, p, r])
            h_cols.append("Macro-F1"); h_vals.append(macro_f1 / k)
            return h_cols, h_vals

        # Write hyperparameter headers
        h_cols = []; h_vals = []
        h_cols.append("vocab_size"); h_vals.append(self.vocab_size)
        h_cols.append("cit_as_vocab"); h_vals.append(self.cit_as_vocab)
        h_cols.append("target"); h_vals.append(self.target)
        h_cols.append("text_span"); h_vals.append(self.text_span)
        h_cols.append("max_docs"); h_vals.append(self.max_docs)
        h_cols.append("max_cit_count"); h_vals.append(self.max_cit_count)
        h_cols.append("klist"); h_vals.append(print_list(self.klist))
        h_cols.append("partition"); h_vals.append(self.partition)
        h_cols.append("n"); h_vals.append(metrics["n"])
        h_cols.extend([k for k in h.keys()])
        h_vals.extend([print_list(v) if isinstance(v,list) 
                        else v for v in h.values()])

        if self.target == "cit_idx":
            h_cols.extend([f"recall_{k}" for k in self.klist])
            h_vals.extend(_log_cit_idx(self, metrics))
        elif self.target == "cit_class":
            cols, vals = _log_cit_class(self, metrics)
            h_cols.extend(cols)
            h_vals.extend(vals)

        # Print Metrics
        l = []
        for k, v in zip(h_cols, h_vals):
            if isinstance(v, float):
                v = f"{v:.2f}%"
            l.append(f"{k}: {v}")
        message = " | ".join(l)
        print(message)

        if not self.debug:
            if not os.path.isfile(self.log_fpath):
                with open(self.log_fpath, "w") as f:
                    f.write(",".join(h_cols) + "\n")

            with open(self.log_fpath, "a") as f:
                h_vals = [str(v) for v in h_vals]
                f.write(",".join(h_vals) + "\n")


class PredictionModel:

    def __init__(self, tfidfmodel):
        self.tfidfmodel = tfidfmodel
        self.available_cits = tfidfmodel.available_cits.tolist()
        self.metadata_dict = tfidfmodel.metadata_dict
        self.encoded_texts = tfidfmodel.encoded_texts
        self.text_sim_dir = tfidfmodel.text_sim_dir
        self.all_data = tfidfmodel.all_data
        self.train_ids = tfidfmodel.train_ids
        self.dev_ids = tfidfmodel.dev_ids
        self.train_data = tfidfmodel.train_data
        self.train_idx = tfidfmodel.train_idx
        self.train_lengths = tfidfmodel.train_lengths
        self.compute_tfidf_from_context = tfidfmodel.compute_tfidf_from_context
        self.convert_citation_by_target = tfidfmodel.convert_citation_by_target
        self.num_cits = tfidfmodel.num_cits
        self.text_span = tfidfmodel.text_span
        self.target = tfidfmodel.target
        self.debug = tfidfmodel.debug
        self.citloc_fpath = tfidfmodel.citloc_fpath

class PredictionModelSimple(PredictionModel):

    def __init__(self, tfidfmodel):
        super(PredictionModelSimple, self).__init__(tfidfmodel)
        
    # No training required for simple model
    def train(self):
        pass

    # Dot Product
    # train_data: n_contexts x n_vocab
    # test_array: n_pos x n_vocab
    # x : n_contexts x n_pos
    # y : n_cits x n_pos
    # where y is a vector of similarity scores across citations
    # for each position
    # Recall that we had a stacked ragged array storing all the contexts.
    # Now we need to aggregate them by citation using np.add.reduceat.
    def forward(self, test_array, isscode):
        A = self.train_data.dot(test_array.T).toarray()
        x = np.square(A)
        y = np.add.reduceat(x, self.train_idx, axis=0)/self.train_lengths
        return y


class PredictionModelLetor(PredictionModel):

    # Give the ModelLetor object access to data that it needs
    def __init__(self, tfidfmodel, n):
        super(PredictionModelLetor, self).__init__(tfidfmodel)
        self.features = tfidfmodel.features
        self.pred_ids = tfidfmodel.pred_ids
        self.cit_targets = tfidfmodel.cit_targets
        self.n = n
        if '' in self.features:
            self.features = None

        for feature in self.features:
            self.load_feature(feature)

    def train(self):
        if self.target == "cit_idx":
            self.make_pairwise_data()
        elif self.target == "cit_class":
            self.make_pairwise_data()

        # Hyperparameter search for C
        Clist = [1.0]
        #Clist = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
        coeflist = []
        scorelist = []
        for C in Clist:
            coeflist.append(self.train_svm(C))
            metrics = self.tfidfmodel.predict_cases(100)
            
            # Maximise recall_5 for cit_idx
            if self.target == "cit_idx":
                n = metrics["n"]
                scorelist.append(metrics[f"recall_5"]/n*100)

            # Maximise macro_f1 for cit_class
            elif self.target == "cit_class":
                n = metrics.pop("n")
                k = len(metrics)
                macro_f1 = 0
                for cit_name, d in metrics.items():
                    p = calc_precision(d)
                    r = calc_recall(d)
                    f1 = calc_f1(p, r)
                    macro_f1 += f1
                macro_f1 = macro_f1 / k
                scorelist.append(macro_f1)
                print(f"C: {C} | Macro-F1: {macro_f1:.3f}")

        # Retrain with best C
        idx = scorelist.index(max(scorelist))
        C = Clist[idx]
        print(f"Selected best C={C} with score of {max(scorelist)}.")
        self.coef = coeflist[idx]
        print("Finished model training.")


    def extract_features(self, case_id):
        list_features = []
        for feature in self.features:
            if feature == "base":
                v = None
            else:
                v = self.metadata_dict[case_id][feature]
            list_features.append(v)
        return list_features


    # Take a random sample of cases and convert them into pairwise data
    # for training the svm model.
    def make_pairwise_data(self):
        random_ids = random.sample(self.dev_ids, self.n)
        self.X = []
        self.Y = []

        for train_id in random_ids:
            encoded_txt = self.tfidfmodel.all_data[train_id][:]
            list_features = self.extract_features(train_id)

            # Find citations
            # Each citation occurrence is one query, we normalize the scores
            # on a per-query basis
            cit_indices = np.where((encoded_txt < self.num_cits) &
                                   (encoded_txt > 0))[0]
            for ci in cit_indices:
                true_cit = encoded_txt[ci]
                true_cit = self.convert_citation_by_target(true_cit)
                if not true_cit in self.available_cits:
                    continue
                text_span_start = max(ci-self.text_span, 0)
                context = encoded_txt[text_span_start : ci]
                test_tfidf = self.compute_tfidf_from_context(context)
                if test_tfidf is None:
                    continue 
                tfidf_scores = self.get_tfidf_scores(test_tfidf)
                list_scores = []
                for i, feature in enumerate(self.features):
                    score = self.get_feature_scores(feature, list_features[i])
                    list_scores.append(score)

                scores = np.hstack([tfidf_scores] + list_scores)
                scores = normalize(scores)
                self.add_row(true_cit, scores)

        self.X = np.vstack(self.X)
        self.Y = np.concatenate(self.Y)

    # Given a context, return vector of similarity scores
    # against each possible target
    # Refer to the forward method of ModelSimple for more details.
    def get_tfidf_scores(self, test_tfidf):
        A = self.train_data.dot(test_tfidf.T).toarray()
        x = np.square(A)
        y = np.add.reduceat(x, self.train_idx, axis=0)/self.train_lengths
        return y

    # scores: n_cits * n_features
    def add_row(self, true_cit, scores):
        true_idx = self.available_cits.index(true_cit)
        score_true = np.expand_dims(scores[true_idx, :], 0)
        scores = np.delete(scores, true_idx, axis=0)
        scores = score_true - scores
        switch = random.choice([0,1])
        targets = np.array([(-1)**(k+switch) for k in range(scores.shape[0])])
        scores *= np.expand_dims(targets, 1)

        self.X.append(scores)
        self.Y.append(targets)

    def load_feature(self, col_name):
        feature_name = f"feature_{col_name}_{self.target}.pkl"
        feature_fpath = os.path.join(self.text_sim_dir, feature_name)
        with open(feature_fpath, "rb") as f:
            setattr(self, f"f_{col_name}", pickle.load(f))


    def get_feature_scores(self, feature_name, feature_value=None):
        scores = []
        feature = getattr(self, f"f_{feature_name}")
        if not feature_value is None:
            feature = feature[feature_value]
        for cit in self.available_cits:
            scores.append(feature[cit])
        return np.expand_dims(np.array(scores), 1)


    # Set dual=False as number of features is low
    def train_svm(self, C=1.0):
        clf = svm.LinearSVC(C=C, dual=False, penalty='l2', max_iter=5000,
                            fit_intercept=False)
        clf.fit(self.X, self.Y)
        self.coef = clf.coef_.ravel() / np.linalg.norm(clf.coef_)

        coef_weights = [f"text_sim: {self.coef[0]:.3f}"]
        for i, _ in enumerate(self.features):
            coef_weights.append(f"{self.features[i]}: {self.coef[i+1]:.3f}")
        print("Feature Weights| " + " | ".join(coef_weights))
        return self.coef


    # tfidf_scores: n_cits x n_pos
    # every other feature score: n_cits x 1
    # Each feature is multiplied by its weight and added together
    # using numpy broadcasting.
    def forward(self, test_array, list_features):
        tfidf_scores = self.get_tfidf_scores(test_array)
        tfidf_scores = normalize(tfidf_scores)
        scores = tfidf_scores * self.coef[0]

        for i, feature in enumerate(self.features):
            score = self.get_feature_scores(feature, list_features[i])
            score = normalize(score)
            scores += score * self.coef[i+1]
        return scores

################################################################
# Generic helpers
################################################################

def normalize(X):
    X = X - X.min(axis=0)
    X = X / X.max(axis=0)
    return X

def calc_precision(d):
    if d["predict"] == 0:
        precision = 0
    else:
        precision = d["correct"] / (d["predict"]) *100
    return precision

def calc_recall(d):
    if d["actual"] == 0:
        recall = 0
    else:
        recall = d["correct"] / (d["actual"]) *100
    return recall

def calc_f1(p, r):
    if p + r == 0:
        f1 = 0
    else:
        f1 = 2 * (p*r)/(p+r)
    return f1

def print_list(l):
    return "|".join([str(x) for x in l])
