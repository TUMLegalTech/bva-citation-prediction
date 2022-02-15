import os
import h5py
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
from tqdm import tqdm
from pathlib import Path
import pickle
import numpy as np
import random
import re
import json
from multiprocessing import SimpleQueue, Manager, Process
from utils import Timer

class TfidfTokenizer:

    def __init__(self, 
                 postprocessed_dir, 
                 tokenizer_fpath,
                 encoded_txt_fpath,
                 cit_vocabulary,
                 train_ids,
                 dev_ids,
                 test_ids):
        self.train_ids = train_ids
        self.dev_ids = dev_ids
        self.test_ids = test_ids
        self.fnames = [os.path.join(postprocessed_dir, f"{fn}.json") for
                        fn in self.train_ids]

        self.postprocessed_dir = postprocessed_dir 
        self.tokenizer_fpath = tokenizer_fpath
        self.encoded_txt_fpath = encoded_txt_fpath
        self.cit_vocabulary = cit_vocabulary
        self.num_cits = len(cit_vocabulary)

        # Add stopwords
        with open("stopwords.txt", "r") as f:
            self.stopwords = f.read().splitlines()
        self.stopwords = set(self.stopwords)

        # Add cits
        self.cits = []
        for cit in range(self.num_cits):
            self.cits.append(f"cit{cit}")
        self.cits = set(self.cits)

    # Pre-encode by features
    # Then TfidfModel can cutoff features at K and use
    def save_tfidf(self, min_df=10):

        def _add_tf(self, w):
            if not w in self.tfidf_dict:
                self.tfidf_dict[w] = {}
            self.tfidf_dict[w]["tf"] = self.tfidf_dict[w]\
                                            .get("tf", 0) + 1

        def _add_idf(self, w):
            if not w in self.tfidf_dict:
                self.tfidf_dict[w] = {}
            self.tfidf_dict[w]["idf"] = self.tfidf_dict[w]\
                                            .get("idf", 0) + 1

        def _worker_process(self, queue, return_queue, worker_id):
            self.tfidf_dict = {}
            for fname in iter(queue.get, 'STOP'):
                _process_one_doc(self, fname)

            # Return to master
            return_queue.put(self.tfidf_dict)

            print("All done!")

        def _process_one_doc(self, fname):
            with open(fname, "r") as f:
                resolved_txt = f.read()
            
            token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
            word_list = re.findall(token_pattern, resolved_txt)
            word_set = set()
            for w in word_list:
                if w in self.stopwords:
                    continue
                else:
                    _add_tf(self, w)
                    word_set.add(w)

            # Find citations
            cit_list = re.findall(r"@(cit[\d]+)@", resolved_txt)
            cit_set = set(cit_list)

            word_set = list(word_set|cit_set)
            for w in word_set:
                _add_idf(self, w)

        # Start of learning
        timer = Timer()
        queue = SimpleQueue()
        return_queue = SimpleQueue()
        num_workers = os.cpu_count()
        list_workers = []
        for worker_id in range(num_workers):
            process = Process(target=_worker_process, 
                              args=(self, queue, return_queue, worker_id))
            list_workers.append(process)
            process.start()

        # Put files on queue
        for fname in self.fnames:
            queue.put(fname)

        # End workers
        for worker_id in range(num_workers): 
            queue.put('STOP')

        # Retrieve values
        dicts = []
        for _ in list_workers:
            dicts.append(return_queue.get())

        # Join Workers
        for worker_id in range(num_workers): 
            list_workers[worker_id].join()

        # Write to shared dictionary
        tfidf_dict = {}
        while len(dicts) > 0:
            d = dicts.pop()
            for k,v in d.items():
                if k in tfidf_dict:
                    tfidf_dict[k]["tf"] += v.get("tf", 0)
                    tfidf_dict[k]["idf"] += v["idf"]
                else:
                    tfidf_dict[k] = {}
                    tfidf_dict[k]["tf"] = v.get("tf", 0)
                    tfidf_dict[k]["idf"] = v["idf"]

        # Add idf for cits
        cit_idf_list = []
        for cit in range(1, self.num_cits):
            cit = f"cit{cit}"
            if cit in tfidf_dict:
                v = tfidf_dict.pop(cit)
                cit_idf_list.append(v["idf"])
            else:
                cit_idf_list.append(0)

        # Trim out those below min_df
        tf_list = []
        idf_list = []
        word_list = []
        for k, v in tfidf_dict.items():
            if v["idf"] >= min_df:
                tf_list.append(v["tf"])
                idf_list.append(v["idf"])
                word_list.append(k)
        tf_list = np.array(tf_list)
        sort_idx = np.argsort(-tf_list)
        self.words = [word_list[i] for i in sort_idx]
        idf_list = [idf_list[i] for i in sort_idx]
        
        # Combine the idf lists together
        self.idf = np.array(cit_idf_list + idf_list)
        self.idf = np.log(len(self.fnames) / (self.idf+1))
        print(timer.end())

    def save_vocab(self, vocab_size=-1):

        # 0 is <unk>
        self.vocab = {}
        self.vocab["<unk>"] = 0

        # Add cits
        i=1
        for idx in range(1, self.num_cits):
            self.vocab[f"@cit{idx}@"] = i
            i+=1

        # Add normal vocab words
        for word in self.words:
            self.vocab[word] = i
            i+=1

        self.idx_to_word = [k for k in self.vocab.keys()]
        print(f"Saved Vocab Size of {len(self.idx_to_word)}")

    def save(self):
        self.h5file = None
        with open(self.tokenizer_fpath, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def encode(self, txt):
        encoded = [self.vocab[w] if w in self.vocab else self.vocab["<unk>"] 
                    for w in txt.split()]
        return encoded

    def decode(self, encoded):
        decoded = [self.idx_to_word[idx] for idx in encoded]
        return decoded

    def open_h5file(self):
        self.h5file = h5py.File(self.encoded_txt_fpath, mode='w')

    def close_h5file(self):
        self.h5file.close()

    def encode_file_and_save(self, fname):
        with open(fname, "r") as f:
            encoded = np.array(self.encode(f.read()))
        dataset_name = Path(fname).stem
        self.h5file.create_dataset(dataset_name, data=encoded)

    def encode_files(self):
        case_ids = self.train_ids + self.dev_ids + self.test_ids
        fnames = [os.path.join(self.postprocessed_dir, f"{fn}.json") for
                  fn in case_ids]
        print("Encoding files...")
        for fname in tqdm(fnames):
            self.encode_file_and_save(fname)
