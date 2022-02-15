# Tfidf Model

These scripts train a tfidf model for citation prediction. Optimised for running on a GCP e2-standard-8 virtual machine.

## Launch GCP VM (Optional)

If you do not wish to launch a GCP VM, you can also run this script on your local machine, but I would recommend 16GB of RAM if possible.

`install.sh` installs the necessary packages on the virtual machine on startup.

## Big Picture

Here is a high level overview of the pipeline (run these scripts in order):  
1. `data_preprocessing.py`: Preprocess raw text by extracting citations and tokenizing them. Also processes and stores metadata.
2. `tfidf_preprocessing.py`: Create the tfidf vocab and store the idf vector in a tokenizer object, using code from `tokenizer.py`. Also compute metadata features for use in svmRank later.
3. `run.py`: Here we run the experiments. Experiment parameters are stored in `experiment.params`. `run.py` loads these parameters, trains the model accordingly, predicts on the test set, and logs the results.

The rest of the README explains each step in greater detail.

### Setup Folders

Before doing anything, we need to set up the folders and data on the GCP VM or your local machine. This is done by setting the directories in `filepaths.py` and ensuring that the necessary data, namely the raw bva decisions, cla/f3d reporter metadata and the train/dev/test id files are present in the respective folders.

### Data Preprocessing

`data_preprocessing.py` preprocesses the citations within the raw bva decisions. This directly makes use of code from the nlp folder, copied here in `preprocessing.py` and `dataset_vocab.py` with very minor changes.

1. `preprocess_cases` takes raw bva decisions and preprocesses them into a json object, stored in `preprocessed_dir`.
2. `make_raw_citation_vocab` builds the raw citation vocabulary object, which is then pickled.
3. `normalize_vocabulary` normalizes the vocabulary object, which is then pickled.
4. `reduce_sparse_to_unknown` thresholds and discards rarely occurring citations, and the thresholded vocab object is pickled.
5. `CitationReplacer.write_processed_case_text` uses the thresholded vocabulary object to replace all citation occurrences in the raw bva decisions with cit tokens, e.g. @cit1@. This is essentially `get_processed_case_text` but we write the output out for later use. The output is stored in `postprocessed_dir`.
6. `connect_db` creates a connection to the luima MongoDB. To use this function, you will need to create a file `db_config.py` and input the luima DB details.

```
db_details = {
    "DB_HOST": XXXX,
    "DB_PORT": XXXX,
    "DB_DATABASE": XXXX,
    "DB_USERS": XXXX,
    "DB_PASSWORD": XXXX
}
```

7. `MetadataBuilder` connects to the database and processes the metadata features for use later. For e.g. it creates the `issarea` field from the various program codes, and it also creates the `vlj` feature. The processes metadata is stored in the location defined back in `filepaths.py`

### Tfidf Processing
`tfidf_preprocessing.py` trains the tfidf tokenizer that will tokenize the normal (non-citation) parts of each text. The main workhorse is the `TfidfTokenizer` class contained in the `tokenizer.py` file. I initially used sklearn's tfidf tokenizer but found it to be too slow, hence I coded this using multiprocessing Queue to speed things up.

1. `save_tfidf` goes through all decisions found in `train_ids`. For each document, it pulls out all tokens that are i) not stop words ii) do not contain digits. It counts the term frequency and document frequency of each word. Note that both citation tokens and non-citation tokens are counted. Care is taken to ensure that the `self.idf` attribute stores the *citation* tokens first, followed by *non-citation* tokens in descending term frequency order. This allows us to set vocab limits later if we choose to do so, by chopping off the list at a desired size.

2. `save_vocab` saves the `self.vocab` attribute, which is a dictionary from each raw token to its token id. The order of tokens is the same as that of `self.idf` described above.

3. `encode_files` uses the saved `self.vocab` dictionary to encode each text file in the `postprocessed` directory. If a word cannot be found in the dictionary, it is encoded as `<unk>` token. The encoded files are stored as numpy arrays in a `h5py` file, where each dataset has a key corresponding to the case id.

The second part of `tfidf_preprocessing.py` is actually not related to tfidf, but on preprocessing data to be used in the model training later. It uses the `TfidfModel` class from `model.py`. 

1. `compute_cit_locations` creates an *inverted* file dictionary of the following format. This allows us to later extract citation contexts for a specific citation when desired (as opposed to searching through files for a particular citation).

```
{ cit (e.g. 1):
    { docid : [locations of citation occurrence] }
}
```

Note that `cit=1` can mean different things depending on whether `self.target="cit_idx" or "cit_class"`. For this reason, we create a separate citation location dictionary for each target.

2. `make_feature` makes features for the svm model later. Basically, it calculates the quantity P(citation|feature=i), e.g. P(cit=1|year=2009). It does so by going through all documents in `train_ids` and counting this quantity.

### Model Training & Testing

The experiment workflow is articulated in `run.py`, which can be run with the `-d` flag for debug mode. It also requires a `-params` argument with the location of the experiment parameters file. Most of the code is found in the `model.py` file.

1. `run.py` starts by loading data, tokenizer and metadata. Then it sets a bunch of parameters based on `experiment.params` file. `vocab_size` refers to how many non-citation tokens we want to use. `cit_as_vocab` refers to whether we allow citations to be used as part of the vocabulary (or whether to treat them as unk). `target` is cit_idx or cit_class. `text_span` is how many tokens to include as context for each citation. `forecast_window` is an unimplemented feature, ignore it. `klist` specifies which recall@k metrics to log at test time. `max_cit_count` is the maximum number of contexts to store per citation. `max_docs` only affects when running in debug mode - it is the number of documents used for training - all citation occurrences are loaded. 

```
# First line denotes general settings
# 0: vocab_size
# 1: cit_as_vocab
# 2: target
# 3: text_span
# 4: forecast_window
# 5: klist (| separated)
# 6: max_cit_count
# 7: max_docs
25000,1,cit_idx,50,1,1|5|20,100,100

# Each line in the following is one run
# 0: model
# 1: features (| separated)
# 2: n_svm
simple,,100
letor,year,100
letor,year|issarea,100
letor,year|issarea|vlj,100
```

2. `compute_tfidf_stage2` is the training step for the text similarity model. For each citation, it looks through the training data and computes and stores tfidf vectors of each context of citation occurrence, up to `max_cit_count` number. At the end, it stacks all these contexts into a scipy sparse array called `self.train_data`. Given that the number of contexts per citation can vary (some citations only occur a few times), we need to keep track of the index where one citation ends and another citation begins. This is implemented in the `_stack_ragged` function where more details are in code comments.

3. `train_model` is the training step for the svm model, if `model=letor`, implemented in `PredictionModelLetor` class. The number of training documents used for this training is set by the `n_svm` parameter above. For each citation occurrence, the tfidf score calculated using `get_tfidf_scores`, and the feature scores in the `get_feature_scores` function. The data is normalized and then transformed into pairwise data. The svm model is trained and the coefficients stored. 

4. `predict_cases` goes through all the cases set in `run.py` using `set_pred_ids`, and makes a prediction for each citation occurrence in these documents. It does so by calling the `forward` method in the `PredictionModel`, which returns a matrix of predicted scores. These scores are sorted, giving us the top citations predicted for each citation occurrence. `_add_to_metrics` then compares these predictions against the actual true citation to return the metrics.

5. `log_metrics` logs these metrics in log files in the logs directory.
