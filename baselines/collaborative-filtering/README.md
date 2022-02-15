# Collaborative-Filtering Baseline

### What are these files in the folder?
- `setup.sh`: This is what I use to set up the environment on a brand new virtual machine. The Python version should be at least 3.7. Ignore this if your environment is already complete.
- `make_data_structures.py`: This creates the forward index and the inverted list based on the training set (233,513 documents). It takes up to 20 hours to run (if the *unreduced* new citation processor is used). These data structures will be used by the collaborative-filtering model.
- `make_metadata.py`: This creates a csv for document metadata by querying MongoDB. It requires a `db_config.py` file and takes about 30 seconds to run.
```python
# An example db_config.py
db_details = {
    "DB_HOST": "<this_is_db_host>",
    "DB_PORT": 12345,
    "DB_DATABASE": "<this_is_database_name>",
    "DB_USER": "<this_is_username>",
    "DB_PASSWORD": "<this_is_password>"
}
```
- `dataset_vocab.py`: Nothing interesting.
- `util.py`: This includes a set of utility functions to get objects (vocab, dataset, inverted list, etc.) and defines all the paths to data files.
- `model.py`: This defines the `CollabFilter` class as the collaborative filtering model. Two important methods are `train` and `predict`.
- `main.py`: This is the main entry point. It parses the command line arguments, creates a model object, samples pairwise training data from the documents, trains an SVM on the data, and performs evaluation on the test set.
- `summary.py`: When the evaluation is done, run this to produce LaTeX style table rows and perform significance testing.

### What needs to be done to run the whole pipeline?
1. Define all the data file paths in `util.py`, and add the `db_config.py` file.
2. Make data structures and the metadata file by running
```bash
python3 make_data_structures.py && python3 make_metadata.py
```
3. You can run `main.py` to do sampling, training and evaluation, either together or separately. The sampling process will write samples to a file from which the training and evaluation process can pick up. See `main.py` for a complete list of accepted command line arguments.
```bash
# example sampling command: sample training data with all metadata features from 1000 documents
python3 main.py --sample_only -d 1000 -m year issarea vlj
# example train and eval command: train the SVM and evaluate on fix folds on the full test set
python3 main.py --train_eval_only -n 6 -m year issarea vlj
# example complete pipeline:
# sample data from one training document with year features and evaluate on one test document
python3 main.py -d 1 -m year -t 1
```
After running the third command above, you'll expect to get shell outputs similar to the following:
```bash
Namespace(eval_result_folder='results', metadata=['year'], n_folds=1, n_sample_docs=1, n_test_docs=1, recommendation_limit=50, relevant_doc_limit=50, run_id=None, sample_only=False, scoring='binary', svm_c_param=1.0, svm_model_folder='results', train_eval_only=False, verbose='info')
Loading vocab and data structures...
Preparing to make pairwise data...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:19<00:00, 19.16s/it]
Number of training samples: 1444
Samples saved in results/svm_samples.csv
Start training SVM...
Feature Weights: collab_filter: 0.9480 | year: 0.3181
Coefficients saved in results/svm_coef_year.txt
Start predictions: assigning tasks to worker processes...
Collecting worker results...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:19<00:00, 19.52s/it]
Evaluation results saved in results/[small_class|full]_year.csv
```
The evaluation results will then be saved to `results/small_class_year.csv` and `results/full_year.csv`. These two csv files will look like (in pandas.DataFrame format):
```python
# small_class_year.csv
                        fold_1        mean  std
variable index                                 
case     f1-score     0.401894    0.401894  0.0
         precision    0.395473    0.395473  0.0
         recall       0.408528    0.408528  0.0
         support    727.000000  727.000000  0.0
code     f1-score     0.335375    0.335375  0.0
         precision    0.283312    0.283312  0.0
         recall       0.410882    0.410882  0.0
         support    533.000000  533.000000  0.0
reg      f1-score     0.361151    0.361151  0.0
         precision    0.445826    0.445826  0.0
         recall       0.303507    0.303507  0.0
         support    827.000000  827.000000  0.0
all      f1-macro     0.366140    0.366140  0.0
# full_year.csv
             fold_1      mean  std
recall@1   0.014705  0.014705  0.0
recall@5   0.044117  0.044117  0.0
recall@20  0.220588  0.220588  0.0
```
For efficiency, one could first sample training data for all metadata features, and then train the SVM based on different subsets of the metadata. For example:
```bash
python3 main.py --sample_only -d 1000 -m year issarea vlj
python3 main.py --train_eval_only -n 6 -m year
python3 main.py --train_eval_only -n 6 -m year issarea
python3 main.py --train_eval_only -n 6 -m year issarea vlj
```
4. When the evaluation is complete, run the following commands to summarize the tables into LaTeX table rows and get significance test results. See `summary.py` for a list of accepted command line arguments (most of which are similar to those in `main.py`).
```bash
python3 summary.py
python3 summary.py -p small_class
```

### What file paths are required to be defined in `util.py`?

- `ROOT_PATH`: the root path for all data and objects.
- `VOCAB_PATH`: path to the citation vocabulary.
- `METADATA_PATH`: path to the metadata csv, generated by `make_metadata.py`.
- `METADATA_FEATURE_PATH`: path to the metadata features.
- `INV_LIST_PATH`: path to the inverted list json file, generated by `make_data_structures.py`.
- `FWD_INDEX_PATH`: path to the forward index json file, generated by `make_data_structures.py`.
- `PROCESSED_CASES`: folder that contains all preprocessed BVA cases.
- `DATA_SPLIT`: a map from dataset name (one of `train`, `dev` and `test`) to the path that stores all the document IDs in that set. The dataset split is stored in the `./data_split` folder.

## Majority-Vote Baseline

This is implemented in `majority_vote.py`. It always predicts `reg` for small class prediction, and always predicts the 20 most popular citations for full citation prediction. It is evaluated on six folds of the whole test set. Running `python3 majority_vote.py` will begin the prediction and store the results.

