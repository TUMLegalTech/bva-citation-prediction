# BVA Citation Prediction

This repository holds code and data for the paper:

Zihan Huang, Charles Low, Mengqiu Teng, Hongyi Zhang, Daniel E. Ho, Mark Krass and Matthias Grabmair, _Context-Aware Legal Citation Recommendation using Deep Learning_, Proceedings ICAIL 2021 (in print)

The data is available via the [announcement blog](https://reglab.stanford.edu/data/bva-case-citation-dataset/) post at Stanford's RegLab.

Below you can find the instructions to recreate the preprocessing. If you want to recreate the models implemented in the paper, you can directly go to the `nlp` folder for more information.

Credit for valuable code scrubbing, testing, and documentation goes to TUM Research Assistant [Sebastian Moser](https://github.com/sebimo).

# Building Citation Vocabulary & Dataset
You can create a Pytorch [map-like dataset](https://pytorch.org/docs/stable/data.html#map-style-datasets) using the functions provided and illustrated in `dataset_build.py` from a notebook or console.
## 1. Setting Paths
For the training, evaluation and testing of the methods some configuration information needs to be given.
First, a number of paths should be set (fill in your paths) in the `config.yaml` files used or sometimes also within a analysis/processing script.
To recreate the training/testing results, you only need to change the paths in the configuration files (This assumes that you downloaded the data from the Google Drive folder).
```
# paths to caselaw access metadata. can be found in shared Google Drive
citation_dict_fpaths = ['../materials/cla_vet_app_metadata.jsonl']
# path to folder with full dataset as txt files
case_coll_dir = '.../data/bva/all-bva-decisions'
# folder to be used to store all preprocessed cases
preprocessed_dir = '.../data/bva/capstone_2020_preprocessed'
# text file with training set ids. one id per line
train_ids_fpath = '.../materials/train_data_ids.txt'
# like above, but smaller for debugging
train_ids_small_fpath = '.../materials/train_data_ids_small.txt'
```
In the following instructions, you can either proceed using a full `train_ids.txt` file or a smaller file (e.g. 50 decisions), which makes the computation and debugging must faster before moving on to the full training set. If you do, however, you will likely have to adjust the vocabulary size in the encoding training.

If you want to recreate the preprocessing, the following steps are necessary.

## 2. Preprocessing cases for citation analysis
Call `preprocess_cases(case_coll_dir, preprocessed_dir, train_ids_fpath)` using your paths and wait for preprocessing to complete. This may take a fair amount of time and will likely occupy multiple GB in disk space.

## 3. Citation Vocabulary Building
Call `cv = make_raw_citation_vocab(preprocessed_dir, train_ids_fpath)` using your paths. This produces a progress bar (~2 minutes on the training set) and returns a raw citation vocabulary object. You can get some basic stats by calling `cv.vocab_report()`, which may look like this:

```
vocabulary size: 118249
# total counts: 6280877
# code citation: 77539 ; counts: 3961311
# norm-code citation: 0 ; counts: 0
# case citation: 40710 ; counts: 2319566
# norm-case citation: 0 ; counts: 0
# norm citation: 0 ; counts: 0
```

## 4. Normalize the Vocabulary
Call `cvx = normalize_vocabulary(cv, citation_dict_fpaths)` using your paths. After the progress bar finishes `cvx` (~5-10 minutes on the training set) is now a reduced size citation vocabulary based on a deep copy of `cv`. You can again inspect its stats using `cvx.vocab_report()`. You will find that it has shrunk in size significantly:

```
vocabulary size: 20608
# total counts: 7515395
# code citation: 529 ; counts: 1508
# norm-code citation: 14387 ; counts: 5194357
# case citation: 20079 ; counts: 7513887
# norm-case citation: 2041 ; counts: 2239872
# norm citation: 16428 ; counts: 7434229
```

This vocabulary can be further reduced by using  `cvx.reduce_sparse_to_unknown(20)`.

I recommend pickling this object away once it has been computed to save you the time of recreating it.

## 5. Create a dataset object
You can now create a pytorch dataset object: `ds = dv.CitationPredictionDataset(preprocessed_dir, cvx, case_ids_fpath=train_ids_fpath)`  
You can look at the constructor parameters here:
```
__init__(self, case_dir, vocabulary, 
               case_ids=None, 
               case_ids_fpath=None, 
               tokenizer=None,
               target_mode='binary',
               ignore_unknown=True,
               add_case_meta=False,
               forecast_length=16,
               context_length=64,
               negative_sample_prob=.5,
               pre_padding=False)
```
As of this version, only `binary` mode is supported, where the dataset will randomly extract a context of length `context_length` from a decision and assign a binary target stating whether the `forecast_length` tokens succeeding the context contain at least one citation token. `pre_padding` controls whether context starts at least at position 0 in the text or whether it can start before the start of the opinion. You can provide the case ids for your dataset either as a list into `case_ids` or as the path to a file with one id per line via `case_ids_fpath`. Setting `add_case_meta` to `True` will append a dictionary of the elements `bva_id`, `judge`, and `year` from the case file from which the context has been extracted (None if they do not exist) as a third element to the tuple that is being loaded (Attention: This may break automatic batching unless you implement a custom `collate_fn` in Pytorch's dataloader).

`ignore_unknown` triggers whether general unknown tokens are considered positive citation tokens for purposes of data loading. This should be kept at True generally.

You now can try getting a data point with `next(iter(ds))`, but it will complain that the encoder is missing. You can now either load a tokenizer inheriting from [HuggingFace's PreTrainedTokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer) or train a novel encoding using SentencePiece.

### Negative Example Sampling
The parameter `negative_sample_prob` controls how negative examples are loaded (possible values: None or 0-1 float interval, default value `.5`). If set to `None`, every data instance is based on the context and forecasting window of a random offset in the decision, and the positive/negative label distribution will correspond to the actual data distribution. If set to `0`, only positive examples and no negative samples are loaded. If set to `1`, only negative examples are loaded. If set to a floating point probability value between `0` and `1`, the value determines the probability with which a loaded instance will be a negative example. For example, a value of `.2` will converge to a negative to positive ratio of about 2 to 8.

## 6a. Wrap a Huggingface Tokenizer
Create a pretrained Huggingface tokenizer, such as:  
```
from transformers import ReformerTokenizer
tok = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")
```
Check its vocabulary size: `len(tok)`
For the C&P version, this will be `321`  
Now load it into the dataset: `ds.wrap_huggingface_tokenizer(tok)`  
Once complete, `len(tok)` will return a vocabulary size incremented by one and the dataset object is ready to be used for loading. For performance reasons, the citation resolution happens outside the main tokenizer object, so the total vocabulary size (i.e. tokenizer vocabulary + citation vocabulary) can be obtained from the dataset object via `ds.total_vocab_size()`. This method should be used for fine tuning and embedding layer dimensions.

After loading, `ds.tokenizer` is now an object of `WrappedCitationTokenizer`, which provides `encode` and `decode` functions analogue to common tokanizers. It wraps a [HuggingFace's PreTrainedTokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer) object, which is available at `df.tokenizer.wrapped_tokenizer` and can be used accordingly. Importantly, it can be passed to the constructor of validation and testing datasets. There is no automatic loading after training to prevent any unnecessary memory-related slowdown. 

## 6b. Training the SentencePiece Model
Now that we have a training dataset and associated vocabulary, we can train an encoder (including the default parameters for intuitiveness):  
`ds.train_encoding_model(vocab_size=24000, model_prefix='bva_enc')`
Pay attention to the vocabulary size here. The `vocab_size` parameter includes the citation vocabulary as special symbols. For example, the above reduced `cvx` vocabulary has 20608 entries. If we set it the size to 24k, only 3392 slots will be available for actual language. On the other hand, if you are debugging on a small set of decisions, there only may be a couple of hundred words in the total vocabulary and SentencePiece will ask you to set the vocabulary size to at least that number. The error messages are intuitive here if you understand the problem underneath and things are easy to fix by setting the `vocab_size` accordingly. 

The encoding will take some time and produce the files `bva_enc.model` and `bva_enc.vocab` in the current working directory. You can load them into the model (again with default parameter for clarity):  
`ds.load_encoding_model(self, fpath='bva_enc.model')`
After loading, `ds.tokenizer` is now a [SentencePieceProcessor](https://github.com/google/sentencepiece/tree/master/python) object and can be used accordingly. Importantly, it can be passed to the constructor of validation and testing datasets. There is no automatic loading after training to prevent any unnecessary memory-related slowdown.  

## 7. Load Data Instances
We can now get data via `next(iter(ds))`. For example, here is a positive datapoint with context and target:
```
(tensor([20905, 20646, 20652, 20924, 20683, 20618, 20621, 20919, 20977, 20682,
         20635, 20944, 20926, 20733, 20611, 20659, 20611, 20636, 20823, 20611,
         20626, 20787, 20776, 20669, 20611, 20624, 20994, 20763, 20618, 20621,
         20830, 20996, 20665, 20642, 20633, 20835, 20624, 20694, 20613, 20657,
         20619, 20635, 20706, 20686, 20624, 20919, 20977, 20682, 20635, 20944,
         20926, 20733, 20611, 20659, 20613, 20611, 20989, 20646, 20893, 20852,
         20854, 20625, 20611, 20615]),
 tensor([1]))
```
You are now ready to use the dataset object with standard Pytorch data loaders. One epoch will then be one context-target pair from each decision in the dataset with the location chosen at random. `dataset_vocab.py` sets the Pytorch random seed to 42.

## 8. Target Types
There are three different target types that can be specified in the dataloader constructor:  
`binary` : `[LEGACY]` `0` if not citation is present in the forecasting window, `1` if there is  
`cit_class` : `[LEGACY]` Predict class of next citation in forecasting window. `0` => no citation; `1` => citation of unknown/unnormalized type; `2` => case citation; `3` => code citation (USC/USCA); `4` => regulation citation (CFR)  
`cit_idx` : Predict precise index of citation in the vocabulary. `0` means no citation, `1` means general unknown citation, `2`, `3`, and `4` mean normalized but unknown code, regulatioin, and case citation, respectively. Any other number `i` stands for citation at the vocabulary index `i` which can be queried with `<vocab-object>.citation_str_by_index(<idx>)`.
`cit_idx_multi` : `[LEGACY]` Predict a vocabulary-size multi-hot vector of all citations in the forecasting window. This is targeted at situatons where multiple citations occur in sequence, which occurs frequently with the citation normalization for code and regulation citations. An all-zero vector means no citation. Any position where `y[i] == 1.0` stands for citation at the vocabulary index `i` which can be queried with `<vocab-object>.citation_str_by_index(<idx>)`. Note that this multi-hot should not be used with a final softmax activation, but rather a sigmoid.

## Disclaimer
The code is still in alpha stage, largely undocumented and you will have to read source code and error messages, specifically for the vocabulary size parameter in the encoding training. Also notice that, with the default context and forecasting lengths, the data is considerably imbalanced towards negative examples. Increasing the forecasting size should remedy this. 

There is also some legacy code in this repository (e.g. different task definitions in the datasets). Especially, if you want to test the different `[LEGACY]` target types above, you need to make slight changes to the processing pipeline and evaluation code accordingly. For some more information about those tasks search for `LEGACY` in the code.
