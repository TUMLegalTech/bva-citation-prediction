import os
import pickle
import json
import csv
import random
import math
from pandas._config import config
import yaml
import argparse
from typing import Optional, Dict, Tuple, Union, Any

import numpy as np
from pandas import DataFrame
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset_build import make_raw_citation_vocab, normalize_vocabulary
from dataset_vocab import CitationPredictionDataset, WrappedCitationTokenizer, CitationVocabulary
from transformers import RobertaConfig, AutoConfig, TrainingArguments, Trainer, RobertaTokenizerFast
from bilstm import BiLSTM, CheckpointEveryNSteps, bilstm_eval_class, bilstm_eval_idx, bilstm_test_n_fold
from roberta import RobertaForSequenceClassification, roberta_evaluate_class, roberta_evaluate_idx, roberta_test_n_fold, roberta_model_analysis_val_data
from preprocessing import MetadataProcessor, DataPreprocessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FOLDS = 6

def generate_citation(
    input_text: str,
    trainer: Trainer,
    model: RobertaForSequenceClassification,
    wrappedTokenizer: RobertaTokenizerFast,
    cvx: DataFrame,
    input_meta: str='-1,-1,-1', 
    n: int=1, 
    context_length: int=64,
) -> None:
    """
        generateCitation() takes the customize input texts and metadata, extracts citations from texts, clean texts
        and padding(or truncation depends on input length) before forwarding input to model, 
        top n generated citations will be printed, n is configurable with default=1

        Arguments:
            input_text: customize input texts (input in config file)
            input_meta:  customize input metadata (input in config file)
            n: the number of citations desire to generate
            context_length: used context length, customization via config file
    """
    pp = DataPreprocessor()
    processed_input = pp.process_input_texts(input_text)
    pad_token_id = wrappedTokenizer.wrapped_tokenizer.pad_token_id
    citation_indices = [cvx.citation_indices_from_raw(cit)\
                        for cit in processed_input['citation_texts']]
    encoded = wrappedTokenizer.encode(processed_input["txt"], citation_indices)
    attention = torch.tensor([1] * len(encoded))

    # Trim/pad text to context length
    if len(encoded) < context_length:
        pre_padding = torch.tensor([pad_token_id] * (context_length - len(encoded)))
        pre_attention = torch.tensor([0] * (context_length - len(encoded)))
        encoded = torch.cat([torch.tensor(pre_padding, encoded)])
        attention = torch.cat([pre_attention, attention])
    else:
        encoded = torch.tensor(encoded[len(encoded) - context_length:])
        attention = attention[len(encoded) - context_length:]

    metadata = torch.tensor([int(meta) for meta in input_meta.split(",")])
    attention_padding = torch.tensor([0] * len(metadata))

    # Concatenate encoded text with meta data as input to the model
    encoded = torch.cat([encoded, metadata]).unsqueeze(0)
    attention = torch.cat([attention, attention_padding]).unsqueeze(0)

    logits = trainer.prediction_step(model, {"input_ids": encoded, "attention_mask": attention})[1].detach().tolist()[0]
    idxs = sorted(range(len(logits)), key=lambda sub: logits[sub], reverse=True)[:n]
    for idx in idxs:
        cit_str = list(cvx.citation_counts)[idx]
        print(f"Predicted citation string is {cit_str}, cit index is {idx}")


def load_cvx(
    cv_norm_path: str,
    cv_path: str,
    preprocessed_dir: str,
    train_ids_fpath: str,
    citation_dict_fpaths: str
) -> CitationVocabulary:
    """
        loads the reduced and thresholded citation vocab
        Arguments:
            cv_norm_path: (set in config file)
            cv_path: (set in config file)
            preprocessed_dir: (set in config file)
            train_ids_fpath: (set in config file)
            citation_dict_fpaths: (set in config file)
        Note: Only cv_norm_path is necessary here. The other
        arguments are given for LEGACY reasons as they are
        necessary to build the file at the cv_norm_path location. 
    """
    print('loading normalized vocabulary from '+cv_norm_path)
    with open(cv_norm_path, 'rb') as f:
        cvx = pickle.load(f)
        print('Reduced & Thresholded Citation Vocabulary:')
        cvx.vocab_report()

    return cvx

def load_metadata(
    meta_fpath: str,
    add_case_meta: bool
) -> Optional[Dict]:
    meta = None
    if add_case_meta:
        print("Loading metadata")
        meta = dict()
        with open(meta_fpath, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            rows = [row for row in reader]

        header = rows[0]
        rows = rows[1:]
        
        assert header == ["bva_id", "year", "class", "bfmemid"]
        
        # This is equivalent to the pd.DataFrame.groupby().ngroup() function used in the other function
        # In this case, we will have different indices though
        judge_set = set([x[-1] for x in rows])
        # Give each bfmemid a unique number starting at 0
        judge_idx = {judge: idx for (judge, idx) in zip(judge_set, range(len(judge_set)))}

        for row in rows:
            bva_id, year, cla, bfmemid = row
            assert bva_id not in meta

            meta[int(bva_id)] = {
                "year": int(year),
                "issarea": int(cla),
                "judge": judge_idx[bfmemid]
            }

    return meta

def build_metadata(
    meta_fpath: str,
    add_case_meta: bool
) -> Optional[DataFrame]:
    """
    You can use this method, if you have all the metadata available and want to preprocess it.
    For simply using the metadata.csv as given in the data download, use `load_metadata()`
    This is a drop-in replacement for the other function, given the type of metadata you have.
    For more information please look at the MetadataProcessor implementation.
    """
    meta_cache_fpath = meta_fpath.rsplit('.', 1)[0] + "_processed.pickle"
    meta = None
    if add_case_meta and not os.path.exists(meta_cache_fpath):
        print("Building metadata")
        mp = MetadataProcessor()
        meta = mp.build_metadata(metadata_fpath=meta_fpath)
        with open(meta_cache_fpath, 'wb') as f:
            pickle.dump(meta, f)
    elif add_case_meta:
        print("Loading metadata")
        with open(meta_cache_fpath, 'rb') as f:
            meta = pickle.load(f)

    return meta

def get_num_labels(
    task: str
) -> int:
    """
        Returns the corresponding number of labels for each task
    """
    if task.startswith("binary"):
        num_labels = 2
    elif task.startswith("cit_class"):
        num_labels = 5
    elif task.startswith("cit_idx"):
        num_labels = len(cvx)
    else:
        raise ValueError("Unknown task")

    return num_labels


def get_negative_sample_prob(
    task: str
) -> Optional[float]:
    """
        Returns the corresponding negative sampling probability for each task
        Attention: The "binary" task will not use negative sampling
    """
    if task.startswith("binary"):
        negative_sample_prob = None
    elif task.startswith("cit_class"):
        negative_sample_prob = 0.25
    elif task.startswith("cit_idx"):
        negative_sample_prob = 0
    else:
        raise ValueError("Unknown task")

    return negative_sample_prob


def setup_bilstm(
    wrappedTokenizer: RobertaTokenizerFast,
    batch_size: int,
    lr: float,
    enable_meta_year: bool,
    enable_meta_issarea: bool,
    enable_meta_judge: bool,
    num_labels: int,
    num_meta: int,
    task: str
) -> BiLSTM:
    """
    Whole bilstm execution part. 
    """
    model = BiLSTM(
        embedding_dim=768,
        hidden_dim=3072,
        vocab_size=len(wrappedTokenizer),
        batch_size=batch_size,
        label_size=num_labels,
        task=task,
        num_meta=num_meta,
        enable_meta_year=enable_meta_year,
        enable_meta_issarea=enable_meta_issarea,
        enable_meta_judge=enable_meta_judge,
        device=DEVICE,
        lr=lr,
    ).to(DEVICE)

    return model

def train_bilstm(
    model: BiLSTM,
    load_checkpoint_path: str,
    train_dataset: CitationPredictionDataset,
    dev_dataset: CitationPredictionDataset,
    batch_size: int,
    gradient_accumulation_steps: int,
    output_dir: str
):
    """
        Trains a initialized BiLSTM model.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )
    logger = TensorBoardLogger(
        output_dir,
        name="bilstm",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_loss',
        save_top_k=20,
    )
    trainer = pl.Trainer(
        weights_save_path=output_dir,
        checkpoint_callback=checkpoint_callback,
        gpus=int(DEVICE.type == "cuda"),
        logger=logger,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        val_check_interval=1.0,
        resume_from_checkpoint=load_checkpoint_path,
        accumulate_grad_batches=gradient_accumulation_steps,
    )
    trainer.fit(model, train_loader, dev_loader)

def test_bilstm(
    model: BiLSTM,
    wrappedTokenizer: RobertaTokenizerFast,
    cvx: DataFrame,
    load_checkpoint_path: str,
    negative_sample_prob: Optional[float],
    dataset_return_type: str,
    meta: Union[DataFrame, None],
    config_file: Dict
):
    """
        Tests a trained BiLSTM model on the 6 different test folds.
    """
    ckpt = torch.load(load_checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])
    model.freeze()
    del ckpt

    random.seed(42)
    bilstm_test_n_fold(NUM_FOLDS, model, config_file, cvx, wrappedTokenizer, negative_sample_prob, meta, dataset_return_type)

def setup_roberta(
    wrappedTokenizer: RobertaTokenizerFast,
    load_checkpoint_path: str,
    train_dataset: CitationPredictionDataset,
    dev_dataset: CitationPredictionDataset,
    pretrain_name: str,
    batch_size: int,
    lr: float,
    gradient_accumulation_steps: int,
    enable_meta_year: bool,
    enable_meta_issarea: bool,
    enable_meta_judge: bool,
    num_labels: int,
    num_meta: int,
    task: str,
    mode: str,
    judge_embedding_export_file: str,
    output_dir: str,
    config_file: Dict
) -> Tuple[RobertaForSequenceClassification, Trainer]:
    """
        Setup of the RoBERTa model. 
        Setup code for datasets, training schedule, trainer, model, optimizer.
    """
    if load_checkpoint_path is not None:
        print("Load checkpoint model: ", load_checkpoint_path.split('/')[-1])
        config_path = os.path.join(load_checkpoint_path, 'config.json')
        config = RobertaConfig.from_json_file(config_path)
        pretrained_model_name_or_path = load_checkpoint_path
        scheduler = torch.load(os.path.join(load_checkpoint_path, 'scheduler.pt'))
        # load last learning rate of the scheduler checkpoint
        lr = scheduler['_last_lr'][0]
    else:
        config = AutoConfig.from_pretrained(
            pretrain_name,
            is_decoder=False,  # disable causal encoding
        )
        pretrained_model_name_or_path = pretrain_name
    config.update(
        {
            'num_labels': num_labels,
            'num_meta': num_meta,
            'enable_meta_year': enable_meta_year,
            'enable_meta_issarea': enable_meta_issarea,
            'enable_meta_judge': enable_meta_judge,
            'task': task,
            'run_config': config_file # This is the the global config file
        }
    )
    model = RobertaForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path,
        from_tf=False,
        config=config
    )
    if judge_embedding_export_file:
        judge_embeddings = torch.clone(model.embedding.weight)
        je = judge_embeddings.detach().numpy()
        np.save(judge_embedding_export_file,
                je,
                allow_pickle=False)
        print('saved judge embedding model')

    # change embedding size for vocabulary expansion
    model.roberta.resize_token_embeddings(len(wrappedTokenizer))

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        do_train=True,
        do_eval=True,
        fp16=False,
        save_steps=200,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_first_step=False,
        logging_steps=9,
        learning_rate=lr,
        save_total_limit=5,
        dataloader_num_workers=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=8,
        load_best_model_at_end = True
    )

    if task.startswith("cit_class"):
        evaluate_metrics = roberta_evaluate_class
    elif task.startswith("cit_idx"):
        evaluate_metrics = roberta_evaluate_idx
    else:
        evaluate_metrics = None
    no_decay = ["bias", "LayerNorm.weight"]
    # little hack to disable huggingface default scheduler, weight decay is off by defaul
    optimizer_grouped_parameters = [
            {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
            },
            {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
            },
            ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=evaluate_metrics,
        optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.))
    )

    return model, trainer


def train_roberta(
    trainer: Trainer
):
    """
        Trains the RoBERTa model. Trainer is completely initialized by the setup_roberta function.
    """
    trainer.train()
    trainer.save_model()


def eval_roberta(
    trainer: Trainer,
    dev_dataset: CitationPredictionDataset
):
    """
        Evaluates RoBERTa on the development set.
    """
    random.seed(42)
    predOutput = trainer.evaluate(dev_dataset)
    print("metrics: ", predOutput)


def test_roberta(
    wrappedTokenizer: RobertaTokenizerFast,
    trainer: Any,
    cvx: DataFrame,
    negative_sample_prob: Optional[float],
    task: str,
    mode: str,
    dataset_return_type: str,
    predictions_analysis_file: str,
    meta: Union[DataFrame, None],
    config_file: Dict
):
    """
        Will either do the 6-fold testset performance evaluation or further model analysis dependening on `mode`.
    """
    assert mode in ['test', 'analysis']
    
    random.seed(42)
    # export prediction analysis data -> this file is used to save the results
    if ((predictions_analysis_file is not None)
        and task == 'cit_idx_predictions'):
        with open(predictions_analysis_file, 'w') as f:
            f.write('year;issarea;judge;label;pred;position;context;forecast;preds\n')
    
    if mode == 'test':
        roberta_test_n_fold(NUM_FOLDS, trainer, config_file, cvx, wrappedTokenizer, negative_sample_prob, meta, dataset_return_type)
    elif mode == 'analysis':
        roberta_model_analysis_val_data(trainer, config_file, cvx, wrappedTokenizer, negative_sample_prob, meta, dataset_return_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file used for training', required=True)
    args = parser.parse_args()

    with open(args.config) as file:
        config_file = yaml.safe_load(file)

    # Checking metadata settings:
    #  -> append metadata before classification or
    #  -> year, area and judge information are not used in the model
    assert config_file['add_case_meta'] or (not config_file['enable_meta_year'] and not config_file['enable_meta_issarea'] and not config_file['enable_meta_judge'])
    # Check that one of the supported model types is used
    assert config_file['model_type'].startswith('bilstm') or config_file['model_type'].startswith('roberta')
    if config_file['task'] != "cit_idx_predictions":
        # As stated in the config file cit_class, binary, etc. are LEGACY tasks. Especially for cit_class and binary, the whole processing 
        # pipeline is working (just remove this ValueError), but you need to adapt the 'test_[...]' code accordingly to get correct results.
        raise ValueError("[LEGACY] task: You need to adapt the evaluation procedure before using any task except 'cit_idx_predictions'")

    # load cvx = reduced and thresholded citation vocab
    cvx = load_cvx(
        cv_norm_path=config_file['cv_norm_path'],
        cv_path=config_file['cv_path'],
        preprocessed_dir=config_file['preprocessed_dir'],
        train_ids_fpath=config_file['train_ids_fpath'],
        citation_dict_fpaths=config_file['citation_dict_fpaths']
    )

    meta = load_metadata(
        meta_fpath=config_file["meta_fpath"],
        add_case_meta=config_file["add_case_meta"]
    )

    # set the number of metadata features for model
    num_meta = 3 if config_file['add_case_meta'] else 0

    # set num_labels based on task
    num_labels = get_num_labels(config_file["task"])
    negative_sample_prob = get_negative_sample_prob(config_file["task"])
    print(f"num labels {num_labels}")

    lr = float(config_file['learning_rate'])

    # For bilstm: 'lightning' for roberta: 'features' (there are currently only two different model types)
    dataset_return_type = 'lightning' if config_file['model_type'].startswith('bilstm') else 'features'

    tokenizer = RobertaTokenizerFast.from_pretrained(config_file['pretrain_name'])
    wrappedTokenizer = WrappedCitationTokenizer(tokenizer, cvx)

    train_dataset = CitationPredictionDataset(
        config_file['preprocessed_dir'],
        cvx,
        case_ids=None,
        case_ids_fpath=config_file['train_ids_fpath'],
        tokenizer=wrappedTokenizer,
        target_mode=config_file['task'],
        ignore_unknown=True,
        negative_sample_prob=negative_sample_prob,
        add_case_meta=config_file['add_case_meta'],
        meta=meta,
        forecast_length=config_file['forecast_length'],
        context_length=config_file['context_length'],
        pre_padding=False,
        return_type=dataset_return_type,
    )
    dev_dataset = CitationPredictionDataset(
        config_file['preprocessed_dir'],
        cvx,
        case_ids=None,
        case_ids_fpath=config_file['dev_ids_fpath'],
        tokenizer=wrappedTokenizer,
        target_mode=config_file['task'],
        ignore_unknown=True,
        negative_sample_prob=negative_sample_prob,
        add_case_meta=config_file['add_case_meta'],
        meta=meta,
        forecast_length=config_file['forecast_length'],
        context_length=config_file['context_length'],
        pre_padding=False,
        return_type=dataset_return_type,
    )

    if config_file['model_type'].startswith('bilstm'):
        assert config_file['mode'] in ["train", "test"]
        model = setup_bilstm(
            wrappedTokenizer=wrappedTokenizer,
            batch_size=config_file['batch_size'],
            lr=lr,
            enable_meta_year=config_file['enable_meta_year'],
            enable_meta_issarea=config_file['enable_meta_issarea'],
            enable_meta_judge=config_file['enable_meta_judge'],
            num_labels=num_labels,
            num_meta=num_meta,
            task=config_file['task']
        )
        if config_file['mode'] == 'train':
            train_bilstm(
                model=model,
                load_checkpoint_path=config_file['load_checkpoint_path'],
                train_dataset=train_dataset,
                dev_dataset=dev_dataset,
                batch_size=config_file['batch_size'],
                gradient_accumulation_steps=config_file['gradient_accumulation_steps'],
                output_dir=config_file['output_dir']
            )
        else:
            test_bilstm(
                model=model,
                wrappedTokenizer=wrappedTokenizer,
                cvx=cvx,
                load_checkpoint_path=config_file['load_checkpoint_path'],
                negative_sample_prob=negative_sample_prob,
                dataset_return_type=dataset_return_type,
                meta=meta,
                config_file=config_file
            )
    elif config_file['model_type'].startswith('roberta'):
        assert config_file['mode'] in ['train', 'test', 'analysis', 'eval']
        model, trainer = setup_roberta(
            wrappedTokenizer=wrappedTokenizer,
            load_checkpoint_path=config_file['load_checkpoint_path'],
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            pretrain_name=config_file['pretrain_name'],
            batch_size=config_file['batch_size'],
            lr=lr,
            gradient_accumulation_steps=config_file['gradient_accumulation_steps'],
            enable_meta_year=config_file['enable_meta_year'],
            enable_meta_issarea=config_file['enable_meta_issarea'],
            enable_meta_judge=config_file['enable_meta_judge'],
            num_labels=num_labels,
            num_meta=num_meta,
            task=config_file['task'],
            mode=config_file['mode'],
            judge_embedding_export_file=config_file['judge_embedding_export_file'],
            output_dir=config_file['output_dir'],
            config_file=config_file
        )

        if config_file['mode'] == "train":
            train_roberta(trainer)
        elif config_file['mode'] in ['test', 'analysis']:
            test_roberta(
                wrappedTokenizer=wrappedTokenizer,
                trainer=trainer,
                cvx=cvx,
                negative_sample_prob=negative_sample_prob,
                task=config_file['task'],
                mode=config_file['mode'],
                dataset_return_type=dataset_return_type,
                predictions_analysis_file=config_file['prediction_analysis_file'],
                meta=meta,
                config_file=config_file
            )
        elif config_file['mode'] == 'eval':
            eval_roberta(trainer=trainer, dev_dataset=dev_dataset)
        elif config_file['mode'] == 'predict':
            assert config_file['input_text'] is not None
            generate_citation(
                input_text=config_file['input_text'],
                trainer=trainer,
                model=model,
                wrappedTokenizer=wrappedTokenizer,
                cvx=cvx,
                input_meta=config_file['input_meta'], 
                context_length=config_file['context_length']
            )
        else:
            raise ValueError("Unknown 'mode' for RoBERTa model selected. Options: 'train', 'test', 'analysis', 'eval' or 'predict'.")
