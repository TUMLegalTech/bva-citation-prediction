# Reference: https://github.com/clairett/pytorch-sentiment-classification

import os
import pickle
import torch
import random
import tqdm
import json
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from torch import nn
from sklearn.metrics import classification_report
from dataset_vocab import CitationPredictionDataset
from torch.utils.data import DataLoader


class BiLSTM(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size, label_size, task, num_meta,
                 enable_meta_year, enable_meta_issarea, enable_meta_judge, device, lr=1e-4):
        assert num_meta == 3 or (num_meta == 0 and not enable_meta_year and not enable_meta_issarea and not enable_meta_judge)

        super().__init__()
        self.hidden_dim = hidden_dim
        self.device_type = device
        self.batch_size = batch_size
        self.lr = lr
        self.task = task
        self.num_meta = num_meta
        self.enable_meta_year = enable_meta_year
        self.enable_meta_issarea = enable_meta_issarea
        self.enable_meta_judge = enable_meta_judge
        
        # layers for text
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_1 = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim, bidirectional=True)

        # layers for metadata
        if enable_meta_judge:
            self.judge_embedding = nn.Embedding(367, 3)

        meta_dim = int(enable_meta_year) * 21 + int(enable_meta_issarea) * 17 + int(enable_meta_judge) * 3
        self.dense = nn.Linear(hidden_dim * 2 + meta_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_dim, label_size)
    
    def forward(self, x):
        if self.num_meta > 0:
            txt, meta = x[:, :-self.num_meta], x[:, -self.num_meta:]
        else:
            txt = x

        # BiLSTM layers
        out = self.embeddings(txt).permute(1, 0, 2)
        out, _ = self.lstm_1(out)
        out, _ = self.lstm_2(out)
        out = out[-1]

        # concatenate metadata
        if self.enable_meta_year:
            year = F.one_hot(meta[:, 0] - 1999, num_classes=21).float()
            out = torch.cat((out, year), dim=1)
        if self.enable_meta_issarea:
            issarea = F.one_hot(meta[:, 1], num_classes=17).float()
            out = torch.cat((out, issarea), dim=1)
        if self.enable_meta_judge:
            judge = self.judge_embedding(meta[:, 2] + 1)
            out = torch.cat((out, judge), dim=1)

        # classification head
        out = self.dropout(out)
        out = self.dense(out)
        out = torch.tanh(out)
        out = self.dropout(out)
        out = self.out_proj(out)
        return out
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.to(self.device_type)
        if self.task.startswith("cit_idx_multi"):
            y = y.to(self.device_type)
        else:
            y = y.view(self.batch_size,).to(self.device_type)

        logits = self.forward(x)

        if self.task.startswith("cit_idx_multi"):
            loss = F.binary_cross_entropy_with_logits(logits, y)
        else:
            loss = F.cross_entropy(logits, y)
        
        logs = {"loss": loss}
        batch_dictionary = {
            "loss": loss,
            "log": logs,
        }
        return batch_dictionary
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {
            'loss': avg_loss,
        }
        epoch_dictionary={
            'loss': avg_loss,
            'log': tensorboard_logs
        }
        return epoch_dictionary
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.to(self.device_type)
        if self.task.startswith("cit_idx_multi"):
            y = y.to(self.device_type)
        else:
            y = y.view(self.batch_size,).to(self.device_type)
        logits = self.forward(x)
        if self.task.startswith("cit_idx_multi"):
            loss = F.binary_cross_entropy_with_logits(logits, y)
        else:
            loss = F.cross_entropy(logits, y)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch        
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class CheckpointEveryNSteps(pl.Callback):
    def __init__(self, save_step_frequency):
        self.save_step_frequency = save_step_frequency
       
    def on_batch_end(self, trainer: pl.Trainer, pl_module):
        # save checkpoints
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 1:
            filename = f"epoch={epoch}_global_step={global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


def bilstm_eval_class(model, data_loader):
    # Evaluation for [LEGACY] task cit_class
    device = model.device_type
    batch_size = model.batch_size
    total = []
    predictions = []
    total_loss = 0.0
    for i, (x, labels) in enumerate(tqdm.tqdm(data_loader)):
        x = x.to(device)
        labels = labels.view(batch_size, ).to(device)
        outputs = model(x)
        loss = F.cross_entropy(outputs, labels)
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        predictions = np.append(predictions, preds.cpu().numpy())
        total = np.append(total, labels.cpu().numpy())

    loss = total_loss / len(data_loader)
    print(f"loss:{loss}")
    print(classification_report(total, predictions, output_dict=True))


def bilstm_eval_idx(model, data_loader):
    # produce predictions
    device = model.device_type
    batch_size = model.batch_size
    total_loss = 0.0
    preds = np.empty((0, 20), int)
    totals = []
    for i, (x, labels) in enumerate(tqdm.tqdm(data_loader)):
        x = x.to(device)
        labels = labels.view(batch_size, ).to(device)
        outputs = model(x)

        loss = F.cross_entropy(outputs, labels)
        total_loss += loss.item()

        _, classes = torch.topk(outputs, 20, dim=1)

        preds = np.append(preds, classes.cpu().numpy(), axis=0)
        totals = np.append(totals, labels.cpu().numpy())

    # calculate recall
    recall_1 = 0
    recall_3 = 0
    recall_5 = 0
    recall_20 = 0
    count_dict = defaultdict(int)
    predict_dict = defaultdict(int)
    for i in range(0, len(preds)):
        idx = preds[i].astype(int)
        label = int(totals[i])

        count_dict[str(label)] += 1
        predict_dict[str(idx[0])] += 1

        if label == idx[0]:
            recall_1 += 1
        elif label in idx[:3]:
            recall_3 += 1
        elif label in idx[:5]:
            recall_5 += 1
        elif label in idx:
            recall_20 += 1

    recall_1_ratio = recall_1 / (len(preds) + 0.0)
    recall_3 += recall_1
    recall_3_ratio = recall_3 / (len(preds) + 0.0)
    recall_5 += recall_3
    recall_5_ratio = recall_5 / (len(preds) + 0.0)
    recall_20 += recall_5
    recall_20_ratio = recall_20 / (len(preds) + 0.0)

    if '0' in count_dict:
        print(f"no citation count {count_dict['0']}/{len(preds)}")
    print(f"loss: {(total_loss + 0.0) / len(data_loader)}")
    print(f"{recall_1}, {recall_3}, {recall_5}, {recall_20}, {len(preds)}")
    print(f"recall@1: {recall_1_ratio}, recall@3: {recall_3_ratio}, recall@5: {recall_5_ratio}, recall@20: {recall_20_ratio}")
    return recall_1_ratio, recall_3_ratio, recall_5_ratio, recall_20_ratio


def bilstm_test_n_fold(num_fold, model, config_file, cvx, wrappedTokenizor, negative_sample_prob, meta, dataset_return_type):
    recall_1s = []
    recall_5s = []
    recall_20s = []

    for i in range(num_fold):
        test_chunk_fpath = f"{config_file['test_ids_fpath_prefix']}_fold_{i}.txt"
        test_dataset = CitationPredictionDataset(
            config_file['preprocessed_dir'],
            cvx,
            case_ids=None,
            case_ids_fpath=test_chunk_fpath,
            tokenizer=wrappedTokenizor,
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
        test_loader = DataLoader(
            test_dataset,
            batch_size=config_file['batch_size'],
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
        recall_1_ratio, recall_3_ratio, recall_5_ratio, recall_20_ratio = bilstm_eval_idx(model, test_loader)

        recall_1s.append(recall_1_ratio * 100)
        recall_5s.append(recall_5_ratio * 100)
        recall_20s.append(recall_20_ratio * 100)
    print(f'recall@1: {recall_1s}, mean: {np.mean(recall_1s)}, se: {np.std(recall_1s) / np.sqrt(num_fold)}')
    print(f'recall@5: {recall_5s}, mean: {np.mean(recall_5s)}, se: {np.std(recall_5s) / np.sqrt(num_fold)}')
    print(f'recall@20: {recall_20s}, mean: {np.mean(recall_20s)}, se: {np.std(recall_20s) / np.sqrt(num_fold)}')
