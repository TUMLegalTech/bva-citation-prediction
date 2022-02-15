# reference: https://huggingface.co/transformers/_modules/transformers/modeling_roberta.html#RobertaForSequenceClassification

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import hashlib
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaConfig, RobertaModel
from sklearn import metrics
from collections import defaultdict
from dataset_vocab import CitationPredictionDataset
from torch.utils.data import DataLoader


class RobertaPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        assert config.num_meta == 3 or (config.num_meta == 0 and not config.enable_meta_year and not config.enable_meta_issarea and not config.enable_meta_judge)
        meta_dim = int(config.enable_meta_year) * 21 + int(config.enable_meta_issarea) * 17 + int(config.enable_meta_judge) * 3
        self.dense = nn.Linear(config.hidden_size + meta_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
       # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_meta = config.num_meta
        self.enable_meta_year = config.enable_meta_year
        self.enable_meta_issarea = config.enable_meta_issarea
        self.enable_meta_judge = config.enable_meta_judge
        self.task = config.task
        self.run_config = config.run_config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if config.enable_meta_judge:
            self.embedding = torch.nn.Embedding(367, 3)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        forecast_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.num_meta > 0:
            input_ids, meta = input_ids[:, :-self.num_meta], input_ids[:, -self.num_meta:]
            attention_mask = attention_mask[:, :-self.num_meta]

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # take <s> token (equiv. to [CLS]), was in classifier, but take out to add meta data
        sequence_output = torch.squeeze(outputs[0][:, 0, :])

        if self.enable_meta_year:
            year = F.one_hot(meta[:, 0].long() - 1999, num_classes=21).float()
            sequence_output = torch.cat([sequence_output, year], dim=1)
        if self.enable_meta_issarea:
            issarea = F.one_hot(meta[:, 1].long(), num_classes=17).float()
            sequence_output = torch.cat([sequence_output, issarea], dim=1)
        if self.enable_meta_judge:
            judge = self.embedding(meta[:, 2].long() + 1)
            sequence_output = torch.cat([sequence_output, judge], dim=1)

        logits = self.classifier(sequence_output)

        if self.task.startswith("cit_idx_multi"):
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.view(-1, self.num_labels))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1,))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        ## write out metrics for analysis
        if ((self.run_config['mode'] in ['test', 'analysis']) and self.run_config['task'] == 'cit_idx_predictions'):
            if (self.run_config['predictions_analysis_file'] is not None):
                for i in range(len(labels)):
                    label = labels[i]
                    year = meta[:,0].tolist()[i]
                    issarea = meta[:,1].tolist()[i]
                    judge = meta[:,2].tolist()[i]
                    pred_indexes = sorted(range(len(logits[i])), key=lambda sub: logits[i][sub], reverse=True)
                    label_pos = pred_indexes.index(label)
                    pred_label = pred_indexes[0]
                    context_id_str = ','.join([str(i) for i in input_ids[i].tolist()])
                    forecast_id_str = ','.join([str(i) for i in forecast_ids[i].tolist()])
                    pred_list_str = ','.join([str(i) for i in pred_indexes])
                    log_str = f'{year};{issarea};{judge};{label};{pred_label};{label_pos};{context_id_str};{forecast_id_str};{pred_list_str}\n'
                    if (self.run_config['predictions_analysis_file'] is not None):
                        with open(self.run_config['predictions_analysis_file'], 'a') as f:
                            f.write(log_str)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def roberta_evaluate_class(result, error_distribution=None):
    preds = result.predictions
    preds = [np.argmax(raw_pred) for raw_pred in preds]
    labels = result.label_ids
    labels = [label for label in labels]
    if error_distribution != None:
        confusion_metrics = metrics.confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])
        print("confusion_metrics", confusion_metrics)

    result = metrics.classification_report(labels, preds,
                                           labels=[0, 1, 2, 3, 4],
                                           target_names=['No Citation', 'Other', 'case', 'code', 'reg'],
                                           digits=5, output_dict=True)

    print("report: ", result)
    return result


def roberta_evaluate_idx(result):
    preds = result.predictions
    labels = result.label_ids
    recall_1 = 0
    recall_5 = 0
    recall_3 = 0
    recall_20 = 0
    count_dict = defaultdict(int)
    with open('./preds.txt', 'w') as f:
        for i in range(0, len(preds)):
            raw_pred = list(preds[i])
            label = labels[i]
            count_dict[str(label)] += 1
            idx = sorted(range(len(raw_pred)), key=lambda sub: raw_pred[sub], reverse=True)[:20]
            f.write(str(label) + " " + ",".join([str(i) for i in idx]) + '\n')
            if label not in idx:
                continue
            elif label not in idx[:5]:
                recall_20 += 1
            elif label not in idx[:3]:
                recall_5 += 1
            elif label != idx[0]:
                recall_3 += 1
            else:
                recall_1 += 1

    recall_1_ratio = recall_1 / len(preds)
    recall_3 += recall_1
    recall_3_ratio = recall_3 / len(preds)
    recall_5 += recall_3
    recall_5_ratio = recall_5 / len(preds)
    recall_20 += recall_5
    recall_20_ratio = recall_20 / len(preds)

    return {'recall@1': recall_1_ratio, "recall@3": recall_3_ratio, "recall@5": recall_5_ratio,
            "recall@20": recall_20_ratio}


def roberta_test_n_fold(num_fold, trainer, config_file, cvx, wrappedTokenizor, negative_sample_prob, meta, dataset_return_type):
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
        predOutput = trainer.evaluate(test_dataset)
        if config_file['task'].startswith("cit_idx"):
            recall_1s.append(predOutput['eval_recall@1'] * 100)
            recall_5s.append(predOutput['eval_recall@5'] * 100)
            recall_20s.append(predOutput['eval_recall@20'] * 100)

    print(f'recall@1: {recall_1s}, mean: {np.mean(recall_1s)}, std: {np.std(recall_1s) / math.sqrt(num_fold)}')
    print(f'recall@5: {recall_5s}, mean: {np.mean(recall_5s)}, std: {np.std(recall_5s) / math.sqrt(num_fold)}')
    print(f'recall@20: {recall_20s}, mean: {np.mean(recall_20s)}, std: {np.std(recall_20s) / math.sqrt(num_fold)}')


def roberta_test_n_fold(num_fold, trainer, config_file, cvx, wrappedTokenizor, negative_sample_prob, meta, dataset_return_type):
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
        predOutput = trainer.evaluate(test_dataset)
        # Only outputs scores for cit_idx_predictions, as other tasks are LEGACY
        if config_file['task'].startswith("cit_idx"):
            recall_1s.append(predOutput['eval_recall@1'] * 100)
            recall_5s.append(predOutput['eval_recall@5'] * 100)
            recall_20s.append(predOutput['eval_recall@20'] * 100)

    print(f'recall@1: {recall_1s}, mean: {np.mean(recall_1s)}, std: {np.std(recall_1s) / math.sqrt(num_fold)}')
    print(f'recall@5: {recall_5s}, mean: {np.mean(recall_5s)}, std: {np.std(recall_5s) / math.sqrt(num_fold)}')
    print(f'recall@20: {recall_20s}, mean: {np.mean(recall_20s)}, std: {np.std(recall_20s) / math.sqrt(num_fold)}')


def roberta_model_analysis_val_data(trainer, config_file, cvx, wrappedTokenizor, negative_sample_prob, meta, dataset_return_type):
    recall_1s = []
    recall_5s = []
    recall_20s = []
    val_dataset = CitationPredictionDataset(
        config_file['preprocessed_dir'],
        cvx,
        case_ids=None,
        case_ids_fpath=config_file['dev_ids_fpath'],
        tokenizer=wrappedTokenizor,
        target_mode=config_file['task'],
        ignore_unknown=True,
        negative_sample_prob=negative_sample_prob,
        add_case_meta=config_file['add_case_meta'],
        meta=meta,
        forecast_length=config_file['forecast_length'],
        context_length=config_file['context_length'],
        pre_padding=False,
        return_type=dataset_return_type)
    predOutput = trainer.evaluate(val_dataset)
    # Only outputs scores for cit_idx_predictions, as other tasks are LEGACY
    if config_file['task'].startswith("cit_idx"):
        print(predOutput['eval_recall@1'] * 100)
        print(predOutput['eval_recall@5'] * 100)
        print(predOutput['eval_recall@20'] * 100)

