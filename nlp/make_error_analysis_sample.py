from dataset_vocab import WrappedCitationTokenizer
from transformers import RobertaTokenizerFast
import pickle
import random
import re
from tqdm import tqdm

stats_fpath = '../../../data/bva/logs/roberta-idx-all-meta-f128-c256_checkpoint-49005_prediction_stats.csv'
export_correct_fpath = '../../../data/bva/logs/roberta-idx-all-meta-f128-c256_checkpoint-49005_prediction_analysis_correct_sample'
export_wrong_fpath = '../../../data/bva/logs/roberta-idx-all-meta-f128-c256_checkpoint-49005_prediction_analysis_wrong_sample'
cvx_path = '../../../data/bva/vocab/thresholded_vocab.pkl'
num_correct_samples = 200
num_wrong_samples = 200
seed = 42

random.seed(seed)
with open(cvx_path, 'rb') as f:
    cvx = pickle.load(f)
cits_decoded = list(cvx.citation_counts)

def decode_cit_inline(m):
    i = int(m.group(1))
    return f'[{i}: {cits_decoded[i]}]'

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
wtokenizer = WrappedCitationTokenizer(tokenizer, cvx)

print('counting instances')
correct_idxs = []
wrong_idxs = []
with open(stats_fpath) as f:
    for i, line in tqdm(enumerate(f)):
        year, issarea, judge, label, pred, position, context, forecast, preds = line.split(';')
        if label == pred:
            correct_idxs.append(i)
        else:
            wrong_idxs.append(i)

print(f'{len(correct_idxs)} correct instances found')
print(f'{len(correct_idxs)} wrong instances found')

print(f'sampling instances')
sample_correct_idxs = random.sample(correct_idxs, num_correct_samples)
sample_wrong_idxs = random.sample(wrong_idxs, num_wrong_samples)

print(f'compiling analysis sample')
with open(stats_fpath) as f,\
        open(export_correct_fpath+'.txt', 'w') as gc_txt,\
        open(export_correct_fpath+'.csv', 'w') as gc_csv,\
        open(export_wrong_fpath+'.txt', 'w') as gw_txt,\
        open(export_wrong_fpath+'.csv', 'w') as gw_csv:
    for csv_out in [gc_csv, gw_csv]:
        csv_out.write('idx;target;target_type;pred;pred_type;target_rank;in_window?\n')
    for i, line in tqdm(enumerate(f)):
        out_file_txt, out_file_csv = None, None
        if i in sample_correct_idxs:
            out_file_txt = gc_txt
            out_file_csv = gc_csv
        elif i in sample_wrong_idxs:
            out_file_txt = gw_txt
            out_file_csv = gw_csv
        if out_file_txt:
            year, issarea, judge, label, pred, rank, context, forecast, preds = line.split(';')
            label = int(label)
            pred = int(pred)
            context_ids = [int(i) for i in context.split(',')]
            context = wtokenizer.decode(context_ids)
            forecast_ids = [int(i) for i in forecast.split(',')]
            forecast = wtokenizer.decode(forecast_ids)
            pred_cit_token = f'@cit{pred}@'
            pred_in_forecast = 1 if re.search(pred_cit_token, forecast) else 0
            context = re.sub(r'@cit(\d+)@', decode_cit_inline, context)
            forecast = re.sub(r'@cit(\d+)@', decode_cit_inline, forecast)
            out_file_txt.write(f'#IDX: {i}\n')
            out_file_txt.write(f'#YEAR: {year}; #ISSAREA: {issarea}; #JUDGE: {judge}\n')
            out_file_txt.write(f'#CONTEXT:\n{context}\n')
            out_file_txt.write(f'#FORECAST:\n{forecast}\n')
            out_file_txt.write(f'#TRUE: [{label}] {cits_decoded[int(label)]}\n')
            out_file_txt.write(f'#PRED: [{pred}] {cits_decoded[int(pred)]}\n')
            out_file_txt.write(f'#TRUE RANK: {rank}\n')
            out_file_txt.write(f'#PRED IN FC: {pred_in_forecast}\n')
            out_file_txt.write(f'\n\n')
            label_class = cvx.citation_source_class_by_index(label)
            pred_class = cvx.citation_source_class_by_index(pred)
            out_file_csv.write(f'{i};{label};{label_class};{pred};{pred_class};{rank};{pred_in_forecast}\n')
