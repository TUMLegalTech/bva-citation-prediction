from dataset_vocab import WrappedCitationTokenizer
from transformers import RobertaTokenizerFast
import pickle
import random
import re
from tqdm import tqdm


class DistanceBucket:

    def __init__(self, start, length):
        self.min = start
        self.max = start+length-1
        self.instances = []

    def applies(self, distance):
        return self.min <= distance <= self.max

    def add(self, distance, rank):
        self.instances.append((distance, rank))

    def __len__(self):
        return len(self.instances)

    def recall_at_k(self, k):
        if len(self.instances) == 0:
            return 0.0
        return len([_ for _, rank in self.instances if rank < k]) / len(self.instances)


stats_fpath = '../../../data/bva/logs/roberta-idx-all-meta-f128-c256_converged_v4_valdata_prediction_stats_latest.csv'
forecast_csv_fpath = '../../../data/bva/logs/roberta-idx-all-meta-f128-c256_converged_v4_valdata_prediction_forecast_analysis.csv'
forecast_metrics_fpath = '../../../data/bva/logs/roberta-idx-all-meta-f128-c256_converged_v4_valdata_prediction_forecast_metrics.csv'
forecast_log_fpath = '../../../data/bva/logs/roberta-idx-all-meta-f128-c256_converged_v4_valdata_prediction_forecast_analysis.log'
cvx_path = '../../../data/bva/vocab/vocab_norm_min20_v4.pkl'
num_buckets = 8
bucket_size = 16

with open(cvx_path, 'rb') as f:
    cvx = pickle.load(f)
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
wtokenizer = WrappedCitationTokenizer(tokenizer, cvx)

print('counting instances')
n = 0
with open(stats_fpath) as f:
    _ = f.readline()  ## skip first line with csv header
    for _ in f:
        n+= 1

buckets = [DistanceBucket(i*bucket_size, bucket_size) for i in range(num_buckets)]

with open(stats_fpath) as f,\
        open(forecast_csv_fpath, 'w') as g,\
        open(forecast_log_fpath, 'w') as h,\
        open(forecast_metrics_fpath, 'w') as m:
    print(f'doing analysis')
    g.write('idx;target;target_type;target_distance;pred;pred_type;target_rank;target_distance\n')
    m.write('bucket;min;max;n;r@1;r@5;r@20\n')
    _ = f.readline()  ## skip first line with csv header
    num_empty_forecasts = 0
    for i, line in tqdm(enumerate(f)):
        year, issarea, judge, label, pred, rank, context, forecast, preds = line.split(';')
        label = int(label)
        pred = int(pred)
        rank = int(rank)
        context_ids = [int(i) for i in context.split(',')]
        forecast_ids = [int(i) for i in forecast.split(',')]
        label_token_id = wtokenizer.token_id_for_citation_id(label)
        if label_token_id not in forecast_ids:
            forecast = wtokenizer.decode(forecast_ids)
            h.write(f'IDX: {i}\n')
            h.write(forecast+'\n')
            h.write(f'missing: {label} / {label_token_id}\n\n\n')
            num_empty_forecasts += 1
        else:
            label_distance = forecast_ids.index(label_token_id)
            label_class = cvx.citation_source_class_by_index(label)
            pred_class = cvx.citation_source_class_by_index(pred)
            for b in buckets:
                if b.applies(label_distance):
                    b.add(label_distance, rank)
            g.write(f'{i};{label};{label_class};{label_distance};{pred};{pred_class};{rank}\n')
    print('exporting buckets')
    for i, b in enumerate(buckets):
        r1 = round(b.recall_at_k(1)*100, 1)
        r5 = round(b.recall_at_k(5)*100, 1)
        r20 = round(b.recall_at_k(20)*100, 1)
        m.write(f'{i};{b.min};{b.max};{len(b)};{r1};{r5};{r20}\n')

print(f'# data points: {n}')
print(f'# empty forecast windows: {num_empty_forecasts}')

