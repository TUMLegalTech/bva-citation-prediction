This is an archive of the results. Training data are sampled from 1000 documents, and the evaluation is performed on six folds of the test set. These results should be reproducible by running
```bash
# sample training data (svm_samples.csv)
python3 main.py --sample_only -d 1000 -m year issarea vlj
# perform training (svm_coef_*.txt) and evaluation (full_*.csv and small_class_*.csv)
python3 main.py --train_eval_only -n 6
python3 main.py --train_eval_only -n 6 -m year
python3 main.py --train_eval_only -n 6 -m year issarea
python3 main.py --train_eval_only -n 6 -m year issarea vlj
# majority vote results (full_majority.csv and small_class_majority.csv)
python3 majority_vote.py
```
