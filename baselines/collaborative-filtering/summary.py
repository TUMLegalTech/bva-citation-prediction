"""
When the evaluation is complete, run this to make rows for the LaTeX table
and perform significance testing.
"""
import argparse
import pandas as pd
import math
import os

from scipy.stats import ttest_rel


# read the csv results for the given setting
def get_results(args, setting):
    result_file_name = args.prediction_task + '_' + setting
    if args.run_id:
        result_file_name += '_' + args.run_id
    result_file_name = os.path.join(
        args.eval_result_folder, result_file_name + '.csv')

    if args.prediction_task == 'full':
        df = pd.read_csv(result_file_name, index_col=0)
    else:
        df = pd.read_csv(result_file_name).set_index(['variable', 'index'])
        df = df[df.index.get_level_values(1) != 'support']
    return df


# make rows for LaTeX tables
def make_latex_table(df, task):
    n_folds = df.shape[1] - 2
    latex = []

    def fmt(x, n_digits=1):
        return str(round(x * 100, n_digits)) + '\\%'

    # full prediction row format:
    # recall_1_mean (s.e.) recall_5_mean (s.e.) recall_20_mean (s.e.)
    if task == 'full':
        for _, row in df.iterrows():
            latex.append(fmt(row['mean']))
            latex.append('(' + fmt(row['std'] / math.sqrt(n_folds), 2) + ')')
        return ' & '.join(latex) + ' \\\\'

    # small class prediction row format (only use mean):
    # macro_f1 case_f1 case_p case_r code_f1 code_p code_r reg_f1 reg_p reg_r
    elif task == 'small_class':
        numbers = df['mean'].values
        latex.append(fmt(numbers[-1]))
        for number in numbers[:-1]:
            latex.append(fmt(number))
        return ' & '.join(latex) + ' \\\\'

    else:
        raise ValueError("Unknown prediction task: " + task)


# perform statistical significance tests on two samples
def test_significance(one_sample, another_sample):
    t, p_value = ttest_rel(one_sample, another_sample, axis=1)

    summary = pd.DataFrame()
    summary['mean_1'] = one_sample.mean(axis=1)
    summary['mean_2'] = another_sample.mean(axis=1)
    summary['t'] = t
    summary['p_value'] = p_value
    summary['sig'] = summary.p_value.apply(pvalue_to_sig)
    return summary


# helper function: get setting name from metadata names
def get_setting(args, i):
    return 'orig' if i == 0 else '_'.join(args.metadata[:i])


# helper function: turn p-value to significance symbols
def pvalue_to_sig(p_value):
    if p_value < .001:
        return '***'
    if p_value < .01:
        return '**'
    if p_value < .05:
        return '*'
    return '-'


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prediction_task", default='full', choices=['full', 'small_class'],
                        help='Which prediction results to do significance testing on.')
    parser.add_argument("-m", "--metadata", nargs='+', default=['year', 'issarea', 'vlj'],
                        help='Add metadata one by one. Must be a subset of {year, issarea, vlj}.')
    parser.add_argument("-r", "--run_id", default=None,
                        help='Suffix of the saved result files.')
    parser.add_argument("--eval_result_folder", default='results',
                        help='Path where evaluation results are saved.')
    args = parser.parse_args()

    # produce LaTeX style table rows for the results
    for i in range(1 + len(args.metadata)):
        setting = get_setting(args, i)
        print(setting)

        df1 = get_results(args, setting)
        print(make_latex_table(df1, args.prediction_task))

    # perform significance testing
    for i in range(1, 1 + len(args.metadata)):
        prev_setting = get_setting(args, i - 1)
        curr_setting = get_setting(args, i)
        df1 = get_results(args, prev_setting)
        df2 = get_results(args, curr_setting)
        print(prev_setting, 'vs.', curr_setting)
        print(test_significance(df1, df2))
