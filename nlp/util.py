import os
import json
from collections import defaultdict


def load_case_ids_from_file(partition_ids_fpath):
    '''load a list of case ids from a text file with one
    id per line'''
    partition_ids = set()
    with open(partition_ids_fpath) as f:
        for line in f:
            partition_ids.add(line.strip())
    return partition_ids


def test_sanity_metadata(dataset):
    print("Checking dataset...")
    count_dict = defaultdict(int)
    invalid_meta_ids = []

    for i in range(len(dataset)):
        try:
            txt, cit_indices, case_meta = dataset.get_processed_case_text(i, replace_citations=False)
        except:
            fname = dataset.fnames[i]
            with open(os.path.join(dataset.case_dir, fname)) as f:
                data = json.load(f)
            print(f'no metadata for case {data["bva_id"]}')
            invalid_meta_ids.append(data["bva_id"])
            continue

        if case_meta['year'] < 0:
            print(f'missing column in metadata for case {case_meta["id"]}, meta {case_meta}')
            invalid_meta_ids.append(case_meta["id"])
        else:
            count_dict[case_meta["issarea"]] += 1

    print(f'class count dict: {count_dict}')
    print(f'{len(invalid_meta_ids)}/{len(dataset)} with invalid metadat, ids {invalid_meta_ids}')


def partition_test_ids(test_ids_fpath, num_fold):
    with open(test_ids_fpath) as file:
        ids = file.read().splitlines()
    ids = list(set(ids))

    chunk_num = num_fold
    chunk_size = len(ids) // chunk_num + 1
    dir_path, fname = test_ids_fpath.rsplit('/', 1)
    for i in range(chunk_num):
        chunk_ids = ids[i * chunk_size: min((i + 1) * chunk_size, len(ids))]
        print(f'fold {i} num {len(chunk_ids)} data')
        with open(os.path.join(dir_path, f'test_data_ids_fold_{i}.txt'), 'w') as file:
            file.write('\n'.join(chunk_ids))
