"""
Make data structures necessary for collab filtering.
These include an inverted index and a forward index.

Usage: python3 make_data_structures.py
"""
import datetime
import json
import math
import concurrent

import util


def worker_process(bva_id):
    citations = util.get_citations(bva_id, vocab)
    if citations is None:
        return None

    # generate a forward index entry
    tf_vec = {}
    for citation in citations:
        for cit in citation:
            tf_vec[cit] = tf_vec.get(cit, 0) + 1
    tf_vec_norm = math.sqrt(sum(x ** 2 for x in tf_vec.values()))

    # Example: {
    #     'citations': [[80], [650], [749, 324], [24], [749], [180]],
    #     'counter': {80: 1, 650: 1, 749: 2, 324: 1, 24: 1, 180: 1},
    #     'tf_vec_norm': 3.0
    # }
    fwd_idx = {
        'citations': citations,
        'counter': tf_vec,
        'tf_vec_norm': tf_vec_norm,
    }
    return bva_id, fwd_idx, citations


if __name__ == '__main__':
    print('Loading vocab and dataset...')

    vocab = util.get_vocab()
    train_ids = util.get_dataset('train')
    citation_inverted_list = {}
    citation_forward_index = {}
    finished_cnt = 0

    print('Starting processing...')
    start_time = datetime.datetime.now()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(worker_process, train_ids):
            if result is None:
                continue

            bva_id, fwd_idx, citations = result
            citation_forward_index[bva_id] = fwd_idx

            # update inverted list entries
            for citation in citations:
                for cit in citation:
                    if cit not in citation_inverted_list:
                        citation_inverted_list[cit] = {}
                    inverted_list = citation_inverted_list[cit]
                    inverted_list[bva_id] = inverted_list.get(bva_id, 0) + 1

            finished_cnt += 1
            if finished_cnt % 1000 == 0:
                elapsed_time = datetime.datetime.now() - start_time
                print('{} / {} finished ({} elapsed)'.format(finished_cnt, len(train_ids), elapsed_time))

    print('Writing to json files...')

    with open(util.FWD_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(citation_forward_index, f)

    with open(util.INV_LIST_PATH, 'w', encoding='utf-8') as f:
        json.dump(citation_inverted_list, f)
    
    print('Data structures completed!')
