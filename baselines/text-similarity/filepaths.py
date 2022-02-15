from os.path import join as pjoin

# Directories to be set
# TODO Set your paths accordingly
raw_dir = "/mnt/data-disk/raw/"
text_sim_dir = "/mnt/data-disk/text_sim/"
preprocessed_dir = '/mnt/data-disk/preprocessed/'
postprocessed_dir = '/mnt/data-disk/postprocessed/'
vocab_dir = '/mnt/data-disk/vocab/'
log_dir = '/mnt/data-disk/logs/'

# Files that need to be present
case_coll_dir = pjoin(raw_dir, "all-bva-decisions")
citation_dict_fpaths = [
        pjoin(raw_dir, 'cla_vet_app_metadata.jsonl'),
        pjoin(raw_dir, 'f3d.jsonl')
        ]
train_id_fpath = pjoin(raw_dir, "train_data_ids.txt")
dev_id_fpath = pjoin(raw_dir, "dev_data_ids.txt")
test_id_fpath = pjoin(raw_dir, "test_data_ids.txt")
case_id_fpaths = [train_id_fpath, dev_id_fpath, test_id_fpath]

# Files that will be generated
metadata_ids_fpath = pjoin(raw_dir, "metadata_ids.txt")
raw_vocab_fpath = pjoin(vocab_dir, "raw_vocab.pkl")
reduced_vocab_fpath = pjoin(vocab_dir, "reduced_vocab.pkl")
thresholded_vocab_fpath = pjoin(vocab_dir, "thresholded_vocab.pkl")
tokenizer_fpath = pjoin(vocab_dir, 'tfidf_tokenizer.pkl')
metadata_fpath = pjoin(text_sim_dir, "metadata.csv")
encoded_txt_fpath = pjoin(text_sim_dir, 'encoded_txt.hdf5')
 
