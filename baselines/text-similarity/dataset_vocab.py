from __future__ import annotations 
import json
from tqdm import tqdm
import os
import copy
import pickle
from collections import OrderedDict
import re
import random
import torch
import sentencepiece as spm
from typing import List, Dict, Tuple, Optional, Callable, Type
from torch.utils.data import Dataset
random.seed(42)

# Example citation data structure in CLA
#{'id': 6053943, 
# 'url': 'https://api.capapi.org/v1/cases/6053943/', 
# 'name': 'Michael ARZIO, Claimant-Appellant, v. Eric K. SHINSEKI, Secretary of Veterans Affairs, Respondent-Appellee', 
# 'name_abbreviation': 'Arzio v. Shinseki', 
# 'decision_date': '2010-04-19', 
# 'docket_number': 'No. 2009-7107', 
# 'first_page': '1343', 'last_page': '1348', 
# 'citations': [{'cite': '602 F.3d 1343', 'type': 'official'}], 
# 'volume': {'volume_number': '23', 'barcode': '32044132280314', 'url': 'https://api.capapi.org/v1/volumes/32044132280314/'}, 
# 'reporter': {'id': 938, 'full_name': "West's Veterans Appeals Reporter", 'url': 'https://api.capapi.org/v1/reporters/938/'}, 
# 'court': {'name_abbreviation': 'Fed. Cir.', 'name': 'United States Court of Appeals for the Federal Circuit', 'id': 8955, 'slug': 'fed-cir', 'url': 'https://api.capapi.org/v1/courts/fed-cir/'}, 
# 'jurisdiction': {'name': 'U.S.', 'id': 39, 'slug': 'us', 'name_long': 'United States', 'url': 'https://api.capapi.org/v1/jurisdictions/us/', 'whitelisted': False}, 
# 'cites_to': [{'cite': '35 F.3d 1516'}, {'cite': '327 F.3d 1371'}, {'cite': '590 F.3d 1317'}, {'cite': '120 F.3d 1239'}, {'cite': '100 S.Ct. 1747'}, {'cite': '67 L.Ed.2d 1'}, {'cite': '64 L.Ed.2d 381'}, {'cite': '119 L.Ed.2d 157'}, {'cite': '446 U.S. 398'}, {'cite': '101 S.Ct. 836'}, {'cite': '330 F.3d 1345'}, {'cite': '480 F.3d 1111'}, {'cite': '584 F.3d 1379'}, {'cite': '504 U.S. 374'}, {'cite': '301 F.3d 1354'}, {'cite': '450 U.S. 1'}, {'cite': '2009 WL 799554'}, {'cite': '112 S.Ct. 2031'}, {'cite': '34 F.3d 1039'}, {'cite': '451 F.3d 1331'}], 
# 'frontend_url': 'https://cite.capapi.org/f3d/602/1343/', 'preview': []}

# place holders
CITATION_PLACE_HOLDER = '@cit@'
PARA_BREAK_TOKEN = '@pb@'

# unknown/special tokens
NO_CITATION_TOKEN = 'NOCIT'
UNKNOWN_CITATION_TOKEN = 'UNKCIT'
NORMAL_CODE_PREFIX = 'NC: '
NORMAL_REG_PREFIX = 'NR: '
NORMAL_CASE_PREFIX = 'NJ: '
UNKNOWN_CODE_TOKEN = NORMAL_CODE_PREFIX+UNKNOWN_CITATION_TOKEN
UNKNOWN_REG_TOKEN = NORMAL_REG_PREFIX+UNKNOWN_CITATION_TOKEN
UNKNOWN_CASE_TOKEN = NORMAL_CASE_PREFIX+UNKNOWN_CITATION_TOKEN

# small class prediction classes and indices
CLASS_NOCIT = 'nocit'
CLASS_UNK = 'unk'
CLASS_CODE = 'code'
CLASS_REG = 'reg'
CLASS_CASE = 'case'
CLASS_NOCIT_IDX = 0 
CLASS_UNK_IDX = 1
CLASS_CASE_IDX = 2
CLASS_CODE_IDX = 3
CLASS_REG_IDX = 4

# tokenizer modes
TOKENIZER_MISSING = 0
TOKENIZER_SENTENCE_PIECE = 1
TOKENIZER_WRAPPED_HF = 2


# ==============================================================================
# Main Class for Citation Vocabulary Collection and Processing
# ==============================================================================


class CitationVocabulary:
    
    def __init__(self):
        self.citation_counts = self.init_empty_vocabulary()
        self.transform_fns = []
        
    def init_empty_vocabulary(self) -> OrderedDict:
        cc = OrderedDict({NO_CITATION_TOKEN: 0,
                          UNKNOWN_CITATION_TOKEN: 0,
                          UNKNOWN_CODE_TOKEN: 0,
                          UNKNOWN_REG_TOKEN: 0,
                          UNKNOWN_CASE_TOKEN: 0})
        return cc
        
    def __len__(self):
        return len(self.citation_counts)
    
    def total_count(self):
        return sum(self.citation_counts.values())
    
    def citation_str_by_index(self, idx: int) -> str:
        '''get citation string by citation index'''
        return list(self.citation_counts)[idx]
    
    def citation_source_class_by_str(self, cit: str) -> str:
        '''get class of citation in vocabulary from string'''
        if cit.startswith(NORMAL_CODE_PREFIX):
            return CLASS_CODE
        elif cit.startswith(NORMAL_REG_PREFIX):
            return CLASS_REG
        elif cit.startswith(NORMAL_CASE_PREFIX):
            return CLASS_CASE
        else:
            return CLASS_UNK
    
    def citation_source_class_by_index(self, idx: int) -> str:
        '''get class of citation in vocabulary at index'''
        cit_str = list(self.citation_counts)[idx]
        return self.citation_source_class_by_str(cit_str)
        
    def normal_case_citation_stats(self) -> Tuple[int, int]:
        '''get number of unique normalized case citations in the vocabulary
        and their counts
        Returns: Tuple (<number>, <counts>)'''
        num = 0
        total_count = 0
        for cit, count in self.citation_counts.items():
            if cit.startswith(NORMAL_CASE_PREFIX):
                num += 1
                total_count += count
        return num, total_count
    
    def normal_code_citation_stats(self) -> Tuple[int, int]:
        '''get number of unique normalized USC citations in the vocabulary
        and their counts
        Returns: Tuple (<number>, <counts>)'''
        num = 0
        total_count = 0
        for cit, count in self.citation_counts.items():
            if cit.startswith(NORMAL_CODE_PREFIX):
                num += 1
                total_count += count
        return num, total_count
    
    def normal_reg_citation_stats(self) -> Tuple[int, int]:
        '''get number of unique normalized CFR citations in the vocabulary
        and their counts
        Returns: Tuple (<number>, <counts>)'''
        num = 0
        total_count = 0
        for cit, count in self.citation_counts.items():
            if cit.startswith(NORMAL_REG_PREFIX):
                num += 1
                total_count += count
        return num, total_count
    
    def normal_citation_stats(self) -> Tuple[int, int]:
        '''get number of unique normalized citations in the vocabulary
        and their counts across USC, CFR, and cases
        Returns: Tuple (<number>, <counts>)'''
        case_num, case_count = self.normal_case_citation_stats()
        code_num, code_count = self.normal_code_citation_stats()
        reg_num, reg_count = self.normal_reg_citation_stats()
        return case_num+code_num+reg_num, case_count+code_count+reg_count
        
    def code_citation_stats(self):
        '''get number of unique non-normalized, regex-detected USC 
        citations in the vocabulary and their counts
        Returns: Tuple (<number>, <counts>)'''
        num = 0
        total_count = 0
        for cit, count in self.citation_counts.items():
            if re.search(r'U[\.,]? ?S[\.,]? ?C[\.,]?', cit) is not None:
                num += 1
                total_count += count
        return num, total_count
    
    def reg_citation_stats(self):
        '''get number of unique non-normalized, regex-detected USC 
        citations in the vocabulary and their counts
        Returns: Tuple (<number>, <counts>)'''
        num = 0
        total_count = 0
        for cit, count in self.citation_counts.items():
            if re.search(r'C[\.,]? ?F[\.,]? ?R[\.,]?', cit) is not None:
                num += 1
                total_count += count
        return num, total_count
    
    def presumed_case_citation_stats(self) -> Tuple[int, int]:
        '''get number and counts of citations in vocabulary which are not
        regex-detected USC or CFR citations, and are hence presumed to be
        case citations'''
        code_num, code_count = self.code_citation_stats()
        reg_num, reg_count = self.reg_citation_stats()
        return (len(self)-code_num-reg_num), (self.total_count()-code_count-reg_count)
        
    
    def duplicate(self) -> CitationVocabulary:
        '''duplicate the citation vocabulary via a deepcopy'''
        return copy.deepcopy(self)

    def load_from_case_dir(self,
                           case_dir: str, 
                           case_ids: List[str] = None, 
                           case_id_fpath: str = None):
        '''Gather all unique citations from preprocessed json case files
        in a folder, optionally restricted to a set of case ids provided
        either as a list or as a one-id-per line file'''
        #if case_id_fpath is not None:
        #    case_ids = util.load_case_ids_from_file(case_id_fpath)
        if case_ids is not None:
            fnames = [f"{fn}.json" for fn in case_ids]
            #fnames = [fn for fn in fnames if fn[:-5] in case_ids]
        for fname in tqdm(fnames):
            with open(os.path.join(case_dir, fname)) as f:
                data = json.load(f)
                for ct in data['citation_texts']:
                    self.citation_counts[ct] = 1 + self.citation_counts.get(ct, 0)
                    
    def set_transform_fns(self, tfns: List[Callable[str, List[str]]]) -> None:
        '''set the vocabulary tranformation function sequence. transform functions
        take a raw string citation string and return a list of transformed citations
        that have been derived from the passed citation'''
        self.transform_fns = tfns
        
    def transform_citation(self, cit: str) -> List[str]:
        '''transform a citation using the sequence of transformation functions
        in transform_fns. produces a list of citations, as single citations may
        be broken apart in the process'''
        cits = [cit]
        for ctfn in self.transform_fns:
            transformed = []
            for cit in cits:
                transformed += ctfn(cit)
            cits = transformed
        return cits
                    
    def transform_vocabulary(self, log_save_path=None):
        '''transform all citations in the vocabulary using the sequence of
        transformation functions in transforms_fns and recompute all counts
        accordingly, as some citations may be broken into smaller components'''
        new_citation_counts = self.init_empty_vocabulary()
        transforms = {}
        for cit, count in tqdm(self.citation_counts.items()):
            trf = []
            for nc in self.transform_citation(cit):
                assert isinstance(nc, str), 'citation not of type string'
                new_citation_counts[nc] = count + new_citation_counts.get(nc, 0)
                trf.append(nc)
                transforms[cit] = trf
        if log_save_path:
            with open(log_save_path, 'wb') as f:
                pickle.dump(transforms, f)
        self.citation_counts = new_citation_counts

    def citation_index(self, cit: str) -> int:
        '''returns the index of a citation if it exists in the vocabulary,
        else None'''
        if cit in self.citation_counts:
            return list(self.citation_counts.keys()).index(cit)
        else:
            return list(self.citation_counts.keys()).index(UNKNOWN_CITATION_TOKEN)
        
    def citation_indices_from_raw(self, cit: str) -> List[int]:
        '''Transform a given (raw) citation and give a list of all indices of
        the citations produced thereby'''
        resolved = self.transform_citation(cit)
        return [self.citation_index(r) for r in resolved]

    def save_to_txt(self, fpath: str, alphabet_sort: str = False):
        '''save the vocabulary and counts to a text file. Optionally sorted
        by alphabestic index rather than counts'''
        cit_items = sorted(self.citation_counts.items(),
                          key=(lambda x: x[0]) if alphabet_sort else (lambda x: x[1]))
        with open(fpath, 'w') as f:
            for cit, count in cit_items:
                f.write(f'{count} | {cit}\n')
                
    def reduce_sparse_to_unknown(self, min_n: int):
        new_citation_counts = self.init_empty_vocabulary()
        for cit, count in self.citation_counts.items():
            if count < min_n:
                if cit.startswith(NORMAL_CODE_PREFIX):
                    new_citation_counts[UNKNOWN_CODE_TOKEN] += count
                elif cit.startswith(NORMAL_REG_PREFIX):
                    new_citation_counts[UNKNOWN_REG_TOKEN] += count
                elif cit.startswith(NORMAL_CASE_PREFIX):
                    new_citation_counts[UNKNOWN_CASE_TOKEN] += count
                else:
                    new_citation_counts[UNKNOWN_CITATION_TOKEN] += count
            else:
                new_citation_counts[cit] = count
        self.citation_counts = new_citation_counts
        
    def vocab_report(self):
        '''print a report of vocabulary statistics'''
        size = len(self)
        total_count = self.total_count()
        code_num, code_count = self.code_citation_stats()
        reg_num, reg_count = self.reg_citation_stats()
        normal_code_num, normal_code_count = self.normal_code_citation_stats()
        normal_reg_num, normal_reg_count = self.normal_reg_citation_stats()
        case_num, case_count = self.presumed_case_citation_stats()
        normal_case_num, normal_case_count = self.normal_case_citation_stats()
        normal_num, normal_count = self.normal_citation_stats()
        unk_count = self.citation_counts[UNKNOWN_CITATION_TOKEN]
        code_unk_count = self.citation_counts[UNKNOWN_CODE_TOKEN]
        reg_unk_count = self.citation_counts[UNKNOWN_REG_TOKEN]
        case_unk_count = self.citation_counts[UNKNOWN_CASE_TOKEN]
        total_unk_count = unk_count + code_unk_count + reg_unk_count + case_unk_count
        print(f'number / count')
        print(f'vocab all: {size} / {total_count}')
        print(f'vocab norm: {normal_num} / {normal_count}')
        print(f'norm coverage: {normal_num/size} / {normal_count/total_count}')
        print(f'unknown count: {unk_count}')
        print(f'unknown %: {total_unk_count/total_count}')
        print(f'code regex: {code_num} / {code_count}')
        print(f'code norm: {normal_code_num} / {normal_code_count}')
        print(f'code unk: {code_unk_count}')
        print(f'reg regex: {reg_num} / {reg_count}')
        print(f'reg norm: {normal_reg_num} / {normal_reg_count}')
        print(f'reg unk: {reg_unk_count}')
        print(f'case presumed: {case_num} / {case_count}')
        print(f'case norm: {normal_case_num} / {normal_case_count}')
        print(f'case unk: {case_unk_count}')
        stats = {'vocab all': [size, total_count],
                 'vocab norm': [normal_num, normal_count],
                 'norm coverage': [normal_num/size, normal_count/total_count],
                 'unknown count': unk_count,
                 'unknown %': total_unk_count/total_count,
                 'code regex': [code_num, code_count],
                 'code norm': [normal_code_num, normal_code_count],
                 'code unk': code_unk_count,
                 'reg regex': [reg_num, reg_count],
                 'reg norm': [normal_reg_num, normal_reg_count],
                 'reg unk': reg_unk_count,
                 'case presumed': [case_num, case_count],
                 'case norm': [normal_case_num, normal_case_count],
                 'case unk': case_unk_count
                }
        return stats
        

# ==============================================================================
# Processing Case Citations through CaseLawAccess
# ==============================================================================
                    

class CLACitationProcessor:
    '''Toolbox class around CaseLawAccess reporter exports. After loading
    CLA data, the object provides a `resolve_case_citation` method which
    can be used as a citation transformation in a CitationVocabulary
    object for purposes of normalizing case citations that can be resolved
    to CLA cases that have been loaded.'''
    
    def __init__(self):
        self.cla_citations = []  # list to store all loaded CLA citations
        self.cla_ids = set()
        self.cit_regexes = [ # reporter-specific regexes to extract volume and first page elements
                            r'(?P<volume>\d+) [Vv]et\.? ?[Aa]pp\.? (at )?(?P<first_page>\d+)',
                            r'(?P<volume>\d+) [Ss]\.? ?[Cc]t\.? (at )?(?P<first_page>\d+)',
                            r'(?P<volume>\d+) [Ff]\.? ?2d\.? (at )?(?P<first_page>\d+)',
                            r'(?P<volume>\d+) [Ff]\.? ?3d\.? (at )?(?P<first_page>\d+)',
                            r'(?P<volume>\d+) [Ff]\.? ?[Aa]pp.x\.? (at )?(?P<first_page>\d+)',
                            ] 
        ## pre-compile to speed up preprocessing
        self.cit_regexes = [re.compile(regex) for regex in self.cit_regexes]
        
    def add_cla_citation(self, cla_c: dict):
        '''add a CLA case element to the preprocessor'''
        ## Note: this function has a fair amount of control flow for irregularities 
        # in CLA data
        
        ## F3d edge case: some last pages are `?`, which we do not load as we
        # cannot robustly resolve cases to them. we are generous here and discard
        # everything where last pages are not exclusively numbers
        if not re.match(r'\d+', cla_c['last_page']):
            return
        
        ## extract the single official citation from the CLA data
        official_cits = [cit['cite'] for cit in cla_c['citations'] 
                         if cit['type'] == 'official']
        assert len(official_cits) == 1, f'case {cla_c["id"]} has no single official citation'
        oc = official_cits[0]
        ## use regexes on official citations
        for cre in self.cit_regexes:
            m = cre.search(oc)
            if m is not None:
                ## determine first and last page
                ## save volume and reporter regex specific to the official citation
                cla_c['cit_volume'] = m.group('volume')
                cla_c['cit_regex'] = cre
                # some first_page last_page entries have hyphenated 'XXX-XXX' ranges
                # take the first/second number of the range for a clean entry
                n1 = re.match(r'(?P<first_first_page>\d+)-\d+', cla_c['first_page'])
                if n1 is not None:
                    cla_c['cit_first_page'] = n1.group('first_first_page')
                else:
                    cla_c['cit_first_page'] = cla_c['first_page']
                n2 = re.match(r'\d+-(?P<second_last_page>\d+)', cla_c['last_page'])
                if n2 is not None:
                    cla_c['cit_last_page'] = n2.group('second_last_page')
                else:
                    cla_c['cit_last_page'] = cla_c['last_page']
        ## mark if citation did not respond to any regex
        if 'cit_regex' not in cla_c:
            cla_c['cit_regex'] = None
        cla_c['official_citation'] = oc
        self.cla_citations.append(cla_c)
        self.cla_ids.add(cla_c['id'])
        
    def __len__(self):
        return len(self.cla_citations)

    def load_cla_citation_dictionary(self, fpath: str):
        '''loads all citations in a CaseLawAccess citation dictionary JSONL file'''
        with open(fpath, 'r') as f:
            for line in f.readlines():
                cit = json.loads(line)
                if not cit['id'] in self.cla_ids:
                    self.add_cla_citation(cit)
                else:
                    print(f'citation with id {cit["id"]} already present. skipping')
                    
    def resolve_case_citation(self, citation: str) -> List[str]:
        '''Citation transformation function to be used as part of the transform_fns
        sequence in a CitationVocabulary. Takes a raw citation and returns a
        single-element list with one element that is either the official CLA citation
        that has been prefix-marked as normalized, or the same raw citation'''
        for cla_c in self.cla_citations:
            if cla_c['cit_regex'] is not None:
                m = cla_c['cit_regex'].search(citation)
                if m is not None:
                    try:
                        volume_match = int(m.group('volume')) == int(cla_c['cit_volume'])
                        first_page_match = int(m.group('first_page'))\
                                           >= int(cla_c['cit_first_page'])
                        last_page_match = int(m.group('first_page'))\
                                          <= int(cla_c['cit_last_page'])
                        if volume_match and first_page_match and last_page_match:
                            return [NORMAL_CASE_PREFIX + cla_c['name_abbreviation']\
                                    +', '+cla_c['official_citation']
                                    + ', CLA#'+str(cla_c['id'])]
                    except ValueError:
                        print(f'error matching cit first page _{m.group("first_page")}_')
                        print(f'with reference first page _{cla_c["cit_first_page"]}_')
                        print(f'and reference last page _{cla_c["cit_last_page"]}_')
                        print(f'in CLA case {cla_c["id"]}')
        return [citation]


# ==============================================================================
# Citation Transformation functions
# ==============================================================================
                    
                    
def remove_trailing_parenth_years(citation):
    m = re.search(r'(?P<trail>\(([We]est\.? )?([Ss]upp\.? )?\d{4}\)?)$', citation)
    if m is not None:
        return [citation[:-(len(m.group('trail')))]]
    else:
        return [citation]


def remove_trailing_punctuation(citation):
    m = re.search(r'(?P<trail>[ \-,;]+)$', citation)
    if m is not None:
        return [citation[:-(len(m.group('trail')))]]
    else:
        return [citation]
    

usc_re = re.compile(r'^(?P<chapter>\d+)\s+U[\.,]?\s?S[\.,]?\s?C[\.,]?\s?(A[\.,]?\s?)?\s+(\xa7+ *)?(?P<tail>.+)$')
cfr_re = re.compile(r'^(?P<chapter>\d+)\s+C[\.,]?\s?F[\.,]?\s?R[\.,]?\s+(\xa7+\s*)?(?P<tail>.+)$')
tail_re = re.compile(r'\d[\d\w\.\(\)]*')   # has to start with number


def split_statutory_tail(tail: str) -> List[str]:
    tail = re.sub(r',', ' ', tail)
    tail = re.sub(r' +', ' ', tail).strip()
    return tail.split(' ')
    
    
def split_statutory_citations(citation: str) -> List[str]:
    new_cites = []
    for rex, code_norm in [[usc_re, 'USC'], [cfr_re, 'CFR']]:
        m = rex.match(citation)
        if m is not None:
            tails = split_statutory_tail(m.group('tail'))
            all_tails_ok = True
            for t in tails:
                if tail_re.match(t) is None:
                    all_tails_ok = False
                    break
            if all_tails_ok:
                for t in tails:
                    if code_norm == 'USC':
                        prefix = NORMAL_CODE_PREFIX
                    elif code_norm == 'CFR':
                        prefix = NORMAL_REG_PREFIX
                    nc = f'{prefix}{m.group("chapter")} {code_norm} {t}'
                    new_cites.append(nc)
                return new_cites
    return [citation]


# ==============================================================================
# Dataset Class
# ==============================================================================


class CitationPredictionDataset(Dataset):
    
    def __init__(self, case_dir, cit_vocabulary, 
                 case_ids=None, 
                 case_ids_fpath=None, 
                 tokenizer=None,
                 target_mode='binary',
                 ignore_unknown=True,
                 negative_sample_prob=.5,
                 add_case_meta=False,
                 forecast_length=16,
                 context_length=64,
                 pre_padding=False):
        fnames = [fn for fn in os.listdir(case_dir) 
                  if fn.lower().endswith('.json')]
        print(f'found {len(fnames)} case files in folder')
        if case_ids_fpath is not None:
            case_ids = util.load_case_ids_from_file(case_ids_fpath)
        if case_ids is not None:
            fnames = [fn for fn in fnames if fn[:-5] in case_ids]
        print(f'found {len(case_ids)} case ids')
        print(f'reduced to {len(fnames)} cases from id list')
        self.case_dir = case_dir
        self.fnames = fnames
        self.cit_vocabulary = cit_vocabulary
        self.tokenizer = tokenizer
        self.target_mode = target_mode
        self.add_case_meta = add_case_meta
        self.forecast_length = forecast_length
        self.context_length = context_length
        self.pre_padding = pre_padding
        self.negative_sample_prob = negative_sample_prob
        self.ignore_unknown=ignore_unknown
        self.is_cit_re = re.compile(r'@cit(\d+)@')
        self.tokenizer_mode = TOKENIZER_MISSING
        # VOCAB DATA CACHING FOR FASTER LOADING
        # citation class list in same order as citation count
        self.citation_source_class_index =\
            [self.cit_vocabulary.citation_source_class_by_index(i)
             for i in range(len(self.cit_vocabulary))]
            
        
    def __len__(self):
        return len(self.fnames)
    
    def get_processed_case_text(self, idx: int, replace_citations: bool = True):
        '''get the processed text of a case in the corpus at index idx
        Args:
            replace_citations: if True, replaces all generic citation tokens in the
            text with their respective indexed citation tokens. If False, the generic
            tokens will not be replaced and the function returns a list of citation
            indices instead, which can be replaced after tokenization for better
            performance
        Returns:
            if replace_citations == True:
                Tuple (<txt>, <case meta dict>)
            if replace_citations == False:
                Tuple (<txt>, <citation indices list> , <case meta dict>)
        '''
        fname = self.fnames[idx]
        with open(os.path.join(self.case_dir, fname)) as f:
            data = json.load(f)
        # citation indices are ints, with 0 being unknown
        citation_indices = [self.cit_vocabulary.citation_indices_from_raw(cit) 
                            for cit in data['citation_texts']]
        txt = data['txt']
        case_meta = {'id': data['bva_id'],
                     'judge': data.get('judge', None),
                     'year': data.get('year', None)}
        if not replace_citations:
            return txt, citation_indices, case_meta
        else:
            for cis in citation_indices:
                cit_string = ' '.join([f'@cit{ci}@' for ci in cis])
                txt = txt.replace(CITATION_PLACE_HOLDER, cit_string, 1)
            return txt, case_meta
    
    def __is_encoded_citation_token(self, t_idx: int, ignore_unknown: bool = False) -> bool:
        '''return true if this token index belongs to a citation. If
        `ignore_unknown` is set to true, then non-normalized citations
        (i.e. such with general unknown small class) are ignored'''
        # if we are dealing with a wrapped HF tokenizer, it is easy to tell non-citations
        if self.tokenizer_mode == TOKENIZER_WRAPPED_HF:
            if t_idx < self.tokenizer.first_custom_token_id:
                return False
        token = self.tokenizer.decode(t_idx)
        m = self.is_cit_re.match(token)
        if m is not None: # is citation
            if ignore_unknown: 
                ## if ignore_unknown, then only return true if citation is normalized
                cit_idx = int(m.group(1))
                return self.citation_source_class_index[cit_idx] != CLASS_UNK_IDX
            else:
                return True
        else:
            return False
    
    def __get_citation_token_vocabulary_index(self, t_idx: int) -> Optional[int]:
        '''for a token index, if it is a citation, return the citation index
        or None if it is not a citation.'''
        token = self.tokenizer.decode(t_idx)
        m = self.is_cit_re.match(token)
        if m is not None:
            cit_idx = int(m.group(1))
            return cit_idx
        return None
    
    def __get_citation_token_source_class(self, t_idx: int) -> Optional[int]:
        '''for a token index, if it is a citation, return its small class index
        or None if it is not a citation'''
        token = self.tokenizer.decode(t_idx)
        m = self.is_cit_re.match(token)
        if m is not None:
            cit_idx = int(m.group(1))
            cl = self.citation_source_class_index[cit_idx]
            if cl == CLASS_CASE:
                return CLASS_CASE_IDX
            elif cl == CLASS_CODE:
                return CLASS_CODE_IDX
            elif cl == CLASS_REG:
                return CLASS_REG_IDX
            elif cl == CLASS_UNK: ## unknown is 1, so 0 can stand for no citation
                return CLASS_UNK_IDX
        return None
    
    def __getitem__(self, idx: int):
        '''instance getter'''
        assert self.tokenizer is not None, 'dataset has no tokenizer'
        ## STEP 1: TOKENIZE CASE TEXT
        if self.tokenizer_mode == TOKENIZER_SENTENCE_PIECE:
            txt, case_meta = self.get_processed_case_text(idx)
            pad_token_id = self.tokenizer.pad_id() # Sentence Piece
            encoded = self.tokenizer.encode(txt)
        elif self.tokenizer_mode ==  TOKENIZER_WRAPPED_HF:
            txt, cit_indices, case_meta =\
                self.get_processed_case_text(idx, replace_citations=False)
            pad_token_id = self.tokenizer.wrapped_tokenizer.pad_token_id
            encoded = self.tokenizer.encode(txt, cit_indices)
        ## STEP 2: PAD
        if self.pre_padding:
            pre_pad_length = self.context_length
        else:
            # pad so that at least one datapoint can be extracted
            pre_pad_length = max(0, self.context_length
                                    + self.forecast_length
                                    - len(encoded))
        pre_padding = torch.tensor([pad_token_id] * pre_pad_length)
        post_padding = torch.tensor([pad_token_id] * self.forecast_length)
        if pre_pad_length>0:
            padded = torch.cat([pre_padding, torch.tensor(encoded), post_padding])
        else:
            padded = torch.cat([torch.tensor(encoded), post_padding])
        ## STEP 3: DETERMINE CITATION-CONTAINING OFFSETS
        min_offset = 0
        max_offset = len(padded)-self.forecast_length-self.context_length
        all_offsets = set(range(min_offset, max_offset))
        pos_offsets = set()
        for i in range(self.context_length, len(padded)):
            if self.__is_encoded_citation_token(padded[i].item(),
                                                ignore_unknown=self.ignore_unknown):
                new_pos = range(i-self.context_length-self.forecast_length+1,
                                i-self.context_length+1)
                # +1's b/c citation must be inside the forecasting window
                # only keep valid offsets; necessary to avoid overreach
                valid_new_pos = all_offsets & set(new_pos)
                pos_offsets = pos_offsets | valid_new_pos
        neg_offsets = all_offsets - pos_offsets
        ## STEP 4: CHOOSE AN OFFSET
        # edge case: text exclusively has pos/neg instances:
        if len(neg_offsets) == 0:
            offset_pool = pos_offsets
        if len(pos_offsets) == 0:
            offset_pool = neg_offsets
        # regular case: sample pos/neg offsets
        if self.negative_sample_prob is None:  # no special negative sampling
            offset_pool = all_offsets
        elif self.negative_sample_prob > 0:  # some negative sampling
            threshold = int(self.negative_sample_prob*100)
            if random.randint(1, 100) <= threshold:
                offset_pool = neg_offsets
            else:
                offset_pool = pos_offsets
        else: # no negative samples
            offset_pool = pos_offsets
        offset = random.sample(offset_pool, 1)[0]
        ## STEP 5: SEGMENT CONTEXT AND FORECASTING WINDOW
        forecast_start = offset + self.context_length
        context = padded[offset : forecast_start]
        forecast = padded[forecast_start : forecast_start+self.forecast_length]
        ## get citations in forecast window
        cit_tokens = [fc.item() for fc in forecast
                      if self.__is_encoded_citation_token(fc.item(),
                                                          ignore_unknown=self.ignore_unknown)]
        ## STEP 6: CHOOSE RETURN VALUE
        val = torch.tensor(encoded)
        if self.target_mode == 'binary':
            target = torch.tensor([0])
            if len(cit_tokens)>0:
                target = torch.tensor([1])
            val = context, target
        elif self.target_mode.startswith('cit_class'):
            target = torch.tensor([0])
            if len(cit_tokens)>0:
                cl = self.__get_citation_token_source_class(cit_tokens[0])
                target = torch.tensor([cl])
            val = context, target
        elif self.target_mode.startswith('cit_idx_multi'):
            target = torch.zeros((len(self.cit_vocabulary),))
            for i, ct in enumerate(cit_tokens):
                ci = self.__get_citation_token_vocabulary_index(ct, plus_one=False)
                target[ci] = 1.0
            val = context, target
        elif self.target_mode.startswith('cit_idx'):
            target = torch.tensor([0])
            if len(cit_tokens)>0:
                ci = self.__get_citation_token_vocabulary_index(cit_tokens[0])
                target = torch.tensor([ci])
            val = context, target
        if self.add_case_meta:
            val += (case_meta,)
        return val 
        
    def train_sp_encoder(self, vocab_size=24000, model_prefix='bva_enc'):
        print(f'training SentencePiece model on {len(self)} cases')
        case_txt_iter = iter([self.get_processed_case_text(i)[0] for i in range(len(self))])
        citation_tokens = [f'@cit{i}@' for i in range(len(self.cit_vocabulary))]
        spm.SentencePieceTrainer.train(sentence_iterator=case_txt_iter, 
                                       vocab_size=vocab_size,
                                       model_prefix=model_prefix,
                                       user_defined_symbols=citation_tokens)
        
    def load_sp_model(self, fpath='bva_enc.model'):
        self.tokenizer = spm.SentencePieceProcessor(model_file=fpath)
        self.tokenizer_mode = TOKENIZER_SENTENCE_PIECE
        
    def wrap_huggingface_tokenizer(self, tokenizer):
        self.tokenizer = WrappedCitationTokenizer(tokenizer, self.cit_vocabulary)
        self.tokenizer_mode = TOKENIZER_WRAPPED_HF
        
    def total_vocab_size(self):
        return len(self.tokenizer)
        
        
class WrappedCitationTokenizer:
    
    def __init__(self, tokenizer, cit_vocabulary, 
                 cit_token=CITATION_PLACE_HOLDER,
                 pb_token=PARA_BREAK_TOKEN):
        self.wrapped_tokenizer = tokenizer
        self.cit_vocabulary = cit_vocabulary
        self.wrapped_tokenizer.add_tokens([pb_token])
        self.wrapped_tokenizer.add_tokens([cit_token])  ## add cit placeholder last
        self.pb_token_id = self.wrapped_tokenizer.convert_tokens_to_ids([pb_token])[0]
        self.cit_token_id = self.wrapped_tokenizer.convert_tokens_to_ids([cit_token])[0]
        self.first_custom_token_id = self.pb_token_id
        # it is assumed that cit_token_id is the maximal token index
        # in the encoder vocabulary and the paragraph break token is the first
        # custom token. If this is ever not the case, this
        # code needs to be adapted and resort to the vocabulary size instead
        
    def __len__(self):
        return len(self.wrapped_tokenizer)+len(self.cit_vocabulary)

    def encode(self, txt, cit_indices):
        token_ids = self.wrapped_tokenizer.encode(txt)
        replaced_citations = 0
        new_token_ids = []
        for i, tid in enumerate(token_ids):
            if tid == self.cit_token_id:
                cis = cit_indices[replaced_citations]
                new_token_ids += [self.cit_token_id+ci for ci in cis]
                replaced_citations += 1
            else:
                new_token_ids.append(tid)
        return new_token_ids
        
    def decode(self, token_indices):
        if isinstance(token_indices, int):
            token_indices = [token_indices]
        txt = ''
        for tid in token_indices:
            if tid == self.pb_token_id:
                txt += PARA_BREAK_TOKEN
            elif tid < self.cit_token_id:
                txt += self.wrapped_tokenizer.decode([tid])
            else:
                txt += f'@cit{tid-self.cit_token_id}@'
        return txt
        

class TfidfTokenizer:

    def __init__(self, 
                 processed_case_dir, 
                 tokenizer_fpath,
                 cit_vocabulary,
                 train_ids):
        self.train_ids = train_ids
        self.fnames = [os.path.join(processed_case_dir, f"{fn}.json") for
                        fn in self.train_ids]
        self.processed_case_dir = processed_case_dir 
        self.tokenizer_fpath = tokenizer_fpath
        self.cit_vocabulary = cit_vocabulary
        self.num_cits = len(cit_vocabulary)
        self.vectorizer = None

    def learn_top_tfidf(self, max_df=1.0, min_df=10, max_features=10000):
        # Add stopwords
        with open("stopwords.txt", "r") as f:
            stopwords = f.read().splitlines()
        for cit in range(self.num_cits):
            stopwords.append(f"cit{cit}")

        # Init TfidfVectorizer
        self.vectorizer = TfidfVectorizer(input='filename', 
                analyzer='word',
                stop_words=stopwords, 
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                max_df=max_df, min_df=min_df,
                max_features=max_features)

        # Fit
        self.vectorizer.fit(self.fnames)

    def save_vocab(self):
        # Add citations to vocab
        i=0
        self.vocab = {}
        for idx in range(self.num_cits):
            self.vocab[f"@cit{idx}@"] = i
            i+=1

        # Add vocab
        self.vocab["<unk>"] = i # Position 20608 for unknown tokens
        i+=1
        for v in self.vectorizer.get_feature_names():
            self.vocab[v] = i
            i+=1

        self.idx_to_word = [k for k in self.vocab.keys()]

    def save(self):
        with open(self.tokenizer_fpath, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def encode(self, txt):
        encoded = [self.vocab[w] if w in self.vocab else self.vocab["<unk>"] 
                    for w in txt.split()]
        return encoded

    def decode(self, encoded):
        decoded = [self.idx_to_word[idx] for idx in encoded]
        return decoded

    def open_h5file(self, fpath):
        self.h5file = h5py.File(fpath, mode='w')

    def close_h5file(self):
        self.h5file.close()

    def encode_file_and_save(self, fname):
        with open(fname, "r") as f:
            encoded = self.encode(f.read())
        dataset_name = Path(fname).stem
        self.h5file.create_dataset(dataset_name, data=np.array(encoded))

    def encode_files(self, case_folder):
        fnames = glob.glob(os.path.join(case_folder, "*.json"))
        for fname in tqdm(fnames):
            self.encode_file_and_save(fname)
        
