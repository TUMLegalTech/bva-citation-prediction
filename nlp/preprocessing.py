import os
import re
import json
from tqdm import tqdm
import time
import unicodedata
import sentencepiece as spm
import util
import pandas as pd


citation_place_holder = ' @cit@ '
paragraph_place_holder = ' @pb@ '


class DataPreprocessor:
    
    def __init__(self):
        self.unqualified_bvaid = []
        self.citation_place_holder = citation_place_holder
        self.exclude_punc_ord = [ord('@'), ord('*')]
        self.paragraph_place_holder = paragraph_place_holder
        self.case_pattern = re.compile('(See\s|See,\se\.g\.,\s|See\salso|)([DeMcO\'Mac]*[A-Z][a-z]+\s[A-Z]?[a-z]*\s?v\.\s[A-Z][a-z]+[,\s]+|)(\d{1,4})\s(F.2d|F.3d|F.|Vet. App.|U.S.|F. Supp.|F.Supp.|F. Supp. 2d|F. Supp. 3d|F.Supp.2d|F.Supp.3d|F. Supp. 2d.|F. Supp. 3d.)(\s|\sat\s)(\d{1,4})[,\s]*(\d{1,4}[,\-\s]*\d*|)[,\-\s]*(\d*|)(\s?\([A-Za-z\.\s]*\d{4}\)|)')
        self.statute_pattern = re.compile('([0-9]{1,4}\s)(C\.F\.R\.|U\.S\.C\.A\.|U\.S\.C\.|C\. F\. R\.|U\. S\. C\.| U\. S\. C\. A\.|CFR|C F R|USC|U S C|USCA|U S C A)\s{0,4}(§|§§|\\xa7|\\xa7\\xa7|\\\\xa7|\\\\xa7\\\\xa7|)\s{0,4}(\d{1,4}[A-z]?[\.-]*\d{0,4}[A-z]?)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)([ \t,]?and[ \t,](?!38)|[ \t]?[, -][ \t]?|)(\d{1,4}[A-z]?[\.-]*\d{0,4}[A-z]?|)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)([ \t,]?and[ \t,](?!38)|[ \t]?[, -][ \t]*(?!38)|)(\d{1,4}[A-z]?[\.-]*\d{0,4}[A-z]?|)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)([ \t,]?and[ \t,]+(?!38)|[ \t]?[, -][ \t]*(?!38)|)(\d{1,4}[A-z]?[\.-]*\d{0,4}[A-z]?|)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)([ \t,]?and[ \t,]+(?!38)|[ \t]?[, -][ \t]*(?!38)|)(\d{1,4}[A-z]?[\.-]*\d{0,4}[A-z]?|)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)([ \t,]?and[ \t,]+(?!38)|[ \t]?[, -][ \t]*(?!38)|)(\d{1,4}[A-z]?[\.-]*\d{0,4}[A-z]?|)([ \t]?\([A-z]\),?[ \t]?|[ \t]?\(\d{1,3}\),?[ \t]?|[ \t]?\(i{1,3}v?i{0,3}\),?[ \t]?|)')
        self.multi_space_pattern = re.compile('\s{2,}')
        self.invalid_char_pattern = re.compile('[^A-Za-z0-9(),.!?\'\`@]')
        self.double_break = re.compile('\n{2,}')
        self.header_signal = re.compile('THE ISSUE|the issue')
        self.internal_ands_patt = re.compile('(?<=\d),?\s?and\s?(?=[0-9\.]+)|(?<=\)),?\s?and\s?(?=[0-9\.\(]+)|(?<=\d[A-Z]),?\s?and\s?(?=[0-9\.]+)')
        self.strip_trail_patt = re.compile('(?<=[\d\)])[\s,]{1,3}[A-z]+\s?$|(?<=\d)\s?[,-]+\s?$')
        self.orphan_subsections_patt = re.compile('\s?and\s\([A-z0-9]{1,3}\)\([A-z0-9]{1,3}\)|\s?,\s?\([A-z0-9]{1,3}\)\([A-z0-9]{1,3}\)|\s?and\s\([A-z0-9]{1,3}\)|\s?,\s?\([A-z0-9]{1,3}\)')
        
    def internal_and(self,s):
        return re.sub(self.internal_ands_patt, ', ', s)
    
    def strip_trail(self,s):
        s = re.sub('\.$| $','',s)
        s = s.strip()
        out= re.sub(self.strip_trail_patt,"",s)
        if re.search(self.strip_trail_patt,out):
            return re.sub(self.strip_trail_patt,"",out)
        return out
    
    def complete_orphan_subs(self,s):
        check = re.search(self.orphan_subsections_patt, s)
        if check:
            if check.span()[1] >= len(s):
                orphan = s[check.span()[0]:].strip()
                orphan = re.search('\([A-z0-9]{1,3}\)\([A-z0-9]{1,3}\)|\([A-z0-9]{1,3}\)',orphan).group()
                parent = s[:check.span()[0]].strip()
                sub_parent = re.search('\d[a-z\d\.\(\)-]+(?=\([A-z0-9]+\)$)',parent)
                if sub_parent:
                    orphan = sub_parent.group()+orphan
                    out = parent+ ', '+ orphan
                    return out
                else: return parent
            else:
                orphan = s[check.span()[0]:].strip()
                orphan = re.search('\([A-z0-9]{1,3}\)\([A-z0-9]{1,3}\)|\([A-z0-9]{1,3}\)',orphan).group()
                parent = s[:check.span()[0]].strip()
                post = s[check.span()[1]:].strip()
                sub_parent = re.search('\d[a-z\d\.\(\)-]+(?=\([A-z0-9]+\)$)',parent)
                if sub_parent:
                    orphan = sub_parent.group()+orphan
                    out = parent+ ', '+ orphan + post
                    return out
                else: return s
        else: return s
    
    def collapse_space(self,s):
        m = re.search('\d+[A-Z0-9\(\)]?[ \t]{1,2}\([A-z0-9]{0,3}\)(?=[, \t]{0,3})|\([A-z0-9]\)[ \t]{1,2}\([A-z0-9]{0,3}\)(?=[, \t]{0,3})',s)
        if m:
            s = re.sub('[ \t]{0,2}(?=\([A-z0-9]{0,3}\)[, \t]{0,3})','',s[:m.span()[1]]) + s[m.span()[1]:]
        return s
    
    def citeparse(self, components): 
        """Given a case citation, label its constituent components"""
        components = [item.strip() for item in components]
        signal, caption, volume, reporter, s, startpage, pin1, pin2, year = components
        out = {'signal': signal, # this is an indication of how the authority is being used
               'caption': caption, # title of the cited case
               'volume': volume, # volume of the book where the case is reported
               'reporter': reporter, # case reporter
               'startpage': startpage, # case start page in the reporter
               'pincites': [pin1,pin2], #specific reference in the reporter
               'yr': year, 
               'case': volume + ' ' + reporter + ' ' + startpage}
        return out
    
    def statparse(self, components): 
        components = [item.strip() for item in components]
        vol, code = components[:2]
        ss = components[2:]

        # Delete entries that are just punctuation
        #ss = [x for x in ss if x != '']
        ss = [x for x in ss if x != '']
        ss = [el for el in ss if not re.match('^\.*\s*-*$',el)]
        
        # First, merge any elements consisting entirely of a letter subsection indicator
        # [e.g. (a) or (b)] with the previous element and remove. Thus ['4456','(a)'] --> ['4456(a)']
        ss_prev = ss[:-1]
        to_delete = []
        for idx, el in enumerate(ss_prev):
            if re.search('\([A-z]\)|\(\d{1,3}\)|\(i{1,3}v?i{0,3}\)', el):
                to_delete.append(idx)
        for el in to_delete:
            ss[el] = ''.join([ss[el],ss_prev[el]])
            ss[el+1] = ''
        
        
        # Next, split any elements containing two subsections. Thus, ['4456,4432'] --> ['4456','4432']
        ss_clean = []
        for el in ss:
            if re.search(',',el):
                ss_clean.extend(','.split(el))
            else:
                ss_clean.append(el)
        # create a dict and count refs to each statute section
        out = {'statutes': [' '.join([vol, code, s]) for s in ss],
               'volume': vol,
               'code': code,
               # 'year': year,
               'sections': ss}
        return out
    
    def find_citation_by_pattern(self, text, citation_type, pattern, parser):
        return [{'position': m.span(), 'type': citation_type, **parser(m.groups())}\
                    for m in re.finditer(pattern, text)] 
    
    def clean_pipeline(self, citation_text):
        out = self.strip_trail(citation_text)
        out = self.collapse_space(out)
        out = self.complete_orphan_subs(out)
        out = self.internal_and(out)
        out = self.complete_orphan_subs(out)
        out = self.collapse_space(out)
        out = self.collapse_space(out)
        out = self.complete_orphan_subs(out)
        out = self.strip_trail(out)
        return out
    
    
    def extract_citations(self, document):
        # document: dict, processed_document: dict
        # extract citations from text, change original text to ' ### '
        
        # Find cases, then statutes
        # Remove \r, \n and replacement character to better match citations
        text = document['txt'].replace('\r', ' ').replace('\n', ' ').replace('\ufffd', ' ')
        text = self.multi_space_pattern.sub(' ', text)
        
        cases = self.find_citation_by_pattern(text, 'case', self.case_pattern, self.citeparse)
        statutes = self.find_citation_by_pattern(text, 'statute', self.statute_pattern, self.statparse)

        # Sort citations by start position
        citations = sorted(cases + statutes, key=lambda k: k['position'][0], reverse=False)
        text = list(text)
        citation_texts = [''.join(text[ct['position'][0]: ct['position'][1]]) for ct in citations]
        # strip whitespace
        citation_texts = [c.strip() for c in citation_texts]
        # run through cleaning pipeline
        citation_texts = [self.clean_pipeline(c) for c in citation_texts]
        document['citation_texts'] = citation_texts
        # change original text, citation looped from largest pos to smallest pos
        for ct in reversed(citations):
            text[ct['position'][0]: ct['position'][1]] = self.citation_place_holder
        document['txt'] = ''.join(text)
        return document
    
    def is_char_punctuation(self, char):
        """Taken from bert gihub repo: https://github.com/google-research/bert/"""
        """Checks whether `chars` is a punctuation character."""
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        cp = ord(char)
        if cp in self.exclude_punc_ord:
            return False
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False
    
    def run_split_on_punc(self, document):
        """Add spaces to punctuation on a piece of text."""
        chars = list(document)
        output = [" " + c + " " if self.is_char_punctuation(c) else c for c in chars]
        return ''.join(output)
    
    def remove_header(self, document):
        text = document['txt']
        try:
            text = "THE ISSUE" +re.split(self.header_signal, text, 1)[1]
        except:
            print("Header signal THE ISSUE not found in", document['bva_id'])
            return ''
        text = re.sub(self.double_break, self.paragraph_place_holder, text)
        document['txt'] = text
        return document
    
    def add_paragraph_sep(self, document):
        text = re.sub(self.double_break, self.paragraph_place_holder, document['txt'])
        document['txt'] = text
        return document
    
    def clean_text(self, document):
        # all periods and commas removed, may want to keep them
        text = document['txt'].strip().lower()
        text = self.invalid_char_pattern.sub(' ', text)
        text = self.run_split_on_punc(text)
        text = self.multi_space_pattern.sub(' ', text)
        document['txt'] = text
        return document

    def process_input_texts(self, input):
        doc = {}
        doc['txt'] = input
        doc = self.extract_citations(doc)
        doc = self.clean_text(doc)
        return doc

    def process_single_txt_file(self, in_fpath, out_fpath=None):
        bva_id = os.path.basename(in_fpath).split('.')[0]
        #print(f'processing {bva_id}')
        doc = {'bva_id': bva_id}
        with open(in_fpath, encoding='latin_1') as f:
            txt = '\n'.join([line.strip() for line in f.readlines()])
            doc['txt'] = txt
            #doc = self.remove_header_add_paragraph_holder(doc)
            doc = self.add_paragraph_sep(doc)
            doc = self.extract_citations(doc)
            doc = self.citation_vocab(doc)
            doc = self.clean_text(doc)
        if out_fpath is not None:
            with open(out_fpath, 'w') as f:
                f.write(json.dumps(doc))
        return doc
    
    def process_partition_txt_files(self, input_dir, output_dir, partition_ids):
        fnames = os.listdir(input_dir)
        for fname in tqdm([fname for fname in fnames if fname[:-4] in partition_ids]):
            in_fpath = os.path.join(input_dir, fname)
            out_fpath = os.path.join(output_dir, fname[:-3]+'json')
            self.process_single_txt_file(in_fpath, out_fpath=out_fpath)
        
    def citation_vocab(self, doc):
        # transform each citation extracted in to [citation_voc]
        # output [[citation_voc,...], [citation_voc, ...], ...]
        statue_clean = 'abcdefghijklmnopqrstuvwxyz) '
        citations = doc['citation_texts']
        citation_vocab = []
        for ct in citations:
            ct_outputs = self.find_citation_by_pattern(ct.rstrip(', '), 'statute', self.statute_pattern, self.statparse)
            if len(ct_outputs) > 0:
                statutes = ct_outputs[0]['statutes']
                statutes = [st[:st.find('(')]if st.find('(') != -1 else st for st in statutes]
                statutes = [st.rstrip(statue_clean) for st in statutes]
                statutes = [st.replace('U. S. C.', 'U.S.C.') for st in statutes]
                statutes = [st.replace('U. S. C. A.', 'U.S.C.A.') for st in statutes]
                statutes = [st.replace('C. F. R.', 'C.F.R.') for st in statutes]
                citation_vocab.append(statutes)
            else:
                ct_outputs = self.find_citation_by_pattern(ct, 'case', self.case_pattern, self.citeparse)
                names = ct_outputs[0]['caption'].split()
                case = ct_outputs[0]['case']
                if len(names) > 0:
                    citation_vocab.append(['{}_{}'.format(names[0], case)])
                else:
                    citation_vocab.append([case])
        doc['citation_vocab'] = citation_vocab
        return doc


class WordEncodingProcessor:
    def __init__(self, preprocessed_dir, encoding_dir, max_seq_len=4500):
        self.preprocessed_dir = preprocessed_dir
        self.encoding_dir = encoding_dir
        self.max_seq_len = max_seq_len
        self.merged_text_filename = 'merged_text.txt'
        self.merged_id_filename = 'merged_bva_id.txt'
        self.tokenized_text_filename = "_token.txt"
        
    def process_single_file(self, input_path, out_text_file, out_id_file):
        with tqdm.tqdm(total=os.path.getsize(input_path)) as pbar:
            with open(input_path, 'r') as input_file:
                for line in input_file:
                    doc = json.loads(line)
                    text = doc['txt']
                    if len(text.split()) > self.max_seq_len:
                        text = ' '.join(text.split()[:self.max_seq_len])
                    out_text_file.write(text + '\n')  
                    out_id_file.write(doc['bva_id'] + '\n')
                    pbar.update(len(line))
    
    def process(self):
        if not os.path.isdir(self.encoding_dir):
            os.mkdir(self.encoding_dir)
        else:
            print("Already exist input files for sentencepiece")
            return
        file_names = os.listdir(self.preprocessed_dir)
        output_text_path = os.path.join(self.encoding_dir, self.merged_text_filename)
        output_bva_id_path = os.path.join(self.encoding_dir, self.merged_id_filename)
        output_text = open(output_text_path, 'w')
        output_id = open(output_bva_id_path, 'w')
        
        for file_name in file_names:
            if file_name.startswith("a") != True and file_name.startswith("n") != True:
                continue
            print("file_name:", file_name)
            input_path = os.path.join(self.preprocessed_dir, file_name)
            print("File {} being preprocessed".format(file_name))
            self.process_single_file(input_path, output_text, output_id)
            print("File {} completed preprocessing".format(file_name))
            
        output_text.close()
        output_id.close()

    def piece2ids_spm(self, model="bpe.model"):
        sp = spm.SentencePieceProcessor()
        sp.Load(model)
        input_text_path = os.path.join(self.encoding_dir, self.merged_text_filename)
        output_tokens_path = os.path.join(self.encoding_dir, model + self.tokenized_text_filename)
        with open(output_tokens_path, "w") as out_fd:
            with open(input_text_path) as f:
                for line in f:
                    out_fd.write(str(sp.EncodeAsIds(line.strip('\n'))).strip('[').strip(']'))
                    out_fd.write('\n')


class MetadataProcessor:
    def build_metadata(self, metadata_fpath):
        # build metadata and return the processed metadata dict
        df_meta = pd.read_csv(metadata_fpath)
        df_meta['bva_id'] = df_meta['tiread2']
        df_meta = df_meta.drop_duplicates(subset='bva_id', keep='first').set_index('bva_id')
        df_meta['judge'] = df_meta.groupby('bfmemid').ngroup()
        df_meta = df_meta.fillna(-1)
        df_meta['year'] = df_meta['imgadtm'].str[0:4].astype(int)
        df_meta = df_meta.astype({'isscode': int, 'issprog': int, 'isslev1': int})
        df_meta['diagcode'] = self.__get_diag_code(df_meta['isslev2'])
        df_meta['issarea'] = df_meta.apply(
            lambda r: self.__get_metadata_issarea(r['issprog'], r['isscode'], r['isslev1'], r['diagcode']), axis=1)
        df_meta = df_meta[['judge', 'year', 'issarea', 'issprog', 'isscode', 'isslev1', 'diagcode']]
        return df_meta.to_dict('index')

    def __get_diag_code(self, isslev2):
        return [int(str(x)[0]) if x >= 0 else -1 for x in isslev2]

    def __get_metadata_issarea(self, progcode, isscode, isslev1, diagcode):
        if progcode != 2:
            return 0
        else:
            if isscode == 8:
                return 2
            elif isscode == 9:
                return 3
            elif isscode == 17:
                return 4
            elif isscode == 12:
                if isslev1 != 4:
                    return 11
                elif isslev1 == 4:
                    if diagcode == 5:
                        return 12
                    elif diagcode == 6:
                        return 13
                    elif diagcode == 7:
                        return 14
                    elif diagcode == 8:
                        return 15
                    elif diagcode == 9:
                        return 16
            elif isscode == 15:
                if isslev1 != 3:
                    return 5
                elif isslev1 == 3:
                    if diagcode == 5:
                        return 6
                    if diagcode == 6:
                        return 7
                    if diagcode == 7:
                        return 8
                    if diagcode == 8:
                        return 9
                    if diagcode == 9:
                        return 10
            else:
                return 1
