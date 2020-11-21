
import os
import random
from random import shuffle
import string

from pathlib import Path
import pickle
import numpy as np


class Dataset:
    def __init__(self, work_dir, data_dir, q_filter=None, eval_sz=0.05):

        self.wd = work_dir                          # Working dir
        self.rd = data_dir                          # Raw data folder
        self.q_filter = q_filter
        self.t = {}                                 # Vocab - Char to Id
        self.idx = {}                               # Vocab - Id to Char
        self.eval_sz = eval_sz                      # Size of eval data
        self.td = Path(self.wd, 'train')            # Save Training files
        self.ed = Path(self.wd, 'eval')             # Save Eval files 
        self.src_max = 160                          # Max source chars
        self.trg_max = 32                           # Max target chars 
        self.seed = 42                              # Seed for dataset split
        self.vocab_len = 98

    # Creates training and eval dataset and tokenizer
    def run(self):

        random.seed(self.seed)

        if not os.path.isdir(self.td):
            os.mkdir(self.td)
            train_files = list(Path(self.rd).rglob("*.txt"))
            self.create_smaller_files(train_files, self.td)
        
        if not os.path.isdir(self.ed):
            os.mkdir(self.ed)
            self.create_evaluation_data()

        self.create_tokenizer()    

    # Create a tokenizer based on the keras tokenizer
    def create_tokenizer(self, combined_vocab=True):

        vocab = ['pad', 'start', 'end']
        vocab = vocab + list(' ' + string.punctuation + string.digits + string.ascii_uppercase + string.ascii_lowercase)
        self.vocab_len = len(vocab)
        self.t = {k: v for v, k in enumerate(vocab)}
        self.idx = {v: k for k, v in self.t.items()}

    # Create an evaluation split based on eval_sz
    def create_evaluation_data(self):
        train_files = list(Path(self.td).rglob("*.txt"))
        shuffle(train_files)
        eval_idx = int(len(train_files) * self.eval_sz)
        source_files = train_files[:eval_idx]
        for fn in source_files:
            os.rename(fn, self.ed/os.path.basename(fn))

    # Tokenize and add start and end token if it's the target        
    def tokenize(self, sentence, vocab, isTrg):

        tokenized = [vocab[tok] for tok in sentence]
        tokenized = [1] + tokenized + [2] if isTrg else tokenized

        return tokenized
    
    # Create padding according to max len
    def padding(self, tokens, max_len):
        
        padded_tokens = [0] * max_len
        padded_tokens[:len(tokens)] = tokens
        return padded_tokens

    # Split a large file into small files
    def create_smaller_files(self, files, out_dir, lines_per_file=5000):
  
        for fname in files:
            smallfile = None
            if not self.q_filter or os.path.basename(fname) in self.q_filter:
                with open(fname) as bigfile:
                    for lineno, line in enumerate(bigfile):
                        if lineno % lines_per_file == 0:
                            if smallfile:
                                smallfile.close()
                            problem = os.path.basename(fname)[:-4] + '_{}.txt'
                            small_filename = problem.format(lineno + lines_per_file)
                            smallfile = open(out_dir/small_filename, "w")
                        smallfile.write(line)
                    if smallfile:
                        smallfile.close()


    # Create the main data stream for training 
    def generator(self, data_source, padding=True, tokenize=True):

        while(True):
            random.seed()
            files = list(Path(data_source).rglob("*.txt"))
            shuffle(files)

            for fname in files:
                with open(fname, "r") as file:
                    text = file.read()[:-1]
                    text_list = text.split('\n')
                    src_list = text_list[0:][::2]
                    trg_list = text_list[1:][::2]
                    
                    for src, trg in zip(src_list, trg_list):
                        if tokenize:
                            src = self.tokenize(src, self.t, isTrg=False)
                            trg = self.tokenize(trg, self.t, isTrg=True)
                        if padding:
                            src = self.padding(src, self.src_max)
                            trg = self.padding(trg, self.trg_max)
                        yield src, trg

    def batch_generator(self, data_source, bsz=5, padding=True, tokenize=True):

        gen = self.generator(data_source, padding, tokenize)

        while(True):

            src_batch = []
            trg_batch = []

            for _ in range(bsz):
                src, trg = next(gen)
                src_batch.append(src)
                trg_batch.append(trg)
            
            yield np.array(src_batch), np.array(trg_batch)













