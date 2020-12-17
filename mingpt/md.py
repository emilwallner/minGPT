import random
import numpy as np
import torch
import string
import os
import torch.nn as nn
from torch.nn import functional as F

class MemData:
    """ Clean data, tokenizer, helper functions """

    def __init__(self, mem_slots):
        self.mem_slots = mem_slots
        self.vocab = ['pad', 'answer', 'mem', 'mem-end', 'finish', 'right', 'wrong'] + list(' ' + string.punctuation + string.digits + string.ascii_uppercase + string.ascii_lowercase)
        self.vocab_size = len(self.vocab) 
        # Max input characters plus max answer characters, 160 + 32 in original dataset
        self.max_src = 0
        self.max_trg = 0
        self.debug = 1000
        self.split = 0.1
        self.block_size = 0
        self.t = {k: v for v, k in enumerate(self.vocab)} # Character to ID
        self.idx = {v: k for k, v in self.t.items()} # ID to Character
    
    def initiate_mem_slot_data(self, fname):
        # split up all addition problems into either training data or test data    
        head_tail = os.path.split(fname) 
        src, trg = [], []
        with open(fname, "r") as file:
            text = file.read()[:-1] # Excluding the final linebreak
            text_list = text.split('\n')
            src = text_list[0:][::2]
            trg = text_list[1:][::2]
        
        data_len = len(src)
        if self.debug: data_len = self.debug
        r = np.random.RandomState(1337) # make deterministic
        perm = r.permutation(data_len) # Create random indexes
        num_test = int(data_len*self.split) # 10% of the whole dataset
        
        test_indexes = self.sort_data_by_len(perm[:num_test], src)
        train_indexes = self.sort_data_by_len(perm[num_test:], src)
        
        os.remove(fname)
        test_fname = head_tail[0] + '/test_' + head_tail[1]
        train_fname = head_tail[0] + '/train_' + head_tail[1]
        
        self.create_new_file(test_fname, src, trg, test_indexes)
        self.create_new_file(train_fname, src, trg, train_indexes)
    
    def create_new_file(self, fname, src, trg, indexes):
        with open(fname, "a") as file:
            for index in indexes:
                file.write(src[index] + '\n')
                file.write(trg[index] + '\n')
                if self.mem_slots:
                    for _ in range(self.mem_slots + 1):
                         file.write('\n')
    
    
    def prepare_data(self, fname):
        # split up all addition problems into either training data or test data
        # head_tail = os.path.split(fname)
        slots = (self.mem_slots + 3) if self.mem_slots else 2
        dataset = []
        for _ in range(slots):
            dataset.append([])
        with open(fname, "r") as file:
            text = file.read()[:-1] # Excluding the final linebreak
            text_list = text.split('\n')
            for i in range(slots):
                dataset[i] = text_list[i:][::slots]
        
        if not self.max_src:
            self.max_src = len(max(dataset[0], key=len)) + 1# +1 for ending token
            self.max_trg = len(max(dataset[1], key=len)) + 1 # +1 for ending token

        # Src tokens, target tokens, memory tokens, and corresponding end tokens
        # An extra for right/wrong token, and one for end of memory
        self.block_size = self.max_src + (self.max_trg * (self.mem_slots + 1)) + 2 
        return dataset
    
    def sort_data_by_len(self, indexes, data):
        test_data_by_length = []
        for index in indexes:
            test_data_by_length.append([index, len(data[index])])
        test_data_by_length = sorted(test_data_by_length, key=lambda x: x[1])
        return [i[0] for i in test_data_by_length]
    
    def sort_data_by_memory_len(self, indexes, data):
        test_data_by_length = []
        for index in indexes:
            test_data_by_length.append([index, len(data[index])])
        test_data_by_length = sorted(test_data_by_length, key=lambda x: x[1])
        return [i[0] for i in test_data_by_length]
    
    def create_math_data(self, data):
        src = list(data[0]) + ['answer'] 
        trg = list(data[1]) + ['finish']
        mem = []
        if self.mem_slots:
            for item in data[3:]:
                mem += list(item) + ['mem']
            mem += ['mem-end']
            
        return src, mem, trg
    
    def create_marker_data(self, data):
        src = list(data[0]) + ['answer']
        mem = []
        if self.mem_slots:
            for item in data[3:]:
                if item != data[2]: mem += list(item) # Remove prediction from memory
                mem += ['mem']
        
        if(bool(random.getrandbits(1))): # Randomly choose right or wrong example
            trg = list(data[1]) + ['finish'] + ['right']
        else:
            trg = list(data[2]) + ['finish'] + ['wrong']

        return src, mem, trg
    
    def x2Canvas(self, src_mem_trg):
        x = [self.t['pad']] * self.block_size
        x[:len(src_mem_trg[:-1])] = src_mem_trg[:-1]
        return x
    
    def y2Canvas(self, src_mem_trg):
        y = [self.t['pad']] * self.block_size
        y[:len(src_mem_trg[1:])] = src_mem_trg[1:]
        return y
    
    def list2tokens(self, src_mem_trg):
        return [self.t[tok] for tok in src_mem_trg]
        
    def tensor2string(self, tensor):
        return ''.join([self.idx[tok] for tok in tensor.tolist()])
    
    def tensor2list(self, tensor):
        return [self.idx[tok] for tok in tensor.tolist()]
    
    def string2digits(self, string):
        return ''.join([self.t[tok] for tok in string])
    
    def mask_padding(self, digits):
        return [-100 if tok == self.t['pad'] else tok for tok in digits]
    
    def mask_question_memory(self, y, mask_len):
        y[:mask_len] = -100
        return y

    def locate_token(self, token, tensor):
        return None if self.t[token] not in tensor.tolist() else tensor.tolist().index(self.t[token])