from torch.utils.data.dataloader import DataLoader
from mingpt.math_dataset import MathDataset
from tqdm import tqdm
import torch
from mingpt.utils import sample
import datetime
import numpy as np
import os

#remove
import random, string

class AdaptiveExaminer:
    """ Clean data, tokenizer, helper functions """

    def __init__(self, MD, ac=10, max_batch=358):
        # MemData, Trainer, and Dataset classes
        self.mem_slots = MD.mem_slots
        self.max_trg = MD.max_trg
        self.MD = MD
        self.max_batch_size = max_batch
        self.ac = ac
        
        # Batch settings
        self.batch_size = 1
        self.data_lines = 4
     
    
    def exam(self, fname, trainer, test=False, debug=0):
        
        self.test = test
        self.debug = debug
        self.fname = fname
        self.trainer = trainer
        self.create_filenames()
        self.initiate_at_start()
        
        for i in range(self.ac):
            self.iter = i
            self.one_loop()
            
        self.write_file(self, self.train_fn, self.tmp_buffer)
        
        if os.path.exists(self.tmp_fn):
            os.remove(self.tmp_fn)
        if self.test:
            if os.path.exists(self.correct_fn):
                os.remove(self.correct_fn)
        
        r = self.results
        print("Final score: %d/%d = %.2f%% correct" % (np.sum(r), len(r), 100*np.mean(r)))
        
    def one_loop(self):
        
        self.initiate_vars()
        fname = self.train_fn if self.iter == 0 else self.tmp_fn
        dataset = MathDataset(fname=fname, MD=self.MD, marker_data=0.0)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        if self.iter == 0: self.initiate_at_start()
        
        dataset_len = len(loader)
        pbar = tqdm(enumerate(loader), total=dataset_len)
        batch = []
        
        for b, (x, y) in pbar:
            
            batch = self.concatenate_batches(batch, x)
            if b+1 == dataset_len: self.predict = 1
        
            if self.predict:
                output = self.model_predict(batch)
                cut_input = self.get_input_len(batch[0]) + 1
                batch_sz = batch.size(0)
                
                for i in range(batch_sz):
                    clean_output = self.clean_output(batch[i], output[i], cut_input)
                    self.log_result(clean_output)
                
                # Log status
                pbar.set_description(f"Iiter {b} Score: {np.sum(self.tmp_r)}/{len(self.tmp_r)}")
            
            if self.debug and b+1 >= self.debug:
                break
        
        self.save_result_to_file()
                    

    def concatenate_batches(self, batch, x):
        
        new_len = self.get_input_len(x[0])
        
        # Start a new batch for the first iteration
        if batch == []: batch = x
        # Handle input that could not fit into a batch
        elif self.leftover != []:
            batch = self.leftover
            self.leftover = []
       
        # Concat input source with same length
        if self.len == new_len: batch = torch.cat((batch, x), 0)
        else: self.predict, self.leftover = 1, x
   
        self.len = new_len
        self.batches += 1
        
        if self.batches == self.max_batch_size: self.predict = 1
        
        return batch
        
    def model_predict(self, x_in):
        
        # Reset tracker variables
        self.predict, self.batches  = 0, 0
        
        # Prepare input
        cut_index = self.get_input_len(x_in[0]) + 1 
        x_cut = x_in[:, :cut_index]
        pred = x_cut.to(self.trainer.device)
        
        # Make prediction
        output = sample(self.trainer.model, pred, int(self.max_trg+1))
        return output
    
    def get_input_len(self, x):
        
        if self.mem_slots: cut_input = self.MD.locate_token('mem-end', x)
        else: cut_input = self.MD.locate_token('answer', x) 
        
        return cut_input
    
    def clean_output(self, x, pred, cut_input):
        
        # Extract src from data
        cut_src = self.MD.locate_token('answer', x)
        src =  x[:cut_src]
        
        # Extract trg from data
        cut_padding = self.MD.locate_token('pad', x)
        cut_input_2 = self.get_input_len(x) + 1
        trg = x[cut_input:cut_padding] # X does not have the 'finish' token
        
        # Extract prediction from data
        cut_pred = self.MD.locate_token('finish', pred)
        if cut_pred: 
            cut_pred = min(self.max_trg+1, cut_pred)
            mark = self.MD.idx[pred[cut_pred+1].tolist()]
        else: 
            mark = 'error'
            cut_pred = self.max_trg
        pred = pred[cut_input:cut_pred]
        pred = pred[pred > 3] # Filter tokens below 4
        
        # Translate the tensors to strings
        src = self.MD.tensor2string(src)
        trg = self.MD.tensor2string(trg)
        pred = self.MD.tensor2string(pred) 
        
        # Extract memory from data
        mem = []
        if self.mem_slots:
            mem = self.tensor2memory(x[cut_src+1:], pred)
        
        #Debug
        self.get_input_len
        
        if "mem" in trg:
            print(f"X: {self.MD.tensor2string(x)}\n")
            print(f"Org input: {cut_input}\n New input: {cut_input_2}\n")
            print(f"Src: {src}\nTrg: {trg}\nPred: {pred}\nMark:{mark}\nMem: {mem}\n")
        
        return [src] + [trg] + [pred] + [mark] + mem
    
    def tensor2memory(self, x, pred):
        
        mem = []
     
        for i in range(self.mem_slots):
            cut = self.MD.locate_token('mem', x)
            mem_string = self.MD.tensor2string(x[:cut])
            mem_string = mem_string[:self.max_trg+1] # Cut to max target
            mem.append(mem_string)
            x = x[cut+1:] # Remove mem token

        mem = mem if pred in mem else [pred] + mem[:-1]
        return mem

            
    def log_result(self, output):
        
        # Output = [src] + [trg] + [pred] + [mark] + mem
        correct_r = 1 if output[1] == output[2] else 0
        correct_p = output[3] == 'right'
        correct_pr = int(correct_p and correct_r)
        valid = output[3] == 'right' or output[3] == 'wrong'
        prediction_score = int(correct_p == correct_r and valid)
        self.tmp_r.append(correct_pr)
        self.tmp_p.append(prediction_score)
        
        # Create training data
        if correct_pr:
            self.correct_buffer.append(output)
            self.results.append(correct_pr)
        elif correct_p != correct_r:
            output[3] = f"Predicted: {correct_p}, Result: {correct_r}"
            if not self.test: self.train_buffer.append(output)
            self.results.append(correct_pr)
        else:
            if self.iter == self.ac - 1: self.results.append(correct_pr)
            self.tmp_buffer.append(output)

    
    def save_result_to_file(self):
        
        r, p = self.tmp_r, self.tmp_p
        print("Adaptive Compute Iteration: ", self.iter)
        print("Result: %d/%d = %.2f%% correct" % (np.sum(r), len(r), 100*np.mean(r)))
        print("Predictions: %d/%d = %.2f%% correct" % (np.sum(p), len(p), 100*np.mean(p)))
        if not self.test: self.write_file(self.train_fn, self.train_buffer)
        self.write_file(self.tmp_fn, self.tmp_buffer)
        self.write_file(self.correct_fn, self.correct_buffer)
        
    def write_file(self, fname, buffer):
        
        with open(fname, "a") as file:
            indexes = self.sort_data_len(buffer)
            for index in indexes:
                for idx, item in enumerate(buffer[index]):
                    if idx == 0: file.write(item[:self.MD.max_src] + '\n')
                    else: file.write(item[:self.MD.max_trg] + '\n')
        
        
    def sort_data_len(self, data):
        indexes = list(range(len(data)))
        sorted_data = []
        for index in indexes:
            tot_len = sum([len(x) for x in data[index]])
            tot_len -= len(data[index][1]) # Subtract target
            if self.mem_slots:
                tot_len -= len(data[index][2]) # Subtract previous prediction 
                tot_len -= len(data[index][3]) # Subtract status line
            sorted_data.append([index, tot_len])
        
        sorted_data = sorted(sorted_data, key=lambda x: x[1])
        return [i[0] for i in sorted_data]
    
    def create_filenames(self):
        head_tail = os.path.split(self.fname)
        self.train_fn = head_tail[0] + '/' + head_tail[1]
        self.correct_fn = head_tail[0] + '/correct_' + head_tail[1]
        self.tmp_fn = head_tail[0] + '/tmp_' + head_tail[1]
    
    def initiate_vars(self):
        self.tmp_buffer, self.train_buffer, self.correct_buffer = [], [], []
        self.len, self.predict, self.batches, self.leftover = 0, 0, 0, []
        self.tmp_r, self.tmp_p = [], []
        
    def initiate_at_start(self):
        self.results = []
        if os.path.exists(self.train_fn) and not self.test:
            os.remove(self.train_fn)