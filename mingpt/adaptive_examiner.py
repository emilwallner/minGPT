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
        
        self.create_filenames()
        self.test = test
        self.debug = debug
        self.fname = fname
        self.trainer = trainer
        self.initiate_at_start()
        
        for i in range(self.ac):
            self.one_loop(i)
            
        self.write_file(self, self.train_fn, self.tmp_buffer)
        
        if os.path.exists(self.tmp_fn):
            os.remove(self.tmp_fn)
        
        results = self.results
        preds = self.predictions
        print("Final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))
        print("Final predictions: %d/%d = %.2f%% correct" % (np.sum(preds), len(preds), 100*np.mean(preds)))
        
    def one_loop(self, current_it):
        
        self.initiate_vars()
        fname == self.train_fn if current_it == 0 else self.tmp_fn
        dataset = MathDataset(fname=fname, MD=self.MD, marker_data=0.0)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        if current_it == 0: self.initiate_at_start()
        
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
                    self.log_results(clean_output)
                
                # Log status
                pbar.set_description(f"Iiter {b} Score: {np.sum(self.results)}/{len(self.results)}")
            
            if self.debug and b+1 >= self.debug:
                break
        
        self.save_result_to_file()
                    

    def concatenate_batches(self, batch, x):
        
        new_len = self.get_input_len(x[0]) 
        batch = self.leftover if self.buffer == True else batch

        # Concat input source with same length
        if self.len == new_len: batch = torch.cat((batch, x), 0)
        elif self.buffer == False: batch = x
        else: self.predict, self.leftover = 1, x
   
        self.len = new_len
        self.batch_size += 1
        if self.batch_size == self.max_batch_size: self.predict = 1
        
        return batch
        
    def model_predict(self, x_in):
        
        # Reset tracker variables
        self.batch, self.predict, self.len, self.buffer = 0, 0, 0, False
        
        # Prepare input
        cut_index = self.get_input_len(x_in[0]) + 1 
        x_cut = x_in[:, :cut_index]
        pred = x_cut.to(self.trainer.device)
        
        # Make prediction
        output = sample(self.trainer.model, pred, int(self.max_trg+1))
        
        return output
    
    def get_input_len(self, x):
        
        if self.mem_slots: clip_src_mem = self.MD.locate_token('mem-end', x)
        else: clip_src_mem = self.MD.locate_token('answer', x) 
        
        return clip_src_mem
    
    def clean_output(self, x, pred, cut_input):
        
        # Extract src from data
        cut_src = self.MD.locate_token('answer', x)
        src =  x[:cut_src]
        
        # Extract trg from data
        cut_padding = self.MD.locate_token('pad', x)
        trg = x[cut_input:cut_padding] # X does not have the 'finish' token
        
        # Extract prediction from data
        cut_pred = self.MD.locate_token('finish', pred)
        mark = self.MD.idx[pred[cut_pred+1].tolist()]
        pred = pred[cut_input:cut_pred]
        
        # Translate the tensors to strings
        src = self.MD.tensor2string(src)
        trg = self.MD.tensor2string(trg)
        pred = self.MD.tensor2string(pred) 
        
        # Extract memory from data
        mem = []
        if self.mem_slots:
            mem = self.tensor2memory(x[cut_src+1:], pred)
        
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

            
    def log_results(self, output):
        
        # Output = [src] + [trg] + [pred] + [mark] + mem
        correct_p = output[3] == 'right'
        correct_r = 1 if output[1] == output[2] else 0
        correct_pr = int(correct_p and correct_r)
        prediction_score = int(correct_p == correct_r)
        # TODO: Seperate local and global results
        self.results.append(correct_pr)
        self.predictions.append(prediction_score)
        
        # Create training data
        if correct_pr:
            self.correct_buffer.append(output)
        elif (correct_p and not correct_r) or (not correct_p and correct_r):
            output[3] = f"Predicted: {correct_p}, Result: {correct_r}"
            self.train_buffer.append(output)
        else:
            self.temp_buffer.append(output)

    def save_result_to_file(self):
        
        self.write_file(self, self.train_fn, self.train_buffer)
        self.write_file(self, self.tmp_fn, self.tmp_buffer)
        self.write_file(self, self.correct_fn, self.correct_buffer)
        
    def write_file(self, fname, buffer):
        
        with open(fname, "a") as file:
            indexes = self.sort_data_len(buffer)
            for index in indexes:
                for item in buffer[index]:
                    file.write(item + '\n')
        
        
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
        self.temp_fn = head_tail[0] + '/temp_' + head_tail[1]
    
    def initiate_vars(self):
        self.tmp_buffer, self.train_buffer, self.correct_buffer = [], [], []
        self.len, self.predict, self.batch, self.buffer = 0, 0, 0, False
        
    def initiate_at_start(self):
        self.results, self.predictions = [], []
        if os.path.exists(train_fname):
            os.remove(train_fname)