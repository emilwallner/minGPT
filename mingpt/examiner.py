from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
from mingpt.utils import sample
import datetime
import numpy as np
import os

#remove
import random, string

class Examiner:
    """ Clean data, tokenizer, helper functions """

    def __init__(self, MD):
        # MemData, Trainer, and Dataset classes
        self.mem_slots = MD.mem_slots
        self.max_trg = MD.max_trg
        self.MD = MD
        
        # Batch settings
        self.batch_size = 1
        self.max_batch_size = 256
        
    
    def exam(self, fname, dataset, trainer, testing=-1):
        
        self.nbr_predictions = testing
        self.fname = fname
        self.trainer = trainer
        
        # Variables for prediction loop
        self.prev_src_len = 0
        self.predict = 0
        self.batch = 0
        
        # Store results
        self.results = []
        self.correct_buffer = []
        self.train_buffer = []
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        pbar = tqdm(enumerate(loader), total=len(loader))
        x_in = []
        
        for batch, (x, y) in pbar:
            
            x, x_in = self.concatenate_batches(x, x_in)
        
            if self.predict or self.batch == self.max_batch_size or batch+1 == self.nbr_predictions:
                pred, clip_src = self.make_prediction(x_in)
                for i in range(x_in.size(0)):
                    src_trg_pred_mem = self.extract_src_trg_pred_mem(x_in[i], pred[i], clip_src)
                    self.log_results(src_trg_pred_mem)
            
            # report progress
            pbar.set_description(f"Iiter {batch}")
            if self.nbr_predictions >= 0 and batch+1 >= self.nbr_predictions:
                break
        
        results = self.results
        print("Final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))
        print("Saving new files to disk...")
        self.save_result_to_file()
                    

    def concatenate_batches(self, x, x_in):
        
        clip_src_mem = self.get_clip_src_mem_len(x[0]) 
        x_in = leftover if self.prev_src_len == -1 else x_in

        # Concat input source with same length
        if self.prev_src_len == clip_src_mem:
            try:
                x_in = torch.cat((x_in, x), 0)
            except:
                print(x.size())
                print(x_in.size())
                print(x)
                print(x_in)
                print(self.prev_src_len)
                print(clip_src_mem)
        elif self.prev_src_len == 0:
            x_in = x
        else:
            self.prev_src_len = -1
            self.predict = 1
            leftover = x
        self.prev_src_len = clip_src_mem
        self.batch += 1
        
        return x, x_in
        
    def make_prediction(self, x_in):
        
        # Reset tracker variables
        self.batch, self.predict, self.prev_src_len, = 0, 0, 0
        clip_src_mem = self.get_clip_src_mem_len(x_in[0]) + 1 
        x_cut = x_in[:, :clip_src_mem]

        pred = x_cut.to(self.trainer.device)
        pred = sample(self.trainer.model, pred, int(self.max_trg+1))
        
        return pred, clip_src_mem
    
    def get_clip_src_mem_len(self, x):
        
        if self.mem_slots:
            clip_src_mem = self.MD.locate_token('mem-end', x)
        else:
            clip_src_mem = self.MD.locate_token('answer', x) 
        
        return clip_src_mem
    
    def extract_src_trg_pred_mem(self, x, pred, cut_src_mem):
        
        # Extract src from data
        cut_src = self.MD.locate_token('answer', x)
        src =  x[:cut_src]
        
        # Extract trg from data
        cut_padding = self.MD.locate_token('pad', x)
        trg = x[cut_src_mem:cut_padding] # X does not have the 'finish' token
        
        # Extract prediction from data
        pred = pred[pred > 3] # Filter out padding tokens etc
        cut_pred = self.MD.locate_token('finish', pred)
        pred = pred[cut_src_mem:cut_pred]
        
        # Translate the tensors to strings
        src = self.MD.tensor2string(src)
        trg = self.MD.tensor2string(trg)
        pred = self.MD.tensor2string(pred) 
        
        # Extract memory from data
        mem = []
        if self.mem_slots:
            mem = self.tensor_to_mem_slots(x[cut_src+1:], pred)
        
        return [src] + [trg] + [pred] + mem
    
    def tensor_to_mem_slots(self, x, pred):
        
        mem = []
     
        for i in range(self.mem_slots):
            cut = self.MD.locate_token('mem', x)
            mem_string = self.MD.tensor2string(x[:cut])
            mem_string = mem_string[:self.max_trg+1] # Cut to max target
            mem.append(mem_string)
            x = x[cut+1:] # Remove mem token

        mem = mem if pred in mem else [pred] + mem[:-1]
        return mem

            
    def log_results(self, src_trg_pred_mem):
        
        correct = 1 if src_trg_pred_mem[1] == src_trg_pred_mem[2] else 0
        self.results.append(correct)
        if correct:
            self.correct_buffer.append(src_trg_pred_mem)
        else:
            self.train_buffer.append(src_trg_pred_mem)

    def save_result_to_file(self):
        
        head_tail = os.path.split(self.fname)
        correct_fname = head_tail[0] + '/correct_' + head_tail[1] + datetime.datetime.now().strftime('%Y-%m-%d~%H:%M:%S')
        train_fname = head_tail[0] + '/' + head_tail[1]
        
        if os.path.exists(correct_fname):
            os.remove(correct_fname)
        if os.path.exists(train_fname):
            os.remove(train_fname)
        
        with open(correct_fname, "a") as file:
            indexes = self.sort_data_by_srcmem_len(self.correct_buffer)
            for index in indexes:
                for item in self.correct_buffer[index]:
                    file.write(item + '\n')
        
        with open(train_fname, "a") as file:
            indexes = self.sort_data_by_srcmem_len(self.train_buffer)
            for index in indexes:
                for item in self.train_buffer[index]:
                    file.write(item + '\n')
        
    def sort_data_by_srcmem_len(self, data):
        indexes = list(range(len(data)))
        sorted_data = []
        for index in indexes:
            tot_len = sum([len(x) for x in data[index]])
            tot_len -= len(data[index][1]) # Subtract target
            if self.mem_slots:
                tot_len -= len(data[index][2])
            sorted_data.append([index, tot_len])
        
        sorted_data = sorted(sorted_data, key=lambda x: x[1])
        return [i[0] for i in sorted_data]