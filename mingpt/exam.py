from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from mingpt.utils import sample
import numpy as np
import os

class Exam:
    """ Clean data, tokenizer, helper functions """

    def __init__(self, dataset, MD, trainer, fname):
        # MemData, Trainer, and Dataset classes
        self.mem_slots = MD.mem_slots
        self.max_trg = MD.max_trg
        self.trainer = trainer
        self.dataset = dataset
        self.fname = fname
        self.MD = MD
        
        # Batch settings
        self.batch_size = 1
        self.max_batch_size = 512
        self.nbr_predictions = -1
        
        # Variables for prediction loop
        self.prev_src_len = 0
        self.predict = 0
        self.batch = 0
        self.x_in = 0
        
        # Store results
        self.results = []
        self.correct_buffer = []
        self.train_buffer = []
    
    def run(self, fname):
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        pbar = tqdm(enumerate(loader), total=len(loader))
    
        for batch, (x, y) in pbar:
            
            x, x_in = self.concatenate_batches(x, x_in)
        
            if self.predict or self.batch == self.max_batch_size:
                pred, clip_src = self.make_prediction(x_in)
                for i in range(x_in.size(0)):
                    src_trg_pred_mem = self.extract_src_trg_pred_mem(x_in[i], pred[i], clip_src)
                    self.log_results(src_trg_pred_mem)
        
            if self.nbr_predictions >= 0 and batch+1 >= self.nbr_predictions:
                break
        
        results = self.results
        print("Final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))
        
        print("Saving new files to disk...")
        self.save_result_to_file(self)
                    

    def concatenate_batches(self, x, x_in):
        
        src_len = self.MD.locateToken('answer', x[0]) 
        x_in = leftover if self.prev_src_len == -1 else x_in

        # Concat input source with same length
        if self.prev_src_len == src_len:
            x_in = torch.cat((x_in, x), 0)
        elif self.prev_src_len == 0:
            x_in = x
        else:
            self.prev_src_len = -1
            self.predict = 1
            leftover = x
        self.prev_src_len = src_len
        self.batch += 1
        
        return x, x_in
        
    def make_prediction(self, x_in):
        
        clip_src_mem = self.MD.locateToken('answer', x_in[0]) + 1 
        if self.mem_slots:
            clip_src_mem = self.MD.locateToken('mem-end', x_in[0]) + 1

        self.batch, self.predict, self.prev_src_len, = 0, 0, 0
        x_cut = x_in[:, :clip_src_mem]

        pred = x_cut.to(self.trainer.device)
        pred = sample(model, pred, int(self.max_trg+1))
        
        return out, clip_src_mem
    
    def extract_src_trg_pred_mem(self, x, pred, cut_src_mem):
        
        # Extract src from data
        cut_src = self.MD.locateToken('answer', x)
        src =  x[:cut_src]
        
        # Extract trg from data
        cut_padding = self.MD.locateToken('pad', x)
        trg = x[cut_src_mem:cut_padding] # X does not have the 'finish' token
        
        # Extract prediction from data
        cut_pred = self.MD.locateToken('finish', pred)
        pred = pred[cut_src_mem:cut_pred]
        
        # Extract memory from data
        mem = []
        if self.mem_slots:
            mem = self.tensor_to_mem_slots(x[cut_src:])
        
        # Translate the tensors to strings
        src = self.MD.tensor2string(src)
        trg = self.MD.tensor2string(trg)
        pred = self.MD.tensor2string(pred) 
        
        return src + trg + pred + mem
    
    def tensor_to_mem_slots(self, x):
        
        mem = []
        for i in range(self.mem_slots):
            cut = self.MD.locateToken('mem', x)
            mem_string = self.MD.tensor2string(x[:cut])
            mem.append(mem_string)
            x = x[cut:]
        
        return mem
            
    def log_results(self, src_trg_pred_mem):
        
        correct = 1 if src_trg_pred_mem[1] == src_trg_pred_mem[2] else 0
        self.results.append(correct)
        if correct:
            self.correct_buffer.append(src_trg_pred_mem)
        else:
            self.train_buffer.append(src_trg_pred_mem)

    def save_results_to_file(self):
        
        fname_head_tail = os.path.split(self.fname)
        correct_fname = head_tail[0] + '/correct_buffer_' + head_tail[1]
        train_fname = head_tail[0] + '/train_buffer_' + head_tail[1]
        
        if os.path.exists(correct_fname):
            os.remove(correct_fname)
        if os.path.exists(train_fname):
            os.remove(train_fname)
        
        with open(correct_fname, "a") as correct_fname:
            indexes = self.sort_data_by_srcmem_len(self.correct_buffer)
            for index in range(indexes):
                for item in self.correct_buffer[index]:
                    file.write(item + '\n')
        
        with open(train_fname, "a") as train_fname:
            indexes = self.sort_data_by_srcmem_len(self.train_buffer)
            for index in range(len(self.train_buffer)):
                for item in self.train_buffer[index]:
                    file.write(item + '\n')
        
    def sort_data_by_srcmem_len(self, data):
        indexes = list(range(len(data)))
        sorted_data = []
        for index in indexes:
            tot_len = sum([len(x) for x in data[index]])
            tot_len -= len(data[index][1]) # Subtract target
            tot_len -= len(data[index][2]) # Subtract prediction
            sorted_data.append([index, tot_len])
        
        sorted_data = sorted(sorted_data, key=lambda x: x[1])
        return [i[0] for i in sorted_data]