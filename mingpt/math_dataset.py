import random
import numpy as np
import torch
import string
import os
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import Dataset

class MathDataset(Dataset):
    """
    Lorum ipsum ;)
    
    """

    def __init__(self, fname, MD, marker_data=0.2, remove_memory=0.1, size=-1):
        self.MD = MD
        self.mem_slots = MD.mem_slots
        self.marker_data = marker_data
        self.remove_memory = remove_memory
        self.data_lines = 4
        self.dataset = self.MD.prepare_data(fname) # Extract data, source, memory, and target        
        self.ixes = np.array(list(range(len(self.dataset[0])))) 
        self.ixes = self.ixes[:size]
        
    def __len__(self):
        return self.ixes.size

    def __getitem__(self, idx):
        
        xy = []
        for i in range(self.mem_slots + self.data_lines):
            xy.append(self.dataset[i][idx])
            
        use_marker = random.random() < self.marker_data
        src, mem, trg = self.MD.create_marker_data(xy) if use_marker else self.MD.create_math_data(xy, self.remove_memory)
        src_mem_trg = self.MD.list2tokens(src + mem + trg)
        x = self.MD.x2Canvas(src_mem_trg)
        y = self.MD.y2Canvas(src_mem_trg)
        y = self.MD.mask_padding(y)

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long) 
        # we will only train in the output locations. -100 will mask loss to zero
        y = self.MD.mask_question_memory(y, len(src + mem + trg) - 2) if use_marker \
        else self.MD.mask_question_memory(y, len(src + mem)-1)
        
        return x, y