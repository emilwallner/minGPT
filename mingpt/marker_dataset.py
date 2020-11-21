import random
import numpy as np
import torch
import string
import os
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import Dataset

class MarkerDataset(Dataset):
    """
    Dataset that only trains right or wrong.
    
    """

    def __init__(self, fname, MD):
        self.MD = MD
        self.mem_slots = MD.mem_slots
        self.dataset = self.MD.prepare_data(fname) # Extract data, source, memory, and target        
        self.ixes = np.array(list(range(len(self.dataset[0])))) 
        
    def __len__(self):
        return self.ixes.size

    def __getitem__(self, idx):
        
        xy = []
        for i in range(self.mem_slots + 3):
            xy.append(self.dataset[i][idx])
            
        src, mem, trg = self.MD.create_marker_data(xy)
        src_mem_trg = self.MD.list2tokens(src + mem + trg)
        x = self.MD.x2Canvas(src_mem_trg)
        y = self.MD.y2Canvas(src_mem_trg)
        y = self.MD.mask_padding(y)

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long) 
        y = self.MD.mask_question_memory(y, len(src + mem + trg) - 2) # we will only train in the output locations. -100 will mask loss to zero
        
        return x, y