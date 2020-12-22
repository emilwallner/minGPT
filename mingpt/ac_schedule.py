import random
import numpy as np
import torch
import string
import os
import torch.nn as nn
from torch.nn import functional as F
import math

class AcSchedule:
    """ Clean data, tokenizer, helper functions """

    def __init__(self, MD, marker_data=0.2):
        # MemData, Trainer, and Dataset classes
        
        self.init_sz = 50000
        self.scale_sz = 1.5
        self.mem_slots = MD.mem_slots
        self.len = MD.dataset_len
        self.marker_data = marker_data
        
    def create(self, current_it):
        
        current_it += 1
        
        if current_it <= self.mem_slots:
            epoch = 5
            size = self.init_sz
            ac = 1
            warmup = True
        else:
            size = self.init_sz * (self.scale_sz**(current_it - self.mem_slots))
            epoch = 10
            #if size < self.len:epoch = 5
            #else: epoch = max(math.ceil(size/self.len), 5)
            size = min(size, self.len)
            ac = math.ceil((current_it - self.mem_slots) / 2) + 3
            warmup = False
        
        marker_data = 0.0 if current_it == 1 else self.marker_data
        
        return epoch, int(size), ac, marker_data, warmup
                
        
                