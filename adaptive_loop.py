#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)


# In[2]:


# make deterministic
from mingpt.utils import set_seed
set_seed(42)


# In[3]:


import numpy as np
import torch
import string
import os
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
import datetime
from mingpt.md import MemData
from mingpt.math_dataset import MathDataset
from mingpt.ac_schedule import AcSchedule
from mingpt.model import GPT, GPTConfig, GPT1Config
from torch.utils.data.dataloader import DataLoader
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.adaptive_examiner import AdaptiveExaminer
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[4]:


#create a dataset
#get_ipython().system('rm -rf run models')
#get_ipython().system('cp -r data run')
#!mkdir models
fn_data = 'run/numbers__list_prime_factors.txt'


# In[5]:


# Add memory data structure to training data
memory_slots = 7
MD = MemData(memory_slots, debug=0)
MD.initiate_mem_slot_data(fn_data, warmup_sz=50000)


# In[6]:


fn_test = 'run/test_numbers__list_prime_factors.txt'
fn_train = 'run/train_numbers__list_prime_factors.txt'
train_dataset = MathDataset(fname=fn_train, MD=MD, marker_data=0.2)


# In[7]:


len(train_dataset)


# In[8]:


#MD.tensor2string(train_dataset[4][0])


# In[9]:


print(MD.block_size)
print(MD.vocab_size)
print(MD.max_trg)


# In[10]:


# initialize a baby GPT model
mconf = GPTConfig(MD.vocab_size, MD.block_size,
                 n_layer=4, n_head=8, n_embd=256)
model = GPT(mconf)
#model = torch.load('12.pth')


# In[ ]:


max_it = 100
current_it = 0
batch_size = 384

exp_folder = 'models/' + datetime.datetime.now().strftime('%Y-%m-%d~%H:%M:%S')
schedule = AcSchedule(MD)

while(current_it < max_it):
    
    epoch, size, ac, marker_data, warmup = schedule.create(current_it)
    
    # Switch between main training and marker training
    print(f"Training Iteration: {current_it}\n\nEpochs: {epoch}\nSize: {size}\nMarker: {marker_data}\nWarm: {warmup}\nAC: {ac}\n")
    
    train_dataset = MathDataset(fname=fn_train, MD=MD, marker_data=marker_data)
    test_dataset = None
    #test_dataset = MathDataset(fname=fn_test, MD=MD, marker_data=0.0) if not warmup else None
    
    # Trainer Config
    tconf = TrainerConfig(max_epochs=epoch, batch_size=batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=1024, final_tokens=epoch*len(train_dataset)*(MD.block_size/3),
                      num_workers=6)
    
    # Create the first training round
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()
    trainer.save_checkpoint(exp_folder, str(current_it))
    
    # Examine the model and create new dataset
    examiner = AdaptiveExaminer(MD, ac=ac, max_batch=batch_size, warmup=warmup)
    
    print("Training exam \n")
    examiner.exam(fn_train, trainer, size, test=False)
    
    if not warmup:
        print("Test exam \n")
        examiner.exam(fn_test, trainer, size, test=True)
    
    current_it += 1


# In[ ]:





# In[ ]:




