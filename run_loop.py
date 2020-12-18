#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from mingpt.marker_dataset import MarkerDataset
from mingpt.math_dataset import MathDataset
from mingpt.model import GPT, GPTConfig, GPT1Config
from torch.utils.data.dataloader import DataLoader
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.examiner import Examiner
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[4]:


#create a dataset
# get_ipython().system('rm -rf run models')
# get_ipython().system('cp -r data run')
# get_ipython().system('mkdir models')
# fn_data = 'run/numbers__list_prime_factors.txt'


# In[5]:


# Add memory data structure to training data
memory_slots = 7
MD = MemData(memory_slots)
#MD.initiate_mem_slot_data(fn_data)


# In[6]:


fn_test = 'run/test_numbers__list_prime_factors.txt'
fn_train = 'run/train_numbers__list_prime_factors.txt'
train_dataset = MathDataset(fname=fn_train, MD=MD)


# In[7]:


print(MD.block_size)
print(MD.vocab_size)
print(MD.max_trg)


# In[8]:


# initialize a baby GPT model
#mconf = GPTConfig(MD.vocab_size, MD.block_size,
#                  n_layer=4, n_head=8, n_embd=256)
#model = GPT(mconf)
model = torch.load('12.pth')


# In[ ]:


max_it = 100
main_epoch = 5
marker_epoch = 1
current_it = 0

exp_folder = 'models/' + datetime.datetime.now().strftime('%Y-%m-%d~%H:%M:%S')
examiner = Examiner(MD)

while(current_it < max_it):

    # Switch between main training and marker training
    if current_it % 2 == 0:
        print("Loading Main Dataset\n")
        train_dataset = MathDataset(fname=fn_train, MD=MD)
        test_dataset = MathDataset(fname=fn_test, MD=MD)
        epoch = main_epoch
    else:
        print("Loading Marker Dataset\n")
        train_dataset = MarkerDataset(fname=fn_train, MD=MD)
        test_dataset = MarkerDataset(fname=fn_test, MD=MD)
        epoch = marker_epoch
    
    # Trainer Config
    tconf = TrainerConfig(max_epochs=epoch, batch_size=358, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=1024, final_tokens=epoch*len(train_dataset)*(MD.vocab_size+1),
                      num_workers=6)
    
    # Create the first training round
    print("Training: ", str(current_it))
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()
    trainer.save_checkpoint(exp_folder, str(current_it))
    
    # Examine the model and create new dataset
    if current_it % 2 == 0:
        print("Exam and new dataset-------------\n")
        print("Training exam \n")
        examiner.exam(fn_train, train_dataset, trainer)
        print("Test exam \n")
        examiner.exam(fn_test, test_dataset, trainer)
    
    current_it += 1

