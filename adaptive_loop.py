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


# In[ ]:


# make deterministic
from mingpt.utils import set_seed
set_seed(42)


# In[ ]:


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
from mingpt.model import GPT, GPTConfig, GPT1Config
from torch.utils.data.dataloader import DataLoader
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.adaptive_examiner import AdaptiveExaminer
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


#create a dataset
#get_ipython().system('rm -rf run models')
#get_ipython().system('cp -r data run')
#!mkdir models
fn_data = 'run/numbers__list_prime_factors.txt'


# In[ ]:


# Add memory data structure to training data
memory_slots = 7
MD = MemData(memory_slots, debug=0)
MD.initiate_mem_slot_data(fn_data)


# In[ ]:


fn_test = 'run/test_numbers__list_prime_factors.txt'
fn_train = 'run/train_numbers__list_prime_factors.txt'
train_dataset = MathDataset(fname=fn_train, MD=MD, marker_data=0.2)


# In[ ]:


#MD.tensor2string(train_dataset[4][0])


# In[ ]:


print(MD.block_size)
print(MD.vocab_size)
print(MD.max_trg)


# In[ ]:


# initialize a baby GPT model
mconf = GPTConfig(MD.vocab_size, MD.block_size,
                 n_layer=4, n_head=8, n_embd=256)
model = GPT(mconf)
#model = torch.load('12.pth')


# In[ ]:


max_it = 100
current_it = 0
batch_size = 384
#marker_data = 0.2

exp_folder = 'models/' + datetime.datetime.now().strftime('%Y-%m-%d~%H:%M:%S')

while(current_it < max_it):
    
    # Wait until the working memory is filled, then use 5 epochs
    epoch = 1 if current_it < 7 else 20
    # Use marker data once the working memory is full
    marker_data = 0.0 if current_it < 2 else 0.2
    ac = 1 if current_it < 7 else 10
    
    examiner = AdaptiveExaminer(MD, ac=ac, max_batch=batch_size)
    
    # Switch between main training and marker training
    print("Marker Data: ", str(marker_data))
    train_dataset = MathDataset(fname=fn_train, MD=MD, marker_data=marker_data)
    test_dataset = MathDataset(fname=fn_test, MD=MD, marker_data=0.0)
    
    # Trainer Config
    tconf = TrainerConfig(max_epochs=epoch, batch_size=batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=1024, final_tokens=epoch*len(train_dataset)*(MD.vocab_size+1),
                      num_workers=6)
    
    # Create the first training round
    print("Training: ", str(current_it))
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()
    trainer.save_checkpoint(exp_folder, str(current_it))
    
    # Examine the model and create new dataset
    
    print("Exam and new dataset-------------\n")
    print("Training exam \n")
    examiner.exam(fn_train, trainer)
    print("Test exam \n")
    examiner.exam(fn_test, trainer, test=True)
    
    current_it += 1


# In[ ]:





# In[ ]:




