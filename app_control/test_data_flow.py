#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pytest
from data_flow import Controller

def test_stage_start():
    c = Controller(r"C:\Users\user\Desktop\OMNI\1. OMNI",r"C:\Users\user\Desktop\OMNI\2. IMED", 3)
    c.start()
    assert c.current_stage.name == "OMNI"
    
def test_expnum():
    c = Controller(r"C:\Users\user\Desktop\OMNI\1. OMNI",r"C:\Users\user\Desktop\OMNI\2. IMED", 3)
    assert c.initial_exp_count == 3


# In[ ]:




