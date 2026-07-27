#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from enum import Enum
from typing import Optional
from pure_functions import counter
import subprocess

class Stage(Enum):
    OMNI = "OMNI"
    IMED = "IMED"

class Controller:
    def __init__(self, omni_folder: str, imed_folder: str , initial_exp_count: int, omni_filename: str="expttsd", imed_filename: str="imed"):
        
        self.omni_filename = omni_filename
        self.imed_filename = imed_filename
        self.omni_folder = omni_folder
        self.imed_folder = imed_folder
        self.initial_exp_count = initial_exp_count

        self.current_omni_folder = omni_folder
        
        self.current_stage: Optional[Stage] = None
        self.imed_count = 0
        self.omni_count = 0
        self.process_stop = False
    
    def exp_num(self,n):
        self.initial_exp_count = n

    def start(self):
        self.current_stage = Stage.OMNI
        self.process_stop = False

        # START OMNIBUS for the 1st time - self.omni_folder
        # a = subprocess.run(["cmd", "/c", "dir"], capture_output=True, text=True)
        
    def reset(self):
        self.process_stop = False
        self.current_stage = None
        self.initial_exp_count = 0
        self.imed_count = 0
        self.omni_count = 0

    def stop(self):
        self.current_stage = None
        self.process_stop = True  

    def if_omni_finished(self,path,num): 
        return counter(path, self.omni_filename) >= num

    def if_imed_finished(self,path): 
        return counter(path, self.imed_filename) >= 1
    
    def pathcreator_imed(self, counter_val, path_orig):
        if counter_val == 0:
            PATH = path_orig
        else:
            path_I = os.path.join(path_orig, f"_IMED_{counter_val}")
            try:
                os.makedirs(path_I, exist_ok = True)
            except FileExistsError:
                pass
            
            PATH = path_I
        
        return PATH
        
    def pathcreator_omni(self, counter_val, path_orig):
        if counter_val == 0:
            PATH = path_orig
            exp_count = self.initial_exp_count
        else:
            path_O = os.path.join(path_orig, f"_OMNI_{counter_val}")
            try:
                os.makedirs(path_O, exist_ok = True)
            except FileExistsError:
                pass
            PATH = path_O
            exp_count = 1
        return PATH, exp_count


# In[ ]:




