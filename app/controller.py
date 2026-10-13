

import os
from enum import Enum
from typing import Optional
from utilities.data_utils import counter

from PySide6.QtCore import QObject, QTimer

import subprocess

class Stage(Enum):
    OMNI = "OMNI"
    MODEL = "MODEL"

class Controller:
    def __init__(self, omni_folder: str, model_folder: str , initial_exp_count: int, omni_filename: str="expttsd", model_filename: str="model"):
        
        self.omni_filename       = omni_filename
        self.model_filename      = model_filename
        self.omni_folder         = omni_folder
        self.model_folder        = model_folder
        self.initial_exp_count   = initial_exp_count

        self.current_omni_folder = omni_folder
        
        self.model_count  = 0
        self.omni_count   = 0
        self.process_stop = False
        
        self.current_stage: Optional[Stage] = None
    
    def exp_num(self,n):
        self.initial_exp_count = n

    def start(self):
        self.current_stage = Stage.OMNI
        self.process_stop = False

    def reset(self):
        self.process_stop = False
        self.current_stage = None
        self.initial_exp_count = 0
        self.model_count = 0
        self.omni_count = 0

    def stop(self):
        self.current_stage = None
        self.process_stop = True  

    def if_omni_finished(self,path,num): 
        return counter(path, self.omni_filename) >= num
        
    def if_model_finished(self,path): 
        return counter(path, self.model_filename) >= 1
    
    def pathcreator_model(self, counter_val, path_orig):

        path_I = os.path.join(path_orig, f"_model_{counter_val}")
        try:
            os.makedirs(path_I, exist_ok = True)
        except FileExistsError:
            pass
        
        PATH = path_I
        
        return PATH
        
    def pathcreator_omni(self, counter_val, path_orig):

        path_O = os.path.join(path_orig, f"_OMNI_{counter_val}")
        try:
            os.makedirs(path_O, exist_ok = True)
        except FileExistsError:
            pass
        PATH = path_O
        exp_count = 1
        return PATH, exp_count

