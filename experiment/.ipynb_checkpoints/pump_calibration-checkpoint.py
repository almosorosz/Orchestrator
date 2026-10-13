#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
from scipy.optimize      import minimize


# In[128]:


class PumpRateConverter:
    def __init__(self,folder_loc,rpm_col_name,mass_col_name):
        self.descr = 'Convert g/min to rpm'
        self.folder_loc = folder_loc
        self.fun = []
        self.res = []
        self.RPM_fit = []
        self.mass_rate_fit = []
        self.rpm_col_name = rpm_col_name
        self.mass_col_name = mass_col_name
        self.create_conversion()
    def selector(self,A,B):
        return np.array([A[i] for i in B])

    def create_conversion(self):
        Data = pd.read_csv(self.folder_loc,header=1).drop([0])
        Data = Data[0:-1:20] # take only every 20th minutes

        mask  = Data[self.rpm_col_name].astype(float) != 0
        mask2 = Data[self.rpm_col_name].astype(float) <100

        time = Data['Time'][mask & mask2].astype(float).values # min
        time = time / 60.0 # to [h]
        RPM = Data[self.rpm_col_name][mask & mask2].astype(float).values
        MASS = Data[self.mass_col_name][mask & mask2].astype(float).values

        start = [0]
        stop = []
        Val = RPM[0]
        for j,i in enumerate(RPM):
            if i != Val:
                stop.append(j-1)
                start.append(j+1)
                Val = i

        stop.append(-1)        

        RPM_start  = self.selector(RPM,start)
        MASS_start = self.selector(MASS,start)
        time_start = self.selector(time,start)

        RPM_stop   = self.selector(RPM,stop)
        MASS_stop  = self.selector(MASS,stop)
        time_stop  = self.selector(time,stop)

        mass_rate = abs(MASS_start - MASS_stop) / abs(time_start - time_stop +1e-30)

        self.RPM_fit = np.append(RPM_start,np.zeros(100))
        self.mass_rate_fit = np.append(mass_rate,np.zeros(100))

        fun = lambda x,X,Y: (( X - ((x[0] * Y**2) + (x[1] * Y) ))**2).sum(axis=0)
        pred = lambda x,Y: ((x[0] * Y**2) + (x[1] * Y) )
        self.fun = fun
        self.pred = pred
        res = minimize(fun,(1,1),args=(self.RPM_fit,self.mass_rate_fit), method='Nelder-Mead')
        self.res = res

    def convert_to_rpm(self,VAL):
        if self.fun:
            pred_val = np.array(self.pred(self.res.x,VAL))
            return np.array([np.round(i,2) for i in pred_val])
        else:
            print('Please call "self.create_conversion" first !')


# In[ ]:




