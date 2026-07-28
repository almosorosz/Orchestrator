#!/usr/bin/env python
# coding: utf-8

# In[5]:


def counter( folder, filename):
    import os 
    
    count = 0
    with os.scandir(folder) as entries:
        for entry in entries:           
            try:
                A = int(entry.name[-5])
                T = A %2 == 0
            except:
                T = False
            if filename in entry.name and T:
                count += 1
    return count

def file_finder(folder, filename):
    import os 
    files = []
        
    with os.scandir(folder) as entries:
        for entry in entries:        
            try:
                A = int(entry.name[-5])
                T = A%2 == 0
            except:
                T = False
            if filename in entry.name and T:
                files.append(entry.path)
    return files

def corr_func(x,X):
    A = x[0] * X[0] + x[1]
    C = (A.iloc[0].values - X[0].iloc[0].values)**2
    C = (A.iloc[0].values - 0.07332960848175958)**2
    D = (A.iloc[-1].values - X[1].iloc[-1].values)**2

    return  C+D

def W_pred(X_new):
    mu_X = [151.74736042]
    sigma_X = [34.94415798]
    B = [[1.89453211, 3.1580302,  4.58156681]]
    A = [ 9.93866667, 20.08833333, 37.43866667]
    X_new_scaled = (X_new - mu_X) / sigma_X
    Y_scaled_pred_direct = X_new_scaled @ B  
    Y_pred_direct = Y_scaled_pred_direct + A 
    
    return Y_pred_direct



def L_pred(X_new):
    mu_X = [201.78218686, 736.3733133 ]
    sigma_X = [ 91.934371,   239.33190586]
    B = [[30.47983386, 56.41028485, 84.9595095 ], [32.95049616, 60.98284141, 91.84623527]]
    A = [138.22833333, 290.32783333, 434.75716667]
    X_new_scaled = (X_new - mu_X) / sigma_X
    Y_scaled_pred_direct = X_new_scaled @ B  
    Y_pred_direct = Y_scaled_pred_direct + A 
    
    return Y_pred_direct

def smoother(LLL,T1,LX,k):
    import numpy as np
    T = [np.mean([ j for j in LLL.iloc[i:i+k,LX] ]) for i in range(len(LLL)-k)]
    t = np.linspace(min(T1),max(T1),len(T)).reshape(1,-1)
    return t[0], np.array(T).reshape(1,-1)[0]

def solub(T):
    return (0.000011181*(T**2) + 0.0004036447*T + 0.0368389481)

def final_value(T1, WWW_1,finaltime):
    import numpy as np
    mask = (T1 > max(T1) - finaltime)
    return np.mean(WWW_1[mask][:],0)

def SED(L,W):
    
    return ((6*L*(W**2))/3.141592653589793)**(1/3)


# In[ ]:





# In[ ]:




