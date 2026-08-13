#!/usr/bin/env python
# coding: utf-8

# In[343]:


import matplotlib               as mpl
import pandas                   as pd
import matplotlib.pyplot        as plt
import numpy                    as np
from scipy.optimize             import minimize
from matplotlib.ticker          import MaxNLocator
from matplotlib.figure          import Figure

from app_control.pure_functions import W_pred, corr_func, L_pred, smoother, solub, final_value, SED, file_finder

class DataPlotter:
    def __init__(self, omni_filename="expttsd", imed_filename="imed"):
        self.omni_filename = "expttsd"
        self.imed_filename = "imed"
        self.omni_folder = None
        self.DESC_best_W = [2]
        self.DESC_best = [1, 2]
        self.finaltime = 10  # minutes
        self.colors = ['orange', 'magenta', 'blue', 'red', 'green']
        self.conc_corr = True
        self.sizedata = []
        self.EVAL = []
        self.popt = [1, 0]
        self.read_files = []
        self.read_data_from = []
        self.COLUMNS = []
        self.PRE_COLUMNS = []
        self.PRO_COLUMNS = []

        # Modern Matplotlib dark theme configuration
        mpl.rcParams.update({
            'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
            'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
            'grid.alpha': 0.3, 'grid.linewidth': 0.6, 'grid.linestyle': '--',
            'xtick.minor.visible': True, 'ytick.minor.visible': True,

        })
        
    def plot_generator(self):
        if not self.EVAL:
            return None
    
        NBINS = 5
        num_rows = len(self.EVAL) # Exact count of experiments
    
        fig_width = 13.0
        fig_height = max(3.0, num_rows * 2.2)
    
        self.fig = Figure(
            figsize=(fig_width, fig_height), 
            facecolor='none', 
            layout='constrained'
        )
        
        self.axT = self.fig.subplots(num_rows, NBINS)
    
        if num_rows == 1:
            self.axT = np.atleast_2d(self.axT)
    
        if self.idx_t == []:
            return self.fig
    
        for i, E in enumerate(self.EVAL):
            E.columns = self.COLUMNS
    
            # --- Your existing plotting code for counts, conc, length, width, CED ---
            # (Keep the logic we adjusted earlier here)
            for k in range(NBINS):
                ax = self.axT[i, k]
                ax.set_xlabel(f"{self.COLUMNS[self.idx_t]} [{self.PRO_COLUMNS[self.idx_t]}]")
                
                ax.grid(which='major', alpha=0.2)
                ax.grid(which='minor', alpha=0.075)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=NBINS))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=NBINS))
                ax.tick_params(axis='both', labelsize=6)
    
                if self.idx_T:
                    ax.plot(E.iloc[:, self.idx_t], E.iloc[:, self.idx_T], 'white', linestyle='--', alpha=0.9)
                    if k == 0:
                        ax.set_title(f"Experiment no. {i}", color ='white')
                        ax.set_ylabel(f"{self.PRE_COLUMNS[self.idx_T]} [{self.PRO_COLUMNS[self.idx_T]}]", fontsize=6)
    
                if self.idx_AS and k == 0:
                    ax_AS = ax.twinx()
                    ax_AS.plot(E.iloc[:, self.idx_t], E.iloc[:, self.idx_AS], 'red', linestyle='--', alpha=0.9)
                    ax_AS.set_ylabel(f"{self.PRE_COLUMNS[self.idx_AS]}", color='red', fontsize=6)
                    ax_AS.spines['right'].set_color('red')
                    ax_AS.tick_params(axis='y', colors='red', labelsize=6)
                    ax_AS.spines['right'].set_position(('outward', 40))
                    
    
            # Counts Data (Column 0)
            if self.idx_FBRM:
                ax_FBRM = self.axT[i, 0].twinx()
                color = self.colors[i % 5]
                ax_FBRM.plot(E.iloc[:,self.idx_t], E.iloc[:,self.idx_FBRM], color, linestyle='-', alpha=0.7)
                ax_FBRM.tick_params(axis='y', colors=color)
                ax_FBRM.spines['right'].set_color(color)
                ax_FBRM.set_ylabel(f"{self.PRE_COLUMNS[self.idx_FBRM]} [{self.PRO_COLUMNS[self.idx_FBRM]}]",color=color)
                ax_FBRM.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            else:
                self.axT[i, 0].text(0.3, 0.8, 'No counts data', transform=self.axT[i, 0].transAxes, color=self.colors[i % 5])
    
            if self.idx_c:
                ax_conc = self.axT[i, 1].twinx()
                color = self.colors[i % 5]
                if self.conc_corr:
                    bnds = ((1e-10, 100), (-10, 100))
                    x0 = np.array([1.1, 0.002])
                    # Ensure smoother function and solub are defined externally
                    conc_smooth_t, conc_smooth = smoother(E.iloc[:,self.idx_c],E.iloc[:,self.idx_t].to_numpy(), 0, 20)
                    X = [pd.DataFrame(conc_smooth), solub(E.iloc[:,self.idx_T])]
                    res = minimize(corr_func, x0, args=(X), method='nelder-mead', bounds=bnds, options={'xatol': 1e-20, 'disp': False})
                    self.popt = [res.x[0], res.x[-1]]
                    ax_conc.plot(conc_smooth_t, self.popt[0] * X[0] + self.popt[1], c=color, alpha=0.7)
                else:
                    ax_conc.plot(E.iloc[:,self.idx_t], self.popt[0] * E.iloc[:,self.idx_c] + self.popt[1], c=color, alpha=0.7)
                
                ax_conc.plot(E.iloc[:,self.idx_t], solub(E.iloc[:,self.idx_T]), '-.', c=color, alpha=0.5)
                ax_conc.set_ylabel(f"{self.PRE_COLUMNS[self.idx_c]}\n[{self.PRO_COLUMNS[self.idx_c]}]")
                ax_conc.tick_params(axis='y', colors=color)
                ax_conc.spines['right'].set_color(color)
                ax_conc.yaxis.set_major_locator(MaxNLocator(nbins=NBINS))
            else:
                self.axT[i, 1].set_title('No concentration data',color=self.colors[i % 5])

            if self.idx_L:
                ax_L = self.axT[i, 2].twinx()
                color = self.colors[i % 5]
                for col_name in ['OPC R: L10 ', 'OPC R: L50 ', 'OPC R: L90 ']:
                    if col_name in self.COLUMNS:
                        ax_L.plot(E.iloc[:,self.idx_t], E[col_name], color, linestyle='-', alpha=0.7)
                ax_L.yaxis.set_major_locator(MaxNLocator(nbins=NBINS))
                ax_L.tick_params(axis='y', colors=color)
                ax_L.spines['right'].set_color(color)
                ax_L.set_ylabel(f"Length [{self.PRO_COLUMNS[self.idx_L]}]",color=color)
            else:
                self.axT[i, 2].set_title('No length data',color=self.colors[i % 5])
                
            if self.idx_W:
                ax_W = self.axT[i, 3].twinx()
                for col_name in ['OPC R: W10 ', 'OPC R: W50 ', 'OPC R: W90 ']:
                    if col_name in self.COLUMNS:
                        ax_W.plot(E.iloc[:,self.idx_t], E[col_name], color, linestyle='-', alpha=0.7)
                ax_W.yaxis.set_major_locator(MaxNLocator(nbins=NBINS))
                ax_W.tick_params(axis='y', colors=color)
                ax_W.spines['right'].set_color(color)
                ax_W.set_ylabel(f"Width [{self.PRO_COLUMNS[self.idx_W]}]",color=color)
            else:
                self.axT[i, 3].set_title('No width data',color=self.colors[i % 5])
            
            if self.idx_CE:
                ax_CE = self.axT[i, 4].twinx()
                for col_name in ['OPC R: CED10 micron', 'OPC R: CED50 micron', 'OPC R: CED90 micron']:
                    if col_name in self.COLUMNS:
                        ax_CE.plot(E.iloc[:,self.idx_t], E[col_name], color, linestyle='-', alpha=0.7)
                ax_CE.yaxis.set_major_locator(MaxNLocator(nbins=NBINS))
                ax_CE.tick_params(axis='y', colors=color)
                ax_CE.spines['right'].set_color(color)
                ax_CE.set_ylabel(f"CED [{self.PRO_COLUMNS[self.idx_W]}]",color=color)
            else:
                self.axT[i, 4].set_title('No CED data',color=self.colors[i % 5])
        # 2. FIX: Apply dark mode aesthetics to ALL rows (self.axT instead of self.axT[:-1])
        for ax_row in self.axT:
            for ax in ax_row:
                ax.set_facecolor('#1f2937')
                for spine in ax.spines.values():
                    spine.set_color('#9ca3af')
                ax.tick_params(colors='#9ca3af')
                ax.xaxis.label.set_color('#9ca3af')
                ax.yaxis.label.set_color('#9ca3af')
    
        # 3. FIX: Remove the delaxes loop entirely! 
        # (Since num_rows == len(EVAL), there are no trailing unused axes)
    
        return self.fig

    def data_evaluator(self, E1, k):
        E1.columns = self.COLUMNS
        if 'Time' in self.COLUMNS and len(E1['Time']) > 0:
            T1 = E1['Time'].to_numpy()
            SW1 = [] if not all(i in self.COLUMNS for i in {'OPC R: W10 ', 'OPC R: W50 ', 'OPC R: W90 '}) else []
            SL1 = [] if not all(i in self.COLUMNS for i in {'OPC R: L10 ', 'OPC R: L50 ', 'OPC R: L90 '}) else []
            return T1, [], [], SL1, SW1
        return [], [], [], [], []
        
    def data_reader(self):
        for i in file_finder(self.read_data_from, self.omni_filename):
            if i not in self.read_files:
                E1 = pd.read_csv(i, header=[0, 1, 2])
                self.EVAL.append(E1)
                self.read_files.append(i)
        
        if not self.EVAL:
            return

        if not self.COLUMNS:
            for col in self.EVAL[0].columns:
                self.COLUMNS.append(col[1])
                self.PRE_COLUMNS.append(col[0])
                self.PRO_COLUMNS.append(col[2])
        try:
            self.idx_t  = self.COLUMNS.index('Time')
        except:
            self.idx_t  = []
            
        try:
            self.idx_T  = self.COLUMNS.index('H01:External T')
        except:
            self.idx_T  = []
            
        try:
            self.idx_AS = self.COLUMNS.index('File  W: AS flow ')
        except:
            self.idx_AS  = []
            
        try:
            self.idx_FBRM  = self.COLUMNS.index('OPC R: Counts ')
        except:
            self.idx_FBRM   = []
            
        try:
            self.idx_c = self.COLUMNS.index('File R: Conc ')
        except:
            self.idx_c  = []
            
        try:
            self.idx_L = self.COLUMNS.index('OPC R: L50 ')
        except:
            self.idx_L  = []

        try:
            self.idx_W = self.COLUMNS.index('OPC R: W50 ')
        except:
            self.idx_W  = []

        try:
            self.idx_CE = self.COLUMNS.index('OPC R: CED50 micron')
        except:
            self.idx_CE  = []
