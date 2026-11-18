from app.paths import (OMNI_DATA_DIR, OMNI_HEADER_FILE)

import copy
import shutil
import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
import matplotlib        as mpl

import experiment.profiles.profile_gen_helper_funcs as profile_gen_helper_funcs

from pathlib           import Path
from scipy.interpolate import interp1d
from dataclasses       import asdict, dataclass, field, fields
from itertools         import accumulate, chain
from openpyxl          import load_workbook, Workbook
from matplotlib.figure import Figure

mpl.rcParams.update({'font.size': 7,'axes.labelsize': 7,'axes.titlesize': 7,
                     'xtick.labelsize': 7,'ytick.labelsize': 7,'legend.fontsize': 7,
                     'grid.alpha': 0.5, 'grid.linewidth': 0.7,'grid.linestyle': '-', 
                     'xtick.minor.visible': True, 'ytick.minor.visible': True,})

class ExperimentCreator():
    def __init__(self,doe_sampler, component_gen, exp_par, profiles, exp_storage):
        self.exp_par       = exp_par
        self.doe_sampler   = doe_sampler
        self.component_gen = component_gen
        self.profiles      = profiles
        self.exp_storage   = exp_storage

        self.dimensions   = self.doe_sampler.dim_counter(exp_par)
        self.weights      = self.dimensions*[1] # weights to ExperimentProfileParameters !! - not working currently
        self.if_sampling  = False 
        self.samples      = 1        # to be set by user !!
        self.Fract        = 0        # to be set by user !!
        self.TYPE         = 'GREEDY' # to be set by user !!
        
        self.DoE_val = dict()
        self.DoE_samples = np.array([])
        self.DoE_sampled_scaled = np.array([])
        self.plotting_axes = []
        self.plotting_figs = []
        self.actually_sampled_experiments = 0
        self.parameter_snapshots = {}
        
    def set_doe_settings(self, exp_par, sampless, Type = 'GREEDY', Fract = 0):    

        # 0. initial inputs
        self.samples = sampless
        self.Fract   = Fract
        self.TYPE    = Type

        # 1. Correct the settings, so that if a section is false, all of its settings are neglected
        self.exp_par       = profile_gen_helper_funcs.exp_par_condition_corr(self.exp_par)

        # 2. Count the dimensions subject to change
        self.dimensions  = self.doe_sampler.dim_counter(exp_par)

        if self.dimensions < 1 or self.samples < 2:
            self.if_sampling  = False 
        else:
            self.if_sampling = True
            
    def generate_experiments(self):
        # build the experimental DoE --> all profiles for each experiment    

        # generate DoE either Factorial or Optimal sampling-based
        if self.if_sampling:
            self.DoE_samples = self.doe_sampler.doe_generator(1,self.samples,self.dimensions,self.weights,self.TYPE,self.Fract)[0]
            self.DoE_sampled_scaled = self.doe_sampler.scaler(self.DoE_samples)
            RANGE = len(self.DoE_samples) 
            
        elif not self.if_sampling and self.samples == 1:
            RANGE = 1
            
        else:
            RANGE = self.samples
            
        for i in range(RANGE): # take a sample
            dim_index = 0
            if self.if_sampling: # sampling is True then take the optimal samples otherwise just use the default .values
                for f in fields(self.exp_par):
                    if f.type.__name__ == "DefineProcessParameter" and getattr(self.exp_par, f.name).if_in_doe: # take one sub-sample from every dim.
                        param = getattr(self.exp_par, f.name)

                        if param.if_only_minmax: # if if_only_minmax == True don't ge\t the samples, only take min or max alternating
                            if i % 2 == 0:
                                setattr(param, "value", param.min_val)
                                self.DoE_val[f.name] = [param.min_val]                    
                            else:
                                setattr(param, "value", param.max_val)
                                self.DoE_val[f.name] = [param.max_val]

                        else: # if if_only_minmax == False take the sampled value
                            if np.shape(self.DoE_sampled_scaled)[1] == 1:
                                new_value = self.doe_sampler.descaler(self.DoE_sampled_scaled[i],param) # get samples
                            else:
                                new_value = self.doe_sampler.descaler(self.DoE_sampled_scaled[i,dim_index],param) # get samples
                            setattr(param, "value", new_value[0])
                            self.DoE_val[f.name] = [new_value[0]]

                        dim_index += 1

                    if self.exp_par.if_ARR:
                        self.exp_par.ARR = self.exp_par.ARR_types[i%len(self.exp_par.ARR_types)]
            self.parameter_snapshots[f'exp{i+1}'] = asdict(self.exp_par)
            generated, omni_profile = self.profiles.generate_profiles()
            self.exp_storage.add_experiment(f'exp{i+1}', generated, omni_profile)
            
            # Store the changed parameters of the DoE
            self.exp_storage.add_doe_stage( f'exp{i+1}', pd.DataFrame(self.DoE_val)) # this is empty when only one sample is taken
            self.actually_sampled_experiments += 1
            
        for i in range(self.actually_sampled_experiments):
            exp_key = f'exp{i+1}'
            self.exp_storage.profile_omni[exp_key] = self.exp_storage.profile_omni[exp_key].drop_duplicates()

    def plot_sampled_points(self, fig_2d=None, fig_3d=None):
        dim_names = [k for k in self.DoE_val.keys()]
        
        try:
            fig_2D = self.doe_sampler.two_dim_plotter([self.doe_sampler.scaler(self.DoE_samples)],self.TYPE,dim_names,fig=fig_2d)
        except:
            fig_2D = []
            
        try:    
            fig_3D = self.doe_sampler.three_dim_plotter(self.doe_sampler.scaler(self.DoE_samples),
                                               dimname1=dim_names[0],dimname2=dim_names[1],dimname3=dim_names[2],fig=fig_3d)
        except: 
            fig_3D = []
            
        return fig_2D, fig_3D
        
    def export_generated_expts(self,loc_to_save_expts):
        if self.actually_sampled_experiments < 1:
            raise ValueError("Generate at least one experiment before exporting")
        exists = Path(loc_to_save_expts).exists()
        options = {"if_sheet_exists": "replace"} if exists else {}
        with pd.ExcelWriter(loc_to_save_expts, engine="openpyxl", mode="a" if exists else "w", **options) as writer:
            for i in range(self.actually_sampled_experiments):
                DAT = pd.DataFrame(self.exp_storage.profile_omni[f'exp{i+1}'])
                DAT.to_excel(writer, sheet_name=f"exp{i+1}", index=False)

    def export_OMNI_expts(self,loc_to_save_expts):
        mapping = {'T': 'MAIN-TEMP', 'AS': 'AS','stirring_rate': 'MAIN-STR','solution': 'SOLUTION','solvent': 'SOLVENT', 'seed': 'SEED', 
                   'clean_sol':'CLEANING', 'valve_state': 'VALVE', 'stirring_rate_seed': 'SEED-STIR','stirring_rate_sol':'SOLUTION-STIR'}

        reordered_exps = {}
        
        for i in range(self.actually_sampled_experiments):
            reordered_data = {}
            for key, value in mapping.items():
                reordered_data[f't_{value}'] = self.exp_storage.profile_omni[f'exp{i+1}'][f't_{key}']
                reordered_data[value] = self.exp_storage.profile_omni[f'exp{i+1}'][key]

            reordered_data = pd.DataFrame(reordered_data)
            reordered_exps[f'exp{i+1}'] =  reordered_data


        header = pd.read_csv(OMNI_HEADER_FILE)
        
        HEAD = pd.DataFrame(header)
        HEAD = HEAD.apply(lambda col: pd.to_numeric(col, errors="coerce").fillna(col))

        name_to_save_expts = ( Path(loc_to_save_expts) / "agrprofiles.xlsx")

        if self.actually_sampled_experiments < 1:
            raise ValueError("Generate at least one experiment before exporting")
        if name_to_save_expts.exists():
            wb = load_workbook(name_to_save_expts)
            # Replace generated sheets only; preserve user-added sheets.
            for sheet in list(wb.worksheets):
                if sheet.title.startswith("subexp") and sheet.title[6:].isdigit():
                    wb.remove(sheet)
        else:
            wb = Workbook()
            wb.remove(wb.active)
        for i in range(self.actually_sampled_experiments):
            wb.create_sheet(title=f"subexp{i+1}", index=i)
        wb.save(name_to_save_expts)
        wb.close()

        for i in range(self.actually_sampled_experiments):

            DAT = pd.DataFrame(reordered_exps[f"exp{i+1}"])
            with pd.ExcelWriter(name_to_save_expts, engine="openpyxl", mode="a",if_sheet_exists='replace') as writer:
                DAT.to_excel(writer,  sheet_name=f"subexp{i+1}", index=False,startrow=7,header=False)
            with pd.ExcelWriter(name_to_save_expts, engine="openpyxl", mode="a",if_sheet_exists='overlay') as writer:    

                HEAD.to_excel(writer, sheet_name=f"subexp{i+1}", index=False,startrow=0,header=False)



        files_to_copy_w_OMNI_profile = [ "cmp1.1.csv","cmp2.1.csv","cmp11.1.csv",
                                         "cmp12.1.csv","cmp13.1.csv","cmp14.1.csv",
                                         "cmp16.1.csv","cmp18.1.csv","cmp19.1.csv",
                                         "compids.csv","expt.xml", "sstrace.csv"]
        
        files_internal_location = OMNI_DATA_DIR
        
        for filename in files_to_copy_w_OMNI_profile:
            
            location_from = OMNI_DATA_DIR / filename
            location_to = Path(loc_to_save_expts) / filename
            shutil.copy2( location_from, location_to )

    def plot_doe_experiments(self):
        from experiment.profile_plots import profile_figure
        labels = {"T": "Temperature", "AS": "AS", "seed": "Wet seed", "solution": "Solution",
                  "solvent": "Solvent", "clean_sol": "Cleaning", "valve_state": "Valve"}
        self.plotting_figs, self.plotting_axes = [], []
        for index, frame in enumerate(self.exp_storage.data_omni.values(), 1):
            signals = [{"key": role, "role": role, "label": label,
                        "unit": "°C" if role == "T" else "binary" if role == "valve_state" else "RPM",
                        "times": frame["t"], "values": frame[role]}
                       for role, label in labels.items() if role in frame]
            figure, axes = profile_figure(signals, index)
            self.plotting_figs.append(figure)
            self.plotting_axes.append(axes)
        return self.plotting_figs
