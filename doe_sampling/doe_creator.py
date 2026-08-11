import copy
import shutil
import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
import matplotlib        as mpl
import exp_generation.profile_gen_helper_funcs
from pathlib           import Path
from scipy.interpolate import interp1d
from dataclasses       import dataclass, field, fields
from itertools         import accumulate, chain
from openpyxl          import load_workbook, Workbook

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

    def set_doe_settings(self,exp_par, samples, Type = 'GREEDY', Fract = 0):    

        # 0. initial inputs
        self.samples = samples
        self.Fract   = Fract
        self.TYPE    = Type

        # 1. Correct the settings, so that if a section is false, all of its settings are neglected
        self.exp_par       = exp_generation.profile_gen_helper_funcs.exp_par_condition_corr(self.exp_par)

        # 2. Count the dimensions subject to change
        self.dimensions  = self.doe_sampler.dim_counter(exp_par)

        if self.dimensions < 1 or self.samples < 2:
            print(f'Dimensions:{self.dimensions}, samples {self.samples} !')
            print(f'Generating {self.samples} sample(s) using the identical, default settings !')
            self.if_sampling  = False 
        elif self.Fract > 2 and self.TYPE == 'FACT':
            print('Select Fract to be: 0>=Fract>=2}')
        else:
            print(f'Dimensions:{self.dimensions}, samples {self.samples} !')
            self.if_sampling = True

    def generate_experiments(self):
        # build the experimental DoE --> all profiles for each experiment    

        # generate DoE either Factorial or Optimal sampling-based
        if self.if_sampling:
            self.DoE_samples = self.doe_sampler.doe_generator(1,self.samples,self.dimensions,self.weights,self.TYPE,self.Fract)[0]
            # scale the sample points between [0;1] to cover entire variable space
            self.DoE_sampled_scaled = self.doe_sampler.scaler(self.DoE_samples)
        for i in range(self.samples): # take a sample
            dim_index = 0
            if self.if_sampling: # sampling is True then take the optimal samples otherwise just use the default .values
                

                for f in fields(self.exp_par):
                    if f.type.__name__ == "DefineProcessParameter" and getattr(self.exp_par, f.name).if_in_doe: # take one sub-sample from every dim.
                        param = getattr(self.exp_par, f.name)
                        print(f.name)
                        
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


            self.exp_storage.add_experiment(f'exp{i+1}', self.profiles.generate_profiles()[0],self.profiles.generate_profiles()[1])

            # Store the changed parameters of the DoE
            self.exp_storage.add_doe_stage( f'exp{i+1}', pd.DataFrame(self.DoE_val)) # this is empty when only one sample is taken

        for i in range(self.samples):
            exp_key = f'exp{i+1}'
            self.exp_storage.profile_omni[exp_key] = self.exp_storage.profile_omni[exp_key].drop_duplicates()

    def plot_sampled_points(self):
        if self.DoE_samples.size:
            self.doe_sampler.two_dim_plotter([self.doe_sampler.scaler(self.DoE_samples)],[self.TYPE])
            self.doe_sampler.three_dim_plotter(self.doe_sampler.scaler(self.DoE_samples))
        else:
            print('Generate samples first')

    def export_generated_expts(self,loc_to_save_expts):

        for i in range(self.samples):
            DAT = pd.DataFrame(self.exp_storage.profile_omni[f'exp{i+1}'])

            try:
                with pd.ExcelWriter(loc_to_save_expts, engine="openpyxl", mode="a") as writer:
                    DAT.to_excel(writer, sheet_name=f"exp{i+1}", index=False)
            except:
                with pd.ExcelWriter(loc_to_save_expts, engine="openpyxl", mode="w") as writer:
                    DAT.to_excel(writer, sheet_name=f"exp{i+1}", index=False)

    def export_OMNI_expts(self,loc_to_save_expts):
        mapping = {'T': 'MAIN-TEMP', 'AS': 'AS','stirring_rate': 'MAIN-STR','solution': 'SOLUTION','solvent': 'SOLVENT', 'seed': 'SEED', 
                   'clean_sol':'CLEANING', 'valve_state': 'VALVE', 'stirring_rate_seed': 'SEED-STIR','stirring_rate_sol':'SOLUTION-STIR'}

        reordered_exps = {}
        
        for i in range(self.samples):
            reordered_data = {}
            for key, value in mapping.items():
                reordered_data[f't_{value}'] = self.exp_storage.profile_omni[f'exp{i+1}'][f't_{key}']
                reordered_data[value] = self.exp_storage.profile_omni[f'exp{i+1}'][key]

            reordered_data = pd.DataFrame(reordered_data)
            reordered_exps[f'exp{i+1}'] =  reordered_data


        header = pd.read_csv(r"FILES\helper_files_for_omni_export\headers_OMNI.csv")
        HEAD = pd.DataFrame(header)
        HEAD = HEAD.apply(lambda col: pd.to_numeric(col, errors="coerce").fillna(col))

        name_to_save_expts = loc_to_save_expts+r'\agrprofiles.xlsx'

        try:
            wb = load_workbook(name_to_save_expts)
            for sheet in wb.worksheets[1:]:
                wb.remove(ws)
                wb.create_sheet(title=f"subexp{1}")
                wb.save(name_to_save_expts)
        except:
            wb = Workbook()
            default_sheet = wb.active
            wb.remove(default_sheet)
            wb.create_sheet(title=f"subexp{1}")
            wb.save(name_to_save_expts)

        for i in range(self.samples):

            DAT = pd.DataFrame(reordered_exps[f"exp{i+1}"])
            with pd.ExcelWriter(name_to_save_expts, engine="openpyxl", mode="a",if_sheet_exists='replace') as writer:
                DAT.to_excel(writer,  sheet_name=f"subexp{i+1}", index=False,startrow=7,header=False)
            with pd.ExcelWriter(name_to_save_expts, engine="openpyxl", mode="a",if_sheet_exists='overlay') as writer:    

                HEAD.to_excel(writer, sheet_name=f"subexp{i+1}", index=False,startrow=0,header=False)



        files_to_copy_w_OMNI_profile = [r"\cmp1.1.csv", r"\cmp2.1.csv", r"\cmp11.1.csv", r"\cmp12.1.csv",
                                        r"\cmp13.1.csv",r"\cmp14.1.csv",r"\cmp16.1.csv",r"\cmp18.1.csv",
                                        r"\cmp19.1.csv",r"\compids.csv",r"\expt.xml",    r"\sstrace.csv"]
        
        files_internal_location = r'FILES/OMNI_data'
        
        for file in files_to_copy_w_OMNI_profile:
            location_from = Path(files_internal_location +  file)
            location_to   = loc_to_save_expts + file
            shutil.copy2(location_from,location_to)

    def plot_doe_experiments(self):

        for i in range(self.samples):
            fig, ax = plt.subplots(figsize=(4,2))
            ax_AS   = ax.twinx()

            ax_AS.spines['right'].set_color('red')

            ax_seed = ax.twinx()
            ax_seed.spines['right'].set_position(('outward', 40))
            ax_seed.spines['right'].set_color('blue')

            ax_solution   = ax.twinx()
            ax_solution.spines['right'].set_position(('outward', 80))
            ax_solution.spines['right'].set_color('orange')

            ax_solv   = ax.twinx()
            ax_solv.spines['right'].set_position(('outward', 120))
            ax_solv.spines['right'].set_color('magenta')

            ax_cl   = ax.twinx()
            ax_cl.spines['right'].set_position(('outward', 160))
            ax_cl.spines['right'].set_color('cyan')

            ax_valve   = ax.twinx()
            ax_valve.spines['right'].set_position(('outward', 200))
            ax_valve.spines['right'].set_color('yellow')

            self.exp_storage.data_omni[f'exp{i+1}'].plot(x = 't', y='T'   ,       
                                                         color = 'k',      ax = ax,         linestyle ='-.' , label='T',        
                                                         ylabel='Temperature [C]')
            self.exp_storage.data_omni[f'exp{i+1}'].plot(x = 't', y='AS'  ,       
                                                         color = 'r',      ax = ax_AS,      linestyle ='--' , label='AS',       
                                                         ylabel='AS addition rate [rpm]')
            self.exp_storage.data_omni[f'exp{i+1}'].plot(x = 't', y='seed',       
                                                         color = 'b',      ax = ax_seed,    linestyle = '-' , label='seed',     
                                                         ylabel='wet seed addition rate [rpm]')
            self.exp_storage.data_omni[f'exp{i+1}'].plot(x = 't', y='solution',   
                                                         color = 'orange', ax = ax_solution,linestyle = '-.', label='solution', 
                                                         ylabel='solution addition rate [rpm]')
            self.exp_storage.data_omni[f'exp{i+1}'].plot(x = 't', y='solvent', 
                                                         color = 'magenta',ax = ax_solv,    linestyle = '-.', label='solvent',  
                                                         ylabel='solvent addition rate [rpm]')
            self.exp_storage.data_omni[f'exp{i+1}'].plot(x = 't', y='clean_sol', 
                                                         color = 'cyan',   ax = ax_cl,      linestyle = '-.', label='cleaning', 
                                                         ylabel='aceton addition rate [ml/h]')
            self.exp_storage.data_omni[f'exp{i+1}'].plot(x = 't', y='valve_state',            
                                                         color = 'yellow', ax = ax_valve,   linestyle = '-' , label='valve'   , 
                                                         ylabel='Valve stage [1 or 0]')

            ax.set_title(f'Exp{i+1}')

            ax.legend(loc='best', bbox_to_anchor=(1.0, 1.25),fontsize=8)
            ax_AS.legend(loc='best', bbox_to_anchor=(1.2, 1.25),fontsize=8)
            ax_solution.legend(loc='best', bbox_to_anchor=(0.85, 1.25),fontsize=8)
            ax_solv.legend(loc='best', bbox_to_anchor=(0.2, 1.25),fontsize=8)
            ax_cl.legend(loc='best', bbox_to_anchor=(1.5, 1.25),fontsize=8)
            ax_seed.legend(loc='best', bbox_to_anchor=(0.4, 1.25),fontsize=8)
            ax_valve.legend(loc='best', bbox_to_anchor=(1.8, 1.25),fontsize=8)

            ax.grid(alpha=0.2)
            #plt.show()
            self.plotting_figs.append(fig)
            self.plotting_axes.append({'T':ax, 'AS': ax_AS,'seed': ax_seed,'solution': ax_solution,
                                       'solvent': ax_solv,'clean_sol': ax_cl,'valve_state': ax_valve})
