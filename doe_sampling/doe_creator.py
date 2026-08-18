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
        
    def set_doe_settings(self, exp_par, sampless, Type = 'GREEDY', Fract = 0):    

        # 0. initial inputs
        self.samples = sampless
        self.Fract   = Fract
        self.TYPE    = Type

        # 1. Correct the settings, so that if a section is false, all of its settings are neglected
        self.exp_par       = exp_generation.profile_gen_helper_funcs.exp_par_condition_corr(self.exp_par)

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
            self.exp_storage.add_experiment(f'exp{i+1}', self.profiles.generate_profiles()[0],self.profiles.generate_profiles()[1])
            
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

        for i in range(self.actually_sampled_experiments):
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
        
        for i in range(self.actually_sampled_experiments):
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

        for i in range(self.actually_sampled_experiments):

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
    
        # ---------------------------------------------------------
        # Reset previously generated figures
        # ---------------------------------------------------------
        self.plotting_figs = []
        self.plotting_axes = []
    
        # ---------------------------------------------------------
        # GUI / Matplotlib theme
        # ---------------------------------------------------------
        BG_COLOR       = "none"   # Main GUI background
        PANEL_COLOR    = "none"   # GroupBox / card background
        GRID_COLOR     = "#24364d"
        TEXT_COLOR     = "#95979c"
        MUTED_COLOR    = "#94a3b8"
        BORDER_COLOR   = "#16263d"
    
        # Plot colors
        COLORS = {
            "T":          "#95979c",
            "AS":         "#ef4444",
            "seed":       "#3b82f6",
            "solution":   "#f59e0b",
            "solvent":    "#d946ef",
            "clean_sol":  "#06b6d4",
            "valve_state":"#eab308",
        }
    
        # ---------------------------------------------------------
        # Generate one figure per experiment
        # ---------------------------------------------------------
        for i in range(self.actually_sampled_experiments):
    
            exp_name = f"exp{i + 1}"
            df = self.exp_storage.data_omni[exp_name]
    
            # -----------------------------------------------------
            # Figure
            # -----------------------------------------------------
            fig = Figure(figsize=(10, 4), dpi=100)
            ax = fig.add_subplot(111)
            fig.patch.set_facecolor(BG_COLOR)
                
            ax.set_facecolor(PANEL_COLOR)
    
            # -----------------------------------------------------
            # Additional Y axes
            # -----------------------------------------------------
            ax_AS = ax.twinx()
    
            ax_seed = ax.twinx()
            ax_seed.spines["right"].set_position(("outward", 45))
    
            ax_solution = ax.twinx()
            ax_solution.spines["right"].set_position(("outward", 90))
    
            ax_solv = ax.twinx()
            ax_solv.spines["right"].set_position(("outward", 135))
    
            ax_cl = ax.twinx()
            ax_cl.spines["right"].set_position(("outward", 180))
    
            ax_valve = ax.twinx()
            ax_valve.spines["right"].set_position(("outward", 225))
    
            axes = {
                "T": ax,
                "AS": ax_AS,
                "seed": ax_seed,
                "solution": ax_solution,
                "solvent": ax_solv,
                "clean_sol": ax_cl,
                "valve_state": ax_valve,
            }
    
            # -----------------------------------------------------
            # Style all axes
            # -----------------------------------------------------
            for key, current_ax in axes.items():
    
                current_ax.set_facecolor(PANEL_COLOR)
    
                # Tick colors
                current_ax.tick_params(
                    axis="both",
                    colors=MUTED_COLOR,
                    labelsize=8,
                    length=4
                )
    
                # Spine styling
                for spine in current_ax.spines.values():
                    spine.set_color(BORDER_COLOR)
                    spine.set_linewidth(0.8)
    
                # Right-side spine gets the corresponding signal color
                if key != "T":
                    current_ax.spines["right"].set_color(
                        COLORS[key]
                    )
    
                    current_ax.tick_params(
                        axis="y",
                        colors=COLORS[key]
                    )
    
            # -----------------------------------------------------
            # Main T axis
            # -----------------------------------------------------
            ax.spines["left"].set_color(COLORS["T"])
    
            ax.tick_params(
                axis="y",
                colors=COLORS["T"]
            )
    
            # -----------------------------------------------------
            # Plot data
            # -----------------------------------------------------
            ax.plot(df["t"],df["T"],color=COLORS["T"],linewidth=2.0,linestyle="-",label="Temperature")
            ax_AS.plot(df["t"],df["AS"],color=COLORS["AS"],linewidth=1.7,linestyle="--",label="AS")
            ax_seed.plot(df["t"],df["seed"],color=COLORS["seed"],linewidth=1.7,linestyle="-",label="Wet seed")
            ax_solution.plot(df["t"],df["solution"],color=COLORS["solution"],linewidth=1.7,linestyle="-.",label="Solution")
            ax_solv.plot(df["t"],df["solvent"],color=COLORS["solvent"],linewidth=1.7,linestyle="-.",label="Solvent")
            ax_cl.plot(df["t"],df["clean_sol"],color=COLORS["clean_sol"],linewidth=1.7,linestyle="-.",label="Cleaning")
            ax_valve.plot(df["t"],df["valve_state"],color=COLORS["valve_state"],linewidth=1.5,linestyle="-",drawstyle="steps-post",label="Valve")
    
            # -----------------------------------------------------
            # Grid
            # -----------------------------------------------------
            ax.grid(True,which="major",axis="both",color=GRID_COLOR,linestyle="--",linewidth=0.7,alpha=0.7)
    
            ax.set_axisbelow(True)
    
            # -----------------------------------------------------
            # Labels
            # -----------------------------------------------------
            ax.set_xlabel("Time",color=TEXT_COLOR,fontsize=9,labelpad=8)
            ax.set_ylabel("Temperature [°C]",color=COLORS["T"],fontsize=9)
            ax_AS.set_ylabel("AS [rpm]",color=COLORS["AS"],fontsize=8)
            ax_seed.set_ylabel("Wet seed [rpm]",color=COLORS["seed"],fontsize=8)
            ax_solution.set_ylabel("Solution [rpm]",color=COLORS["solution"],fontsize=8)
            ax_solv.set_ylabel("Solvent [rpm]",color=COLORS["solvent"],fontsize=8)
            ax_cl.set_ylabel("Cleaning [rpm]",color=COLORS["clean_sol"],fontsize=8)
            ax_valve.set_ylabel("Valve",color=COLORS["valve_state"],fontsize=8)
    
            # -----------------------------------------------------
            # Title
            # -----------------------------------------------------
            #ax.set_title(f"Experiment {i + 1}",loc="left",color=TEXT_COLOR,fontsize=13,fontweight="bold",pad=12)
    
            # Small experiment indicator
            ax.text(1.0,1.04,f"EXP {i + 1:02d}",transform=ax.transAxes,ha="right",va="bottom",color=MUTED_COLOR,fontsize=8,fontweight="bold" )
    
            # -----------------------------------------------------
            # Single combined legend
            # -----------------------------------------------------
            handles = [
                plt.Line2D([],[],color=COLORS["T"],linewidth=2,label="Temperature" ),
                plt.Line2D([],[],color=COLORS["AS"],linewidth=1.7,linestyle="--",label="AS" ),
                plt.Line2D([],[],color=COLORS["seed"],linewidth=1.7,label="Wet seed" ),
                plt.Line2D([],[],color=COLORS["solution"],linewidth=1.7,linestyle="-.",label="Solution" ),
                plt.Line2D([],[],color=COLORS["solvent"],linewidth=1.7,linestyle="-.",label="Solvent" ),
                plt.Line2D([],[],color=COLORS["clean_sol"],linewidth=1.7,linestyle="-.",label="Cleaning" ),
                plt.Line2D([],[],color=COLORS["valve_state"],linewidth=1.5,label="Valve" ),
            ]
    
            legend = ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.25),
                ncol=4,
                fontsize=8,
                frameon=True,
                fancybox=True,
                framealpha=0.95,
                borderpad=0.7,
                columnspacing=1.4,
                handlelength=2.5
            )
    
            legend.get_frame().set_facecolor(PANEL_COLOR)
            legend.get_frame().set_edgecolor(BORDER_COLOR)
    
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)
    
            # -----------------------------------------------------
            # Remove unnecessary top/right/left clutter
            # -----------------------------------------------------
            ax.spines["top"].set_visible(False)
    
            for current_ax in [
                ax_AS,
                ax_seed,
                ax_solution,
                ax_solv,
                ax_cl,
                ax_valve
            ]:
                current_ax.spines["top"].set_visible(False)
    
            # -----------------------------------------------------
            # Layout
            # -----------------------------------------------------
            fig.subplots_adjust(
                left=0.08,
                right=0.68,
                top=0.82,
                bottom=0.15
            )
    
            # -----------------------------------------------------
            # Store
            # -----------------------------------------------------
            self.plotting_figs.append(fig)
    
            self.plotting_axes.append({
                "T": ax,
                "AS": ax_AS,
                "seed": ax_seed,
                "solution": ax_solution,
                "solvent": ax_solv,
                "clean_sol": ax_cl,
                "valve_state": ax_valve
            })
    
        return self.plotting_figs