
import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
import matplotlib        as mpl
import copy
from scipy.interpolate   import interp1d

from dataclasses         import dataclass, field
from itertools           import accumulate, chain
from dataclasses         import fields, dataclass

# My stuff
from exp_generation.pump_calibration    import PumpRateConverter
from doe_sampling.doe_sampler         import DoESampler

mpl.rcParams.update({'font.size': 7,'axes.labelsize': 7,'axes.titlesize': 7,'xtick.labelsize': 7,'ytick.labelsize': 7,'legend.fontsize': 7,
                     'grid.alpha': 0.5, 'grid.linewidth': 0.7,'grid.linestyle': '-', 'xtick.minor.visible': True, 'ytick.minor.visible': True,})
# Experimental operating parameters
@dataclass
class DefineProcessParameter():
    # ------------------------------------------------------------------------------
    # Define the role of an operating condition:
    # value            : the value it should hold if it is used as a 'constant variable'
    # min_val & max_val: if the variable is selected as a 'modified variable' in the DoE 
    #                    the actual value is selected of the [min_val,max_val] interval
    # if_in_doe        : whether it is 'constant' or 'modified' on the DoE generation
    # if_only_minmax   : if_only_minmax == True and if_in_doe == True, the value is not 
    #                   'sampled' only selected to alternate between min_val and max_val 
    # ------------------------------------------------------------------------------

    value          : float 
    min_val        : float 
    max_val        : float
    if_in_doe      : bool
    if_only_minmax : bool 

@dataclass
class ProfileCreator():
    # -------------------------------------------------------------------------------------------------------
    # Each method represents an 'experimental profile type' that create the parallel profiles for the devices
    # -------------------------------------------------------------------------------------------------------

    t              : list = field(default_factory=list)
    T              : list = field(default_factory=list)
    AS             : list = field(default_factory=list)
    seed           : list = field(default_factory=list)
    clean_sol      : list = field(default_factory=list)
    solution       : list = field(default_factory=list)
    solvent        : list = field(default_factory=list)
    valve_state    : list = field(default_factory=list)
    stirring_rate  : list = field(default_factory=list)

# Store process parameters
@dataclass
class ExperimentProfileParameters():
    # ---------------------------------------
    # Stores default parameters for a profile
    # ---------------------------------------

    # System defaults
    v_heat_max             : float = 2      # Celsius / min
    v_cool_max             : float = 2      # Celsius / min
    AS_add_max             : float = 1.8    # g/min
    T_room                 : float = 20     # Celsiu   s
    sol_add_max            : float = 66     # g/min
    time_to_wait_prior_exp : float = 0.083  # h = 5 min
    aceton_boil_out_T      : float = 56     # Celsius
    wait_after_exp         : float = 0.1     # Celsius
    t_before_open_valve    : float = 0.1

    # Valve actuation
    if_opening          : bool = True
    t_h_open            : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(0.05,  0.0331, 0.1,   False, False))
    be_opened_until     : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(0.03,  0.01,   0.2,   False, False)) # h

    # Cleaning solvent addition
    if_cleaning         : bool = True
    cleaning_sol_amount : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(400 ,  190,     210,   False, False)) 
    t_h_clean           : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(0.05 , 0.0332,  0.1  , False, False))

    # Solvent rinse
    if_solv_rinse       : bool = True
    solv_amount         : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(400 , 190,   210, False, False)) 
    t_h_solv            : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(0.05, 0.033, 0.1, False, False))

    # Initial Solution addition step  - - - (!!!) changing Initial_sol_amount changes everything in the experiment ratios - - - - 
    if_sol_initial      : bool = True
    Initial_sol_amount  : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(200 ,190,  210 ,  False, False)) 
    t_h_solution        : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(0.1 ,0.03, 0.15 , False, False))

    # Stirring rate 
    stir_r              : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(350, 100, 400, False, False))

    # Initial dissolution step (T_diss) and  Aproaching initial stage (T_0)
    if_diss_initial     : bool = True
    T_diss              : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(70 ,  70   ,  50  , False, False))
    t_diss              : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(0.3,  0.1  ,   0.9, False, False))
    T_0                 : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(65 ,  50   ,   55 , False, False))
    t_h_0               : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(1  ,  0.336, 1    ,  False, False))

    # Adding initial AS prior to crystallization to set Ssat
    if_AS_initial       : bool = False
    AS_initial          : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(5,   1,      10,   False, False))
    t_h_as_initial      : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(0.1, 0.337,  1 ,   False, False))

    # Seeding
    if_seed             : bool = True
    seed                : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(2, 0,       5 , True, False))
    seeding_rpm         : float                  = 50.0 # rpm with which the pump was calibrated for seeding 
    seed_rate           : float                  = 1 # rpm
    seed_time           : float                  = 1 
    t_h_seed            : DefineProcessParameter = field(default_factory=lambda: DefineProcessParameter(1, 0.338,   1 , False, False))  
    

    seed_calibration_parameters: list = field(default_factory = lambda  : [ 0.11196005,  0.42810161, 29.11837123] ) # parameters of a second order polinomial

    # ARRANGEMENT
    if_ARR              : bool = True
    ARR                 : str                     = "T"
    ARR_types           : list[str]               = field(default_factory=lambda: ['AS', 'SIM', 'T'])

    # AS addition
    if_AS               : bool = True
    AS_am               : DefineProcessParameter  = field(default_factory=lambda: DefineProcessParameter(230, 200 ,  250, False, False))
    AS_add_time         : DefineProcessParameter  = field(default_factory=lambda: DefineProcessParameter(4,   2 ,    8,   True, False))
    t_h_AS              : DefineProcessParameter  = field(default_factory=lambda: DefineProcessParameter(0.1, 0.339, 1,   False, False))

    # Cooling
    if_cool_in_exp      : bool = True
    t_cool              : DefineProcessParameter  = field(default_factory=lambda: DefineProcessParameter(2,   1,      10,  True, False ))
    T_final             : DefineProcessParameter  = field(default_factory=lambda: DefineProcessParameter(25,  10,     30,  True, False))
    t_h_cool            : DefineProcessParameter  = field(default_factory=lambda: DefineProcessParameter(0.1, 0.3310, 1,   False, False))

    # Equillibrium time
    t_eq                : DefineProcessParameter  = field(default_factory=lambda: DefineProcessParameter(2,   1,       3,  False, False))
    
    # Final boiling out
    if_boil             : bool = False
    t_boil              : DefineProcessParameter  = field(default_factory=lambda: DefineProcessParameter(0.5,   0.1,      1,  False, False ))
    t_h_boil            : DefineProcessParameter  = field(default_factory=lambda: DefineProcessParameter(0.5,   0.1,      1,  False, False )) 

# In[194]:


# Store experimental profiles and convert them into OMNI inputs
class ExperimentProfileStorage():

    def __init__(self,as_rate_converter,solution_rate_converter,solvent_rate_converter,cleaning_sol_rate_converter):

        self.data                         = dict()
        self.data_omni                    = dict()
        self.profile                      = dict()
        self.profile_omni                 = dict()
        
        self.DoE                          = dict()

        self.as_rate_converter            = as_rate_converter
        self.solution_rate_converter      = solution_rate_converter
        self.solvent_rate_converter       = solvent_rate_converter
        self.cleaning_sol_rate_converter  = cleaning_sol_rate_converter

    def add_experiment(self, name, profile, profile_omni):
        self.data[name] = profile
        self.profile[name] = profile_omni

        self.convert_to_omni()
        
        
    def add_doe_stage(self, name, profile):
        self.DoE[name] = profile

    def convert_to_omni(self):

        if self.data:
            self.data_omni    = copy.deepcopy(self.data)
            self.profile_omni = copy.deepcopy(self.profile)
            
            for exp in self.data:
                self.data_omni[exp]['AS'] = self.as_rate_converter.convert_to_rpm(self.data[exp]['AS'])
            for exp in self.data:
                self.data_omni[exp]['solution'] = self.solution_rate_converter.convert_to_rpm(self.data[exp]['solution'])
            for exp in self.data:
                self.data_omni[exp]['solvent'] = self.solvent_rate_converter.convert_to_rpm(self.data[exp]['solvent'])
            for exp in self.data:
                self.data_omni[exp]['clean_sol'] = self.cleaning_sol_rate_converter.convert_to_rpm(self.data[exp]['clean_sol'])

            for exp in self.profile:
                self.profile_omni[exp]['AS'] = self.as_rate_converter.convert_to_rpm(self.profile[exp]['AS'])
            for exp in self.profile:
                self.profile_omni[exp]['solution'] = self.solution_rate_converter.convert_to_rpm(self.profile[exp]['solution'])
            for exp in self.profile:
                self.profile_omni[exp]['solvent'] = self.solvent_rate_converter.convert_to_rpm(self.profile[exp]['solvent'])
            for exp in self.profile:
                self.profile_omni[exp]['clean_sol'] = self.cleaning_sol_rate_converter.convert_to_rpm(self.profile[exp]['clean_sol'])
        
        else:

            raise ValueError('generate data first!')


# In[196]:


class ExpComponentGen():
    # ----------------------------------------------------------
    # II. Generate the segments one-by-one
    # ----------------------------------------------------------

    def __init__(self,helper_funcscs):
        self.helper_funcscs = helper_funcscs
        self.seed_pred = lambda X, m_seed: X[0] + m_seed**X[1]*X[2] # function to convert mass of seed to pumping %%time
        self.AVOID_ZERO_DIFF = 1.0e-3 # 'epsilon' to avid zero t-diff
        
    def valve_prof_gen(self,ExpParams):

        section_datastruct      = ProfileCreator() # create data class
        self.helper_funcscs.initialize(section_datastruct) # initialize all segments

        section_datastruct.T             = [ExpParams.T_room]
        section_datastruct.stirring_rate = [ExpParams.stir_r.value]*len(section_datastruct.t)

        if ExpParams.if_opening:
            initial_struct          = [1e-10, ExpParams.be_opened_until.value, 0.001, ExpParams.t_h_open.value] # ExpParams.AS_add_max is in g / min
            N = len(initial_struct)
            self.helper_funcscs.segment_len_correcter(section_datastruct,N) # correct unused profiles to have [0]*N length

            section_datastruct.t    = initial_struct
            setattr(section_datastruct, 'valve_state', [1, 1, 0, 0])
            section_datastruct.T    = [ExpParams.T_room]*N
            section_datastruct.stirring_rate = [ExpParams.stir_r.value]*len(section_datastruct.t)

        return section_datastruct

    def sol_add_gen(self,fieldname,ExpParams): # by fieldname, it can be changed to generate profiles for solv, solu, or cleaning

        initial_sol_add      = ProfileCreator() # create data class
        self.helper_funcscs.initialize(initial_sol_add) # initialize all segments
        initial_sol_add.T             = [ExpParams.T_room]
        initial_sol_add.stirring_rate = [ExpParams.stir_r.value]*len(initial_sol_add.t)

        if ExpParams.if_sol_initial:
            initial_sol          = [1e-10, abs(ExpParams.Initial_sol_amount.value)/(60*ExpParams.sol_add_max), 0.001, ExpParams.t_h_solution.value] # ExpParams.AS_add_max is in g / min
            N                    = len(initial_sol)

            self.helper_funcscs.segment_len_correcter(initial_sol_add,N) # correct unused profiles to have [0]*N length

            initial_sol_add.t    = initial_sol

            setattr(initial_sol_add,fieldname,[60*ExpParams.sol_add_max, 60*ExpParams.sol_add_max, 0, 0])

            initial_sol_add.T    = [ExpParams.T_room]*N
            initial_sol_add.stirring_rate = [ExpParams.stir_r.value]*len(initial_sol_add.t)

        return initial_sol_add

    def occasional_temp_prof_gen(self,T_start,T_higher,T_final,t_at_high, t_h, ExpParams, preceeding_proc, if_occ_temp=True):
        # ---------------------------------------------------------------------------------
        # Generating the 'Initial segment' for dissolution, and setting starting temp
        # Returns the time points, the T, AS, and seeding profile while Seeding
        # ---------------------------------------------------------------------------------
        initial_T      = ProfileCreator() # create data class
        self.helper_funcscs.initialize(initial_T) # initialize all segments
        initial_T.T    = [preceeding_proc.T[-1]]
        initial_T.stirring_rate = [ExpParams.stir_r.value]*len(initial_T.t)

        if if_occ_temp:
            init_T_prof    = [T_start, T_higher, T_higher, T_final, T_final]
            self.helper_funcscs.init_temp_segment(initial_T,init_T_prof) # initialize Temp segments
            initial_T.t    = [self.AVOID_ZERO_DIFF, 
                              (1.0/60.0) * abs(T_higher - T_start) / ExpParams.v_heat_max, 
                              t_at_high, 
                              (1.0/60.0)*abs(T_final - T_higher)/(ExpParams.v_cool_max), 
                              t_h]

            initial_T.T    = init_T_prof
            initial_T.stirring_rate = [ExpParams.stir_r.value]*len(initial_T.t)

        return initial_T

    def initial_antisolv_gen(self,preceedeing_sect,ExpParams):   
        # ---------------------------------------------------------------------------
        # Generating the Initial AS profile for setting the preceedeing_sect Ssat
        # Returns the time points, the T, AS, and seeding profile while Seeding
        # ---------------------------------------------------------------------------
        initial_AS      = ProfileCreator() # create data class
        self.helper_funcscs.initialize(initial_AS) # set every to [0]
        initial_AS.T    = [preceedeing_sect.T[-1]]
        initial_AS.stirring_rate = [ExpParams.stir_r.value]*len(initial_AS.t)

        if ExpParams.if_AS_initial:

            initial_AS_t    = [self.AVOID_ZERO_DIFF, 
                               abs(ExpParams.AS_initial.value)/(60*ExpParams.AS_add_max),
                               self.AVOID_ZERO_DIFF, 
                               ExpParams.t_h_as_initial.value] # ExpParams.AS_add_max is in g / min

            N               = len(initial_AS_t)
            self.helper_funcscs.segment_len_correcter(initial_AS,N) # correct unused profiles to have [0]*N length
            initial_AS.t    = initial_AS_t
            initial_AS.AS   = [60*ExpParams.AS_add_max, 60*ExpParams.AS_add_max, 0, 0]
            initial_AS.T    = [preceedeing_sect.T[-1]]*N
            initial_AS.stirring_rate = [ExpParams.stir_r.value]*len(initial_AS.t)
        return initial_AS

    def seed_prof_gen(self,preceedeing_sect,ExpParams):
        # ------------------------------------------------------------------------
        # Generating the SEED profile
        # Returns the time points, the T, AS, and seeding profile while Seeding
        # ------------------------------------------------------------------------
        seed_prof      = ProfileCreator() # create data class
        self.helper_funcscs.initialize(seed_prof)
        seed_prof.T    = [preceedeing_sect.T[-1]]

        t_seed         = self.seed_pred(ExpParams.seed_calibration_parameters, ExpParams.seed.value) # convert the seed amount to seed pumping time in s
        seed_prof.stirring_rate = [ExpParams.stir_r.value]*len(seed_prof.t)

        if ExpParams.if_seed:

            seed_prof_t    = [self.AVOID_ZERO_DIFF, t_seed / 3600, self.AVOID_ZERO_DIFF, ExpParams.t_h_seed.value]
            N              = len(seed_prof_t)
            self.helper_funcscs.segment_len_correcter(seed_prof,N) # correct unused profiles to have [0]*N length

            seed_prof.t    = seed_prof_t
            seed_prof.seed = [50, 50, 0, 0]
            seed_prof.T    = [preceedeing_sect.T[-1]]*N
            seed_prof.stirring_rate = [ExpParams.stir_r.value]*len(seed_prof.t)

        return seed_prof

    def antisolv_gen(self,preceeding_sect,ExpParams): 
        # ------------------------------------------------------------------------
        # Generating the AS profile
        # Returns the time points, the T, AS, and seeding profile while AS addition
        # ------------------------------------------------------------------------
        AS      = ProfileCreator() # create data class
        self.helper_funcscs.initialize(AS)
        AS.T    = [preceeding_sect.T[-1]]
        AS.stirring_rate = [ExpParams.stir_r.value]*len(AS.t)

        if ExpParams.if_AS:

            AS_t    = [self.AVOID_ZERO_DIFF, ExpParams.AS_add_time.value, 0.001, ExpParams.t_h_AS.value]
            N       = len(AS_t)
            self.helper_funcscs.segment_len_correcter(AS,N) # correct unused profiles to have [0]*N length
            AS.t    = AS_t 
            AS.AS   = [(ExpParams.AS_am.value/ExpParams.AS_add_time.value)*ExpParams.if_AS,
                       (ExpParams.AS_am.value/ExpParams.AS_add_time.value)*ExpParams.if_AS,
                       0,
                       0]
            AS.T    = [preceeding_sect.T[-1]]*N
            AS.stirring_rate = [ExpParams.stir_r.value]*len(AS.t)

        return AS

    def temp_prof_gen(self,preceeding_sect,ExpParams):   
        # ---------------------------------------------------------------------------
        # Generating the cooling profile
        # Returns the time points, the T, AS, and seeding profile while cooling
        # ---------------------------------------------------------------------------
        T_prof      = ProfileCreator() # create data class
        self.helper_funcscs.initialize(T_prof)
        T_prof.T    = [preceeding_sect.T[-1]]
        T_prof.stirring_rate = [ExpParams.stir_r.value]*len(T_prof.t)    

        if ExpParams.if_cool_in_exp:
            T_prof_t    = [self.AVOID_ZERO_DIFF, ExpParams.t_cool.value, ExpParams.t_h_cool.value]
            N           = len(T_prof_t)
            self.helper_funcscs.segment_len_correcter(T_prof,N) # correct unused profiles to have [0]*N length
            T_prof.t    = T_prof_t
            T_prof.T    = [preceeding_sect.T[-1], ExpParams.T_final.value,ExpParams.T_final.value]
            T_prof.stirring_rate = [ExpParams.stir_r.value]*len(T_prof.t)    

        return T_prof

    def eq_gen(self,preceeding_sect, ExpParams, t_to_wait):   
        # ------------------------------
        # Add equillibrium time profile
        # ------------------------------
        Eq      = ProfileCreator() # create data class
        Eq_t    = [self.AVOID_ZERO_DIFF, t_to_wait]
        N       = len(Eq_t)

        self.helper_funcscs.segment_len_correcter(Eq,N) # correct unused profiles to have [0]*N length

        Eq.t    = Eq_t
        Eq.T    = [preceeding_sect.T[-1]] * 2
        Eq.stirring_rate = [ExpParams.stir_r.value]*len(Eq.t) 

        return Eq

    def simultanizer(self, T, AS):  
        # ---------------------------------------------------------------------------------
        # Brings the T and AS profiles on simultaneous 'terms' so it is executed parallelly
        # Returns profiles on new coordinates
        # ---------------------------------------------------------------------------------

        Sim_prof      = ProfileCreator() # create data class

        # Check if T or AS profile is longer
        t_deviation = self.helper_funcscs.cumulation(T.t)[-1] - self.helper_funcscs.cumulation(AS.t)[-1]

        # Correct the shorter (timewiswe) profiles 
        if t_deviation >= 0:
            self.helper_funcscs.length_modifier(AS,t_deviation)     
        else:
            self.helper_funcscs.length_modifier(T,t_deviation)    

        # Generate common t coordinates
        t_common, t_T, t_AS = self.helper_funcscs.t_commonizer(T,AS)

        # Interpolate profiles on new t points
        self.helper_funcscs.generate_interp_profs(Sim_prof,t_common, t_AS, AS) # resets all attributes (profiles) in Sim_prof on new coordinates (based on AS)

        Sim_prof.T = [int(x) for x in list(np.interp(t_common, t_T,  T.T)) ]      # Interp T

        # Generate the diff of the new t profile so that later the other codes can read it (uncumulated version is later cumulated)
        Sim_prof.t = [self.AVOID_ZERO_DIFF] + list(np.diff(t_common))

        return Sim_prof  


# In[197]:


class ProfileGenerator():
    # ========================================================================
    # Generate T, AS, seed profiles from generating informations
    # =======================================================================

    # ----------------------------------------------------------
    # __init__ -->
    # ---------------------------------------------------------- 

    def __init__(self, exp_params,exp_component_gen,helper_funcscs):
        self.description       = 'Using the class for creating simultaneous T, AS, and seeding profiles'
        self.exp_params        = exp_params
        self.exp_component_gen = exp_component_gen
        self.helper_funcscs    = helper_funcscs

    # ----------------------------------------------------------
    # III. Integrate the segments
    # ----------------------------------------------------------                  

    def generate_profiles(self):
        # --------------------------------------------------------------------------------------------------------------------------------------
        # Generate the T-, AS-, and Seeding profiles based on the predefined variables
        # --------------------------------------------------------------------------------------------------------------------------------------

        exp_params = self.exp_params

        # ---------------------------------------------------------------------------------
        # 1. Load the set parameters to generate profiles
        # --------------------------------------------------------------------------------- 

        # (-1): Initial equillibrium
        Initial_cond = ProfileCreator()
        Initial_cond.T = [exp_params.T_room]
        Eq_in = self.exp_component_gen.eq_gen(Initial_cond, exp_params,exp_params.time_to_wait_prior_exp)

        # (0): Initial section: add solution at room T
        initial_sol_add = self.exp_component_gen.sol_add_gen('solution',exp_params)

        # (1): Generate initial_T T segment (heat up, dissolve + holding time, cool to T0) + holding time
        initial_T = self.exp_component_gen.occasional_temp_prof_gen(exp_params.T_room,
                                                      exp_params.T_diss.value,
                                                      exp_params.T_0.value,
                                                      exp_params.t_diss.value, 
                                                      exp_params.t_h_0.value, 
                                                      exp_params,
                                                      initial_sol_add,
                                                      exp_params.if_diss_initial)

        # (2): Initial AS addition + holding time
        initial_AS = self.exp_component_gen.initial_antisolv_gen(initial_T, exp_params)    

        # (3): Add seed + holding time
        seed_prof = self.exp_component_gen.seed_prof_gen(initial_AS,exp_params)

        if exp_params.ARR == 'AS': # 'AS first' profile
            # (4): Add AS + holding time
            AS = self.exp_component_gen.antisolv_gen(seed_prof,exp_params)
            # (5): Cooling + holding time
            T_prof = self.exp_component_gen.temp_prof_gen(AS,exp_params)
            # (6): Final equillibrium time
            Eq = self.exp_component_gen.eq_gen(T_prof, exp_params, exp_params.t_eq.value)
            # (7): Define first and second 'actions'
            first_action = AS
            second_action = T_prof

        elif exp_params.ARR == 'T': # 'Cool first' profile
            # (4): Cooling + holding time
            T_prof = self.exp_component_gen.temp_prof_gen(seed_prof,exp_params)
            # (5): Add AS + holding time
            AS = self.exp_component_gen.antisolv_gen(T_prof,exp_params,)
            # (6): Final equillibrium time
            Eq = self.exp_component_gen.eq_gen(AS, exp_params, exp_params.t_eq.value)
            # (7): Define first and second 'actions'
            first_action = T_prof
            second_action = AS

        elif exp_params.ARR == 'SIM': # 'Simultaneous' sections
            # (4): Cooling + holding time
            T_prof = self.exp_component_gen.temp_prof_gen(seed_prof,exp_params)
            # (5): Add AS + holding time
            AS = self.exp_component_gen.antisolv_gen(seed_prof,exp_params)
            # (6): **Merge segments**
            Sim_prof = self.exp_component_gen.simultanizer(T_prof, AS)
            # (7): Final equillibrium time
            Eq = self.exp_component_gen.eq_gen(Sim_prof, exp_params, exp_params.t_eq.value)
            # (8): Define first and second 'actions'
            BLANK = ProfileCreator()
            first_action = Sim_prof
            second_action = BLANK

        else:
            raise ValueError(f"Unknown ARR mode '{exp_params.ARR}'. ""Expected one of: 'AS', 'T', 'SIM'." )    

        valve_open        = self.exp_component_gen.valve_prof_gen(exp_params)

        T_prof_back_to_RT = self.exp_component_gen.occasional_temp_prof_gen(Eq.T[-1],exp_params.T_room,
                                                                            exp_params.T_room,0.01,exp_params.t_before_open_valve,
                                                                            exp_params,Eq)             
        Eq_final          = self.exp_component_gen.eq_gen(T_prof_back_to_RT, exp_params, 0.5) 
        valve_open        = self.exp_component_gen.valve_prof_gen(exp_params)            
        cleaning_sol_add  = self.exp_component_gen.sol_add_gen('clean_sol',exp_params)   
        T_boil_out_clean  = self.exp_component_gen.occasional_temp_prof_gen(cleaning_sol_add.T[-1],
                                                                            exp_params.aceton_boil_out_T,
                                                                            exp_params.T_room,exp_params.t_boil.value,exp_params.t_h_boil.value,exp_params,Eq)             
        
        valve_open_1      = self.exp_component_gen.valve_prof_gen(exp_params)            
        solvent_rinse     = self.exp_component_gen.sol_add_gen('solvent',exp_params)    
        Eq_final_2        = self.exp_component_gen.eq_gen(solvent_rinse, exp_params, exp_params.wait_after_exp)     

        sections_to_integrate = [Eq_in,
                                 initial_sol_add,
                                 initial_T,
                                 initial_AS,
                                 seed_prof,
                                 first_action,
                                 second_action,
                                 Eq,
                                 #T_prof_back_to_RT,
                                 valve_open,
                                 cleaning_sol_add,
                                 T_boil_out_clean,
                                 valve_open,
                                 solvent_rinse,
                                 valve_open,
                                 Eq_final_2]

        integrated = self.helper_funcscs.integrator(sections_to_integrate, ProfileCreator()) # create data class)
        profile    = self.helper_funcscs.generate_pandas_profile(integrated)
        profile_pd = self.helper_funcscs.generate_pandas_profile_for_omni(integrated)
        
        return profile, profile_pd

