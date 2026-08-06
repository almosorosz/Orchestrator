from dataclasses        import fields
from itertools          import accumulate, chain

import numpy  as np
import pandas as pd

def set_stirring(A,ExpParams):
    
    A.stirring_rate      = [ExpParams.stirring_rate.value]*len(A.t)    
    A.stirring_rate_seed = [ExpParams.stirring_rate_seed.value]*len(A.t)    
    A.stirring_rate_sol  = [ExpParams.stirring_rate_sol.value]*len(A.t) 
    
    return A
def initialize(A): # Helper function to set each new ProfileCreator() components to [0] at beginning
    for f in fields(A):
        setattr(A, f.name, [0])
    setattr(A,'t',[1e-3])

def init_temp_segment(A,T): # Helper function to set each new ProfileCreator() components to [0] * len(T) in occasional_temp_prof_gen()
    for f in fields(A):
        setattr(A, f.name, [0] * len(T))
    setattr(A,'t',[1e-3]* len(T))

def segment_len_correcter(A,n): # Helper function to each unused ProfileCreator() components to [0] * n
    for f in fields(A):
        setattr(A, f.name, [0] * n)
    setattr(A,'t',[1e-3] * n)

def cumulation(A):
    return list(accumulate(A))

def integrator(sections_to_integrate,integrated): 
    # ---------------------------------------------------------------------------
    # Integrate sections to form a joint profile
    # ---------------------------------------------------------------------------

    for f in fields(integrated):
        list_to_set = list(chain.from_iterable([getattr(s, f.name) for s in sections_to_integrate]))
        setattr(integrated, f.name, list_to_set)

    A = sections_to_integrate[0].t
    for s in sections_to_integrate[1:]:
        A += s.t

    integrated.t    = cumulation(A)

    return  integrated

def generate_pandas_profile_for_omni(integrated):
    data = dict()

    for f in fields(integrated):
        if f.name != 't':
            cur_t_var_name = f't_{f.name}'
            data[cur_t_var_name] = np.array(getattr(integrated, 't'))*60 # in minutes
            data[f.name] = np.array(getattr(integrated, f.name))

    return pd.DataFrame(data)

def generate_pandas_profile(integrated):
    data = dict()
    for f in fields(integrated):
        data[f.name] = np.array(getattr(integrated, f.name))
    return pd.DataFrame(data)   

def length_modifier(A,t_deviation):
    # ---------------------------------------------------------------------------------
    # Modifies the profile, by extending the profile with the last time point of the longer profile so that the T or A is as long as the other
    # returns the extended profile
    # ---------------------------------------------------------------------------------

    for f in fields(A):
        field_value = getattr(A, f.name)
        setattr(A, f.name, field_value + [field_value[-1]])
    A.t    = A.t[0:-1] + [abs(t_deviation)] # different correction for t

def t_commonizer(T,AS):
    # ---------------------------------------------------------------------------------
    # Generates common t points
    # Returns the t points
    # ---------------------------------------------------------------------------------
    t_T  = np.array(cumulation(T.t))
    t_AS = np.array(cumulation(AS.t))
    return np.unique(np.concatenate((t_T, t_AS))),t_T,t_AS

def generate_interp_profs(A,t_common,t_AS,AS):
    # ---------------------------------------------------------------------------------
    # Interpolate all profiles on the new t coordinates
    # Modifies the new profiles on the coordinates
    # ---------------------------------------------------------------------------------

    for f in fields(A):
        field_value = getattr(AS, f.name)
        setattr(A, f.name, [int(x) for x in list(np.interp(t_common, t_AS, field_value))] ) 

def turn_off(var):
    # ------------------------------------------------------------------------------------
    # set var.value to 0 and var.if_in_doe to False
    # ------------------------------------------------------------------------------------

    setattr(var,'value', 0)
    setattr(var,'if_in_doe',False)

def turn_off_by_group(condition,list_of_vars):
    # ------------------------------------------------------------------------------------
    # apply turn_of_by_group() to all elements of list_of_vars
    # ------------------------------------------------------------------------------------

    if not condition:
        for i in list_of_vars:
            turn_off(i)

def exp_par_condition_corr(exp_par):
    # ------------------------------------------------------------------------------------
    # Turn off operating conditions based on user-defined settings to make experiment profile logical
    # ------------------------------------------------------------------------------------

    conditions = [
        (exp_par.if_opening,     [exp_par.t_h_open, exp_par.be_opened_until]),
        (exp_par.if_cleaning,    [exp_par.clean_sol, exp_par.t_h_clean]),
        (exp_par.if_solv_rinse,  [exp_par.solvent, exp_par.t_h_solv]),
        (exp_par.if_sol_initial, [exp_par.solution, exp_par.t_h_solution]),
        (exp_par.if_AS_initial,  [exp_par.AS_initial, exp_par.t_h_as_initial]),
        (exp_par.if_seed,        [exp_par.seed, exp_par.t_h_seed]),
        (exp_par.if_AS,          [exp_par.AS_am, exp_par.AS_add_time, exp_par.t_h_AS]),
        (exp_par.if_cool_in_exp, [exp_par.t_cool, exp_par.T_final, exp_par.t_h_cool, exp_par.t_eq]),
        (exp_par.if_boil,        [exp_par.t_boil, exp_par.t_h_boil])
    ]

    for condition, vars_ in conditions:
        turn_off_by_group(condition, vars_)

    if not exp_par.if_AS:
        exp_par.if_ARR = False
        exp_par.ARR    = 'T'

    if not exp_par.if_cool_in_exp:
        exp_par.if_ARR = False
        exp_par.ARR    = 'AS'

    return exp_par

