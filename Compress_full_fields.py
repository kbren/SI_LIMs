import sys,os
import xarray as xr
import numpy as np
import scipy as spy
import pickle 

from collections import OrderedDict 

import time as timestamp 

sys.path.append("/home/disk/p/mkb22/Documents/si_analysis_kb/LIMs/SI_LIMs/")
import LIM_utils as lim

import LIM_utils_kb as limkb
import LIM_stats_kb as statskb

#================================================================================

# number of EOFs to retain for the LIM state vector
ntrunc = 30 # EOF truncation for individual fields (reduced-state space)
nmodes = 30 # number of coupled EOFs for LIM state space (truncated-state space)
#nmodes = 25
nmodes_sic = 50
#modes_sic = 20

mo='all'

# forecast lead time in months that defines the LIM training
tau = 1

# training data defined by the first ntrain times
# fraction of years used in training
# ensures that start year of both train and validation data is january 
ntrain = 1
ntrain_str = '1'

# variables to include in the LIM (note "vars" is a Python command)
#limvars = ['tas','zg']
#limvars = ['tas','rlut','zg']
limvars = ['tas','tos']
#limvars = ['tas','sic']
#limvars = ['tas','sic','zg','psl','pr','tos']
#limvars = ['tas','tos']
# limvars = ['tas','tos','psl','sit','sic']
# limvars_nosic = ['tas','tos','psl','sit']
#limvars = ['sit','psl']
nvars = len(limvars)

# specify the model source 
train_dsource = 'mpi_hist_kb'
# train_dsource = 'ccsm4_lm_kb'
valid_dsource = 'mpi_hist_kb'
# valid_dsource = 'ccsm4_lm_kb'

sic_separate = True

#================================================================================

full_names = {'tas':'Surface air temperature',
              'psl':'Sea level Pressure',
              'sic':'Sea ice concentration', 
              'sit':'Sea ice thickness',
              'tos':'Sea surface temperature',
              'zg': '500hPa geopotential height'}

areawt_name = {'tas':'areacella',
               'psl':'areacella',
               'sic':'areacello', 
               'sit':'areacello',
               'tos':'areacello',
               'zg': 'areacella'}

month_names = ['January','Februrary','March','April','May','June','July','August',
               'September','October','November','December']
               
               
from datetime import date

today = date.today()

#Year-month-day
today_date = today.strftime("%Y%m%d")

import warnings
warnings.filterwarnings("ignore")

fdic_train = limkb.build_training_dic(train_dsource)

areacell = {}
areacella_dict = {}
areacell['areacella'], areacella_dict = limkb.load_data('areacella', areacella_dict, fdic_train, 
                                                  remove_climo=False, detrend=False, verbose=False)

areacello_dict = {}
areacell['areacello'], areacello_dict = limkb.load_data('areacello', areacello_dict, fdic_train, 
                                                  remove_climo=False, detrend=False, verbose=False)

areacell_dict = {}
areacell_dict['areacello'] = areacello_dict
areacell_dict['areacella'] = areacella_dict

#================================================================================
# load training data...
wt=True
var_dict = {}

for k, var in enumerate(limvars): 
    X_var, var_dict = limkb.load_data(var, var_dict, fdic_train, remove_climo=True, 
                                      detrend=True, verbose=True)
    
#     nytrain = int(X_var.shape[1]*ntrain)
#     X_train = X_var[:,0:nytrain]
    X_train = X_var
    
    
    acell = areacell[areawt_name[var]]
    if len(acell.shape)>1:
        acell_1d = np.reshape(acell,(acell.shape[0]*acell.shape[1]))
    else: 
        acell_1d = acell
     
    [Ptrunc, E3, tot_var,
     tot_var_eig, W_all, 
     standard_factor] = limkb.step1_compress_individual_var(X_train, var, ntrunc, nmodes_sic, 
                                                            var_dict, areawt=acell_1d,
                                                            wt=wt, sic_separate=sic_separate)

    var_save = {}
    var_save['var_dict'] = var_dict
    var_save['Ptrunc'] = Ptrunc
    var_save['E3'] = E3
    var_save['standard_factor'] = standard_factor
    var_save['W_all'] = W_all
    
    if var =='sic':
        savename = ('./truncated_model_data/'+str(var)+'_ntrunc'+str(nmodes_sic)+'_'+valid_dsource+'_'+today_date+
                    '_ntrain'+ntrain_str+'_standtest.pkl')
    else: 
        savename = ('./truncated_model_data/'+str(var)+'_ntrunc'+str(ntrunc)+'_'+valid_dsource+'_'+today_date+
                    '_ntrain'+ntrain_str+'_standtest.pkl')
    pickle.dump(var_save, open(savename, "wb" ) )
        
    del X_var
