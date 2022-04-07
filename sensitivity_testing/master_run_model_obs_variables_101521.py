import sys,os
import xarray as xr
import numpy as np
import scipy as spy
import pickle 

import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from collections import OrderedDict 

import time as timestamp 
import importlib

sys.path.append("/home/disk/p/mkb22/Documents/si_analysis_kb/LIMs/SI_LIMs/")
import LIM_utils as lim

import LIM_utils_kb as limkb
import LIM_stats_kb as statskb
import LIM_plot_kb as plotkb
import LIM_building as limbuild

sys.path.append("/home/disk/kalman2/mkb22/pyLMR/")
import LMR_utils

import run_forecast_model_data as rf

from datetime import date

today = date.today()

#Year-month-day
today_date = today.strftime("%Y%m%d")

#--------------------------------------------------
# START USER PARAMETERS
#--------------------------------------------------
# number of EOFs to retain for the LIM state vector
ntrunc = 40 # EOF truncation for individual fields (reduced-state space)
#nmodes = 15 # number of coupled EOFs for LIM state space (truncated-state space)
#nmodes = 20
nmodes_sic = 30
#modes_sic = 20

mo='all'
#mo=0

# forecast lead time in months that defines the LIM training
tau = 1

# variables to include in the LIM (note "vars" is a Python command)
limvars_list = [['sic'],
                ['tas','sic'],
                ['sit','sic'],
                ['tos','sic'],
                ['psl','sic'],
                ['tas','sit','sic'],
                ['tas','tos','sic'],
                ['tas','tos','sit','sic'],
                ['tas','tos','psl','sit','sic']]
#limvars_list = [['tos','sic']]
#limvars = ['sic']
limvars_nosic = []
# nvars = len(limvars)

# specify the model source 
train_dsource = 'cmip6_mpi_hist'
#train_dsource = 'cmip6_mpi_hist_ssp585'
#train_dsource ='satellite'
#train_dsource = 'mpi_hist_kb'
#train_dsource = 'ccsm4_lm_kb'
#train_dsource = 'satellite'
#train_dsource = 'era5'
#valid_dsource = 'era5'
#valid_dsource = 'satellite'
#valid_dsource = 'mpi_lm_kb'
#valid_dsource = 'mpi_lm_kb'
valid_dsource = 'cmip6_mpi_hist'

sic_separate = True
Insamp = False

exp_setup = {}
exp_setup['lat_cutoff'] = 50
exp_setup['Weight']=True
exp_setup['remove_climo'] = True
exp_setup['detrend'] = True
exp_setup['nyr_train'] = None

lags = [0,1,2,3,4,5,6,7,8]

# era5 settings (out of sample): 
# exp_setup['nyearsvalid'] = 16
# exp_setup['nyearstot'] = 42
# exp_setup['nyears_startvalid'] = 26*12

# era5 settings (in sample):  
# exp_setup['nyearsvalid'] = 26
# exp_setup['nyearstot'] = 42
# exp_setup['nyears_startvalid'] = 1*12

# Satellite settings (out of sample): 
# exp_setup['nyearsvalid'] = 12
# exp_setup['nyearstot'] = 38
# exp_setup['nyears_startvalid'] = 26*12

# Satellite settings (in sample): 
exp_setup['nyearsvalid'] = 25
exp_setup['nyearstot'] = 38
exp_setup['nyears_startvalid'] = 1*12

# Historical settings (out of sample): 
# exp_setup['nyearsvalid'] = 10
# exp_setup['nyearstot'] = 164
# exp_setup['nyears_startvalid'] = 155*12

# Historical settings (in sample): 
# exp_setup['nyearsvalid'] = 10
# exp_setup['nyearstot'] = 164
# exp_setup['nyears_startvalid'] = 1*12

# LM settings
# exp_setup['nyearsvalid'] = 10
# exp_setup['nyearstot'] = 1000
# exp_setup['nyears_startvalid'] = 900*12

# date_of_interest = '20210910'

master_save = False

#--------------------------------------------------
# END USER PARAMETERS
#--------------------------------------------------

if 'hist_ssp585' in train_dsource: 
    folder_add = 'hist_ssp585_concatenated/'
elif 'hist' in train_dsource: 
    folder_add = 'historical/'
elif 'lm' in train_dsource: 
    folder_add = 'last_millennium/'
elif 'satellite' in train_dsource: 
    folder_add = 'satellite/'
elif 'era5' in train_dsource: 
    folder_add = 'reanalysis/'

mod_folder = '/home/disk/p/mkb22/Documents/si_analysis_kb/LIMs/SI_LIMs/truncated_model_data/'+folder_add
save_folder = '/home/disk/p/mkb22/Documents/si_analysis_kb/LIMs/SI_LIMs/sensitivity_testing/variables/'+folder_add

exp_setup['mo'] = mo
if 'all' in str(mo): 
    exp_setup['ind_month_trunc'] = False
else: 
    exp_setup['ind_month_trunc'] = True
exp_setup['tau'] = tau
exp_setup['train_dsource'] = train_dsource
exp_setup['valid_dsource'] = valid_dsource 
exp_setup['sic_separate'] = sic_separate
exp_setup['Insamp'] = Insamp
exp_setup['mod_folder'] = mod_folder
exp_setup['lags'] = lags
exp_setup['Insamp'] = Insamp
exp_setup['step2_trunc'] = False

exp_setup['ntrunc'] = ntrunc 
exp_setup['nmodes_sic'] = nmodes_sic

month_names = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec']

f_folder = save_folder
ar1f_folder = save_folder

mod_filename = ('_ntrunc'+str(exp_setup['ntrunc'])+'_monthall_'+train_dsource+'_latcutoff_'+
                str(exp_setup['lat_cutoff'])+'_wtTrue_dtTrue_ntrain_1850_2004_20211014.pkl')

mod_sic_filename = ('_ntrunc'+str(exp_setup['nmodes_sic'])+'_monthall_'+train_dsource+'_latcutoff_'+
                    str(exp_setup['lat_cutoff'])+'_wtTrue_dtTrue_ntrain_1850_2004_20211014.pkl')

exp_setup['mod_filename'] = mod_filename
exp_setup['mod_sic_filename'] = mod_sic_filename

for l,limvars in enumerate(limvars_list):
    
    exp_setup['limvars'] = limvars
    
    #--------------------------------------------------
    ### Build L from truncated data: 
    #--------------------------------------------------

    LIMd = rf.build_L(exp_setup, save_folder, save=master_save)
    
    if LIMd['npos_eigenvalues'] >0: 
        adj = True
    else: 
        adj = False
    exp_setup['adj'] = adj

    #--------------------------------------------------
    ### Run Forecast: 
    #--------------------------------------------------

    forecast = rf.run_forecast(LIMd,exp_setup, f_folder, verbose=True, save=master_save, save_decomp=False)

    forecast_validation = rf.validate_forecast_monthly(forecast, exp_setup['limvars'], 1, exp_setup, LIMd, f_folder, 
                                                       iplot=False, save=master_save)

    forecast_validation_lags = rf.validate_forecast_lagged(forecast, exp_setup['limvars'], exp_setup, LIMd, 
                                                           f_folder, iplot=False, save=master_save, 
                                                           detrend_truth=True)

    #--------------------------------------------------
    ### Run AR1 Forecast: 
    #--------------------------------------------------

    valid_vars=limvars

    ar1cast = rf.ar1_forecast_valid_by_month(LIMd['P_train'], forecast['P_train_valid'], LIMd,
                                             exp_setup, valid_vars, month_names, ar1f_folder, forecast,
                                             lag=None, iplot=False, save=master_save, save_decomp=False)

    ar1cast_lags = rf.ar1_forecast_valid_by_lag(LIMd['P_train'], forecast['P_train_valid'], LIMd, exp_setup, 
                                                exp_setup['limvars'], month_names, ar1f_folder, forecast,
                                                iplot=False, save=master_save, save_decomp=False,
                                                detrend_truth=False)
    
    #--------------------------------------------------
    ### Save experiment: 
    #--------------------------------------------------
    experiment = {}
    experiment['LIMd'] = LIMd
    experiment['forecast'] = forecast
    experiment['forecast_validation'] = forecast_validation
    experiment['forecast_validation_lags'] = forecast_validation_lags
    experiment['ar1cast'] = ar1cast
    experiment['ar1cast_lags'] = ar1cast_lags
    
    start_yr = str(forecast['var_dict_valid']['sic']['time'][0])[0:4]
    end_yr = str(forecast['var_dict_valid']['sic']['time'][-1])[0:4]

    exp_filename = ('Validation_'+exp_setup['train_dsource']+'_ntrain_'+exp_setup['mod_filename'][-22:-13]+
                    '_validyrs_'+start_yr+'_'+end_yr+'_'+
                    (str(exp_setup['ntrunc'])+"_").join(exp_setup['limvars'])+
                    str(exp_setup['nmodes_sic'])+'_'+today_date+'.pkl')

    print('saving in: '+save_folder+exp_filename)
    pickle.dump(experiment, open(save_folder+exp_filename, "wb" ) )
