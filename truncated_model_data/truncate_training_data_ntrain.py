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

sys.path.append("/home/disk/p/mkb22/Documents/si_analysis_kb/LIMs/SI_LIMs/")
import LIM_utils as lim

import LIM_utils_kb as limkb
import LIM_stats_kb as statskb
import LIM_plot_kb as plotkb
import LIM_building as limbuild

import importlib
importlib.reload(limkb)
importlib.reload(statskb)
importlib.reload(limbuild)

ntrunc_list = [50] # EOF truncation for individual fields (reduced-state space)
nmodes_list = [50] # number of coupled EOFs for LIM state space (truncated-state space)
#nmodes = 25
nmodes_sic_list = [50]

mo='all'
#mo=7

# forecast lead time in months that defines the LIM training
tau = 1
    
lat_cutoff = 30

tscut = False     # time start cut 
#tecut = 16       # time end cut 
#tecut = 10
#tecut = 655       # 500 training years LME
#tecut = 355       # 800 training years LME
# tecut = 1001      # 154 training years LME

#tecut_list = [1055, 955, 855, 755, 655, 555, 455, 355]
tecut_list = [355]

# variables to include in the LIM (note "vars" is a Python command)
#limvars = ['tas','rlut','zg']
limvars = ['tas','tos','psl','zg','sit','sic']
#limvars = ['sic']
#limvars = ['tas','sic','zg','psl','pr','tos']

limvars_nosic = []
nvars = len(limvars)

# specify the model source 
#train_dsource = 'ccsm4_hist_kb'
#train_dsource = 'mpi_hist_kb'
#train_dsource = 'cmip6_cesm2_ssp585'
#train_dsource = 'cmip6_mpi_hist'
train_dsource = 'cesm_lme'
#train_dsource = 'cmip6_mpi_hist'
#train_dsource = 'mpi_lm_kb'
#train_dsource = 'era5'
#train_dsource = 'era5'
valid_dsource = 'mpi_lm_kb'
#valid_dsource = 'ccsm4_lm_kb'
#valid_dsource = 'ccsm4_lm_kb'

sic_separate = True
Insamp = False
dt=True
wt=True


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
elif 'lme' in train_dsource: 
    folder_add = 'last_millennium/'
    
pi = np.pi
    
fdic_train = limkb.build_training_dic(train_dsource)
fdic_valid = limkb.build_training_dic(valid_dsource)

full_names, areawt_name, month_names = limbuild.load_full_names()
areacell, areacell_dict = limbuild.load_areacell_dict(fdic_train, lat_cutoff=lat_cutoff,remove_climo=False, 
                                                      detrend=False, verbose=False )

today_date = '20211014'
var_dict = {}

for t,tecut in enumerate(tecut_list):
    print('Working on tecut = '+str(tecut))
    for k, var in enumerate(limvars): 
        X_var, var_dict = limkb.load_data(var, var_dict, fdic_train, remove_climo=True, 
                                          detrend=dt, verbose=True, cmip6=False, 
                                          tscut=tscut, tecut=tecut, lat_cutoff=lat_cutoff)

        if mo is 'all':
            print('Using month: '+str(mo))
            X_var_in = X_var
        else: 
            print('Using month: '+str(mo))
            tsamp = X_var.shape[1]
            x_var = np.reshape(X_var,(X_var.shape[0],int(tsamp/12),12))[:,:,mo:mo+2]
            X_var_in = np.reshape(x_var, (x_var.shape[0],x_var.shape[1]*x_var.shape[2]))

        if var is 'sic':
            if np.nanmax(X_var)>1:
                print('Changing units of sic be a between 0 to 1')
                X_var_in = X_var_in/100

        acell = areacell[areawt_name[var]]
        if (acell is 'None') and (len(var_dict[var]['lat'].shape) ==1): 
            acell_hold = np.ones((var_dict[var]['lat'].shape[0],var_dict[var]['lon'].shape[0]))
            acell_2d = np.cos(np.deg2rad(var_dict[var]['lat']))[:,np.newaxis]*acell_hold
            acell_1d = np.reshape(acell_2d,(X_var.shape[0]))
            skip_units=True
        elif len(acell.shape)>1:
            acell_1d = np.reshape(acell,(acell.shape[0]*acell.shape[1]))
            skip_units=False
        else: 
            acell_1d = acell
            skip_units=False

        if skip_units is False: 
            if 'km' in areacell_dict[areawt_name[var]][areawt_name[var]]['units']:
                acell_1d = acell_1d
            else: 
                print('changing cellarea units from '+
                      str(areacell_dict[areawt_name[var]][areawt_name[var]]['units'])+' to km^2')
                acell_1d = acell_1d/(1000*1000)

        for n,ntrunc in enumerate(ntrunc_list):
            nmodes = nmodes_list[n]
            nmodes_sic = nmodes_sic_list[n]    

            [Ptrunc, E3, tot_var,
             tot_var_eig, W_all, 
             standard_factor,
             var_expl_by_retained] = limkb.step1_compress_individual_var(X_var_in, var, ntrunc, nmodes_sic, 
                                                                         var_dict, areawt=acell_1d,
                                                                         wt=wt, sic_separate=True)

            mod_data_trunc = {}
            mod_data_trunc['var_dict'] = var_dict
            mod_data_trunc['Ptrunc'] = Ptrunc
            mod_data_trunc['E3'] = E3
            mod_data_trunc['standard_factor'] = standard_factor
            mod_data_trunc['W_all'] = W_all

            if 'datetime64' in str(type(var_dict[var]['time'][0])):
                start_yr = str(var_dict[var]['time'][0].astype('M8[Y]'))
                end_yr = str(var_dict[var]['time'][-1].astype('M8[Y]'))
            else: 
                start_yr = str(var_dict[var]['time'][0].year)
                end_yr = str(var_dict[var]['time'][-1].year)

    #        mod_folder = '/home/disk/p/mkb22/Documents/si_analysis_kb/LIMs/SI_LIMs/truncated_model_data/'
            mod_folder = ('/home/disk/kalman2/mkb22/SI_LIMs/truncated_data/'+folder_add)

            if var is 'sic':
                nmod = nmodes_sic
            else: 
                nmod = nmodes

            mod_filename = (var+'_ntrunc'+str(nmod)+'_002_month'+str(mo)+'_'+str(train_dsource)+'_latcutoff_'+
                            str(lat_cutoff)+'_wt'+str(wt)+'_dt'+str(dt)+'_ntrain_'+start_yr+'_'+end_yr+
                            '_'+today_date+'.pkl')

            print('saving in: '+mod_folder+mod_filename)
            pickle.dump(mod_data_trunc, open(mod_folder+mod_filename, "wb" ) )





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
