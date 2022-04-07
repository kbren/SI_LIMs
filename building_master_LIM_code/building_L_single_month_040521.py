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

sys.path.append("/home/disk/kalman2/mkb22/pyLMR/")
import LMR_utils

import importlib
importlib.reload(limkb)
importlib.reload(statskb)

arc_proj = dict(projection=ccrs.Stereographic(central_latitude=90,
                                              central_longitude=-45,
                                              true_scale_latitude=0.1))
proj = dict(projection=ccrs.Robinson(central_longitude=0.),zorder=1)

months=[0,1,2,3,4,5,6,7,8,9,10]

# number of EOFs to retain for the LIM state vector
ntrunc = 400 # EOF truncation for individual fields (reduced-state space)
nmodes = 60 # number of coupled EOFs for LIM state space (truncated-state space)
#nmodes = 25
nmodes_sic = 50
#modes_sic = 20

# forecast lead time in months that defines the LIM training
tau = 1

# training data defined by the first ntrain times
# fraction of years used in training
# ensures that start year of both train and validation data is january 
ntrain = 0.6

# variables to include in the LIM (note "vars" is a Python command)
#limvars = ['tas','zg']
#limvars = ['tas','rlut','zg']
#limvars = ['sic']
#limvars = ['tas','sic']
#limvars = ['tas','sic','zg','psl','pr','tos']
#limvars = ['tas','psl','tos','sit','sic']
# limvars = ['tas','tos','psl','sit','sic']
# limvars_nosic = ['tas','tos','psl','sit']
limvars = ['tas','tos','sic']
limvars_nosic = ['tas','tos']

nvars = len(limvars)

# specify the model source 
train_dsource = 'mpi_lm_kb'
#train_dsource = 'ccsm4_lm_kb'
valid_dsource = 'mpi_lm_kb'
#valid_dsource = 'ccsm4_lm_kb'

sic_separate = True

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


import warnings
warnings.filterwarnings("ignore")

from datetime import date

today = date.today()

#Year-month-day
today_date = today.strftime("%Y%m%d")


# fill continents if plotting SST; otherwise no
# if var_to_extract == 'tos':
#     noland = True
# else:
#     noland = False

infile_20cr_tas = '/home/disk/kalman3/rtardif/LMR/data/model/20cr/tas_sfc_Amon_20CR_185101-201112.nc'

fdic_ccsm4 = limkb.build_training_dic(train_dsource)
fdic_mpi = limkb.build_training_dic(valid_dsource)

areacell = {}
areacella_dict = {}
areacell['areacella'], areacella_dict = limkb.load_data('areacella', areacella_dict, fdic_ccsm4, 
                                                  remove_climo=False, detrend=False, verbose=False)

areacello_dict = {}
areacell['areacello'], areacello_dict = limkb.load_data('areacello', areacello_dict, fdic_ccsm4, 
                                                  remove_climo=False, detrend=False, verbose=False)

areacell_dict = {}
areacell_dict['areacello'] = areacello_dict
areacell_dict['areacella'] = areacella_dict



for mo in months:
    # load training data...
    wt=True
    var_dict = {}
    tot_var = {}
    tot_var_eig = {}
    W_all = {}
    E3 = {}
    Ptrunc = {}
    standard_factor = {}

    tot_var_valid = {}
    tot_var_eig_valid = {}
    W_all_valid = {}
    E3_valid = {}
    Ptrunc_valid = {}
    standard_factor_valid = {}

    n=0

    for k, var in enumerate(limvars): 
        X_var, var_dict = limkb.load_data(var, var_dict, fdic_ccsm4, remove_climo=True, 
                                          detrend=True, verbose=True)

        tsamp = X_var.shape[1]
        nyears_train = int((tsamp*ntrain)/12)
        #nyears_valid = int(X_all_mpi.shape[2]/12)
        nyears_valid = int((tsamp*(1-ntrain))/12)

        X_t = np.reshape(X_var,(X_var.shape[0],int(tsamp/12),12))
        X_train = X_t[:,0:nyears_train,mo:mo+2]
        X_train_2d = np.reshape(X_train,(X_train.shape[0],nyears_train*2))
    #    X_valid = X_t[:,nyears_train:,mo]
        X_valid = X_t[:,nyears_train:,mo]
    #    truth = X_t[:,nyears_train:,mo+1]
        ntime = X_train.shape[1]

        acell = areacell[areawt_name[var]]
        if len(acell.shape)>1:
            acell_1d = np.reshape(acell,(acell.shape[0]*acell.shape[1]))
        else: 
            acell_1d = acell

        [Ptrunc[var], E3[var], tot_var[var],
         tot_var_eig[var], W_all[var], 
         standard_factor[var]] = limkb.step1_compress_individual_var(X_train_2d, var, ntrunc, nmodes_sic, 
                                                                     var_dict, n, areawt=acell_1d,
                                                                     wt=wt, sic_separate=sic_separate)

        [Ptrunc_valid[var], E3_valid[var], tot_var_valid[var],
         tot_var_eig_valid[var],W_all_valid[var],
         standard_factor_valid[var]] = limkb.step1_compress_individual_var(X_valid, var,ntrunc, nmodes_sic, 
                                                                           var_dict, n, areawt=acell_1d,wt=wt, 
                                                                           sic_separate=sic_separate)
    #     for m in range(12):
    #         X_valid = X_t[:,nyears_train:,m]

    #         [Ptrunc_valid, E3_valid, tot_var_valid,
    #         tot_var_eig_valid, W_all_valid] = step1_compress_individual_var(X_valid, ntrunc, nmodes_sic, var_dict, n, 
    #                                                                         areawt=areacell[areawt_name[var]],
    #                                                                         wt=wt, sic_separate=sic_separate)
    #         Ptrunc_valid_var[:,:,m] = Ptrunc_valid[var]
    #         E3_valid_var[:,:,m] = E3_valid[var]

        del X_var


    # Get indices for each variable:
    start = 0
    for k, var in enumerate(limvars): 
        print('working on '+var)
        inds = var_dict[var]['var_ndof']
        var_inds = np.arange(start,start+inds,1)
        start = inds+start

        var_dict[var]['var_inds'] = var_inds
    
    ### Truncate the training data: 
    ndof_all = limkb.count_ndof_all(limvars, E3, sic_separate=sic_separate)

    [Ptrunc_all, E3_all, 
    Ptrunc_sic, E_sic] = limkb.stack_variable_eofs(limvars, ndof_all, ntrunc, Ptrunc, E3, 
                                                   var_dict, sic_separate=sic_separate)

    [P_train, Fvar, E] = limkb.step2_multivariate_compress(Ptrunc_all,nmodes, E3_all, Ptrunc_sic, 
                                                           sic_separate=sic_separate)
    
    ### Truncate the validation data:                                                      
    ndof_all_valid = limkb.count_ndof_all(limvars, E3_valid, sic_separate=sic_separate)
 
    [Ptrunc_all_valid, E3_all_valid,
     Ptrunc_sic_valid, E_sic_valid] = limkb.stack_variable_eofs(limvars, ndof_all_valid, ntrunc, Ptrunc_valid, E3_valid, 
                                                                var_dict, sic_separate=sic_separate)

    [P_train_valid, Fvar_valid, 
     E_valid] = limkb.step2_multivariate_compress(Ptrunc_all_valid,nmodes, E3_all_valid, Ptrunc_sic_valid,
                                                  sic_separate=sic_separate)
    
    ### Truncate the validation data onto training eofs: 
    ndof_all_valid = limkb.count_ndof_all(limvars, E3_valid, sic_separate=sic_separate)

    [Ptrunc_all_valid2, E3_all_valid2,
     Ptrunc_sic_valid2, E_sic_valid2] = limkb.stack_variable_eofs(limvars, ndof_all_valid, ntrunc, Ptrunc, E3, 
                                                                var_dict, sic_separate=sic_separate)

    [P_train_valid2, Fvar_valid2, 
     E_valid2] = limkb.step2_multivariate_compress(Ptrunc_all,nmodes, E3_all, Ptrunc_sic,
                                                  sic_separate=sic_separate)



    nmo = int(P_train.shape[1]/nyears_train)
    P_train_3d = np.reshape(P_train, (P_train.shape[0],nyears_train,nmo))

    LIMd2, G2 = lim.LIM_train_flex(tau,P_train_3d[:,:,0], P_train_3d[:,:,1])
    print('Training LIM with tau = '+str(tau))

    LIM_save = {}
    LIM_save['LIMd2'] = LIMd2
    LIM_save['var_dict'] = var_dict
    LIM_save['P_train_valid'] = P_train_valid
    LIM_save['P_train_valid2'] = P_train_valid2
    LIM_save['P_train'] = P_train
    LIM_save['E'] = E
    LIM_save['E_sic'] = E_sic
    LIM_save['E_valid'] = E_valid
    LIM_save['E_valid2'] = E_valid2
    LIM_save['E_sic_valid'] = E_sic_valid
    LIM_save['E_sic_valid2'] = E_sic_valid2
    LIM_save['W_all'] = W_all
    LIM_save['W_all_valid'] = W_all_valid
    LIM_save['E3'] = E3
    LIM_save['standard_factor']=standard_factor

    var_nms = [l+'_' for l in limvars]
    savename = ('L_mo'+str(mo)+'_'+ ''.join(var_nms)+ 'ntrunc'+str(ntrunc)+'_nmodes'+str(nmodes)+
                '_nmodessic'+str(nmodes_sic)+'_ntrain'+str(nyears_train)+'_'+str(today_date)+'.pkl')
    
    print('Saving L in: '+savename)
    pickle.dump(LIM_save, open(savename, "wb" ) )

