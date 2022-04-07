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

from datetime import date

today = date.today()

#Year-month-day
today_date = today.strftime("%Y%m%d")

import warnings
warnings.filterwarnings("ignore")

#=============================================================
# BEGIN USER INPUTS 
#=============================================================
# number of EOFs to retain for the LIM state vector
ntrunc = 400 # EOF truncation for individual fields (reduced-state space)
nmodes = 60 # number of coupled EOFs for LIM state space (truncated-state space)
#nmodes = 25
nmodes_sic = 50
#modes_sic = 20

#mo='all'
#mo=0
months = [0,1,2,3,4,5,6,7,8,9,10]

# forecast lead time in months that defines the LIM training
tau = 1

# training data defined by the first ntrain times
# fraction of years used in training
# ensures that start year of both train and validation data is january 
ntrain = 0.9

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
Insamp = True

#lags = [0,1,2,3,4,5,6,7,8,9,10,11]
lags = [0,1]

#=============================================================
# END USER INPUTS 
#=============================================================

infile_20cr_tas = '/home/disk/kalman3/rtardif/LMR/data/model/20cr/tas_sfc_Amon_20CR_185101-201112.nc'

fdic_train = limkb.build_training_dic(train_dsource)
fdic_valid = limkb.build_training_dic(valid_dsource)

full_names, areawt_name, month_names = limbuild.load_full_names()
areacell, areacell_dict = limbuild.load_areacell_dict(fdic_train, remove_climo=False, 
                                                      detrend=False, verbose=False)

#=============================================================
def run_forecast():
    print('Insamp is ' + str(Insamp))
    valid_stats_allmo = {}
    
    for mo in months: 
        print('========================')
        print('Working on MONTH: '+str(mo))
        print('========================')
        # Start with truncated training data: 
        mod_folder = 'truncated_model_data/'
        mod_filename = '_ntrunc400_mpi_lm_kb_20210406.pkl'
        mod_sic_filename = '_ntrunc50_mpi_lm_kb_20210406.pkl'

        [Ptrunc, Ptrunc_valid, E3, tot_var, 
         tot_var_eig, W_all, standard_factor, 
         nyears_train, var_dict] = limbuild.load_training_data_truncated(limvars, mod_folder, mod_sic_filename, 
                                                                         mod_filename, mo, ntrain)

        var_dict = limbuild.get_var_incices(limvars, var_dict)

        # Training data truncation: ==================================
        ndof_all = limkb.count_ndof_all(limvars, E3, sic_separate=sic_separate)

        [Ptrunc_all, E3_all, 
        Ptrunc_sic,E_sic] = limkb.stack_variable_eofs(limvars, ndof_all, ntrunc, Ptrunc, E3, 
                                                      var_dict, sic_separate=sic_separate)

        [P_train, Fvar, E] = limkb.step2_multivariate_compress(Ptrunc_all, nmodes, E3_all, Ptrunc_sic, 
                                                               sic_separate=sic_separate, Trunc_truth=False)

        # Validation data truncation: ==================================
        ndof_all_valid = limkb.count_ndof_all(limvars, E3, sic_separate=sic_separate)

        [Ptrunc_all_valid, E3_all_valid,
         Ptrunc_sic_valid, E_sic_valid] = limkb.stack_variable_eofs(limvars, ndof_all_valid, ntrunc, Ptrunc_valid,
                                                                    E3, var_dict, sic_separate=sic_separate)

        [P_train_valid, Fvar_valid, 
         E_valid] = limkb.step2_multivariate_compress(Ptrunc_all_valid,nmodes, E3_all_valid, Ptrunc_sic_valid,
                                                      sic_separate=sic_separate, Trunc_truth=False)                                                   
        # LIM training ================================================
        print('Training LIM...')
        nmo = int(P_train.shape[1]/nyears_train)
        P_train_3d = np.reshape(P_train, (P_train.shape[0],nyears_train,nmo))

        if mo is 'all':
            LIMd2, G2 = lim.LIM_train(tau,P_train)
            print('Training LIM with tau = '+str(tau)+', all months')
        else: 
            LIMd2, G2 = lim.LIM_train_flex(tau,P_train_3d[:,:,0], P_train_3d[:,:,1])
            print('Training LIM with tau = '+str(tau)+', month = '+str(mo))


        # Run Forecast ==================================================
        print('Running forecast...')
        ntims = len(lags)

        if mo == 'all':
            if Insamp==True: 
                LIM_fcast = limkb.LIM_forecast(LIMd2,P_train[:,0:nyears_train],lags)
            else: 
                LIM_fcast = limkb.LIM_forecast(LIMd2,P_train_valid,lags)
        else: 
            if Insamp==True: 
                P_train_2d = np.reshape(P_train, (P_train.shape[0],int(P_train.shape[1]/2),2))
                LIM_fcast = limkb.LIM_forecast(LIMd2,P_train_2d[:,:,0],lags)
            else: 
                LIM_fcast = limkb.LIM_forecast(LIMd2,P_train_valid,lags)


        # Decompress forecast ==================================================
        print('Decompressing forecast...')
        if len(lags)>1:
            x_forecast_dcomp = np.zeros((len(lags),E.shape[0]+E_sic.shape[0],
                                         LIM_fcast['x_forecast'].shape[2]))
            
            for i,lag in enumerate(lags):
                print('Lag '+ str(lag))
                x_forecast_dcomp[i,:,:] = limkb.decompress_eof_separate_sic(LIM_fcast['x_forecast'][i,:,:],
                                                                            nmodes,nmodes_sic,E,
                                                                            E_sic,limvars,var_dict,
                                                                            W_all,Weights=True,
                                                                            sic_separate=sic_separate)
        else: 
            x_forecast_dcomp = np.zeros((E.shape[0]+E_sic.shape[0],
                                         LIM_fcast['x_forecast'].shape[1]))

            print('Lag '+ str(lags))
            x_forecast_dcomp = limkb.decompress_eof_separate_sic(LIM_fcast['x_forecast'],
                                                                 nmodes,nmodes_sic,E,
                                                                 E_sic,limvars,var_dict,
                                                                 W_all,Weights=True,
                                                                 sic_separate=sic_separate)


        # Validate forecast ==================================================
        print('Validating forecast...')
        validvars = ['tas','tos','sic']
        v = {}
        validation_stats = {}
        gm_variance_mo = {}

        for k, var in enumerate(validvars):
            print(var)
            X_var, _ = limkb.load_data(var, v, fdic_train, remove_climo=True, detrend=True, verbose=False)
            var_3d = np.reshape(X_var,(X_var.shape[0],1000,12))
            x_anom_var = np.nanvar(var_3d,axis=1,ddof=1)
            gm_x_var = statskb.global_mean(x_anom_var,areacell[areawt_name[var]])

            gm_variance_mo[var] = gm_x_var

            corr_tot = np.zeros((len(lags)))
            ce_tot = np.zeros((len(lags)))
            gm_var_ratio = np.zeros((len(lags)))
            valid_stats = {}

            for i,lag in enumerate(lags):
                print('Lag '+str(lag))
                print(mo)
                [truth_anom, forecast_anom] = limbuild.gather_truth_forecast(lag,var,mo,X_var,x_forecast_dcomp,
                                                                             var_dict,ntrain,insamp=Insamp)
                print('Truth_anom shape: '+str(truth_anom.shape))
                print('Forecast_anom shape: '+str(forecast_anom.shape))

                [corr_tot[i], ce_tot[i], gm_var_ratio[i], tot_var_forecast, 
                 tot_var_truth] = limbuild.calc_validataion_stats(var, truth_anom, forecast_anom, var_dict,
                                                                  areacell,areacell_dict,
                                                                  areawt_name,month_names,iplot=True)

            valid_stats['corr_tot'] = corr_tot
            valid_stats['ce_tot'] = ce_tot
            valid_stats['gm_var_ratio'] = gm_var_ratio

            validation_stats[var] = valid_stats

        valid_stats_allmo[mo] = validation_stats   

    return valid_stats_allmo


#=============================================================
def run_save_forecast():
    print('Insamp is ' + str(Insamp))
    valid_stats_allmo = {}
    Fcast_out = {}
    
    for mo in months: 
        print('========================')
        print('Working on MONTH: '+str(mo))
        print('========================')
        fcast={}
        
        # Start with truncated training data: 
        mod_folder = 'truncated_model_data/'
        mod_filename = '_ntrunc400_mpi_lm_kb_20210406.pkl'
        mod_sic_filename = '_ntrunc50_mpi_lm_kb_20210406.pkl'

        [Ptrunc, Ptrunc_valid, E3, tot_var, 
         tot_var_eig, W_all, standard_factor, 
         nyears_train, var_dict] = limbuild.load_training_data_truncated(limvars, mod_folder, mod_sic_filename, 
                                                                         mod_filename, mo, ntrain)

        var_dict = limbuild.get_var_incices(limvars, var_dict)

        # Training data truncation: ==================================
        ndof_all = limkb.count_ndof_all(limvars, E3, sic_separate=sic_separate)

        [Ptrunc_all, E3_all, 
        Ptrunc_sic,E_sic] = limkb.stack_variable_eofs(limvars, ndof_all, ntrunc, Ptrunc, E3, 
                                                      var_dict, sic_separate=sic_separate)

        [P_train, Fvar, E] = limkb.step2_multivariate_compress(Ptrunc_all, nmodes, E3_all, Ptrunc_sic, 
                                                               sic_separate=sic_separate, Trunc_truth=False)

        # Validation data truncation: ==================================
        ndof_all_valid = limkb.count_ndof_all(limvars, E3, sic_separate=sic_separate)

        [Ptrunc_all_valid, E3_all_valid,
         Ptrunc_sic_valid, E_sic_valid] = limkb.stack_variable_eofs(limvars, ndof_all_valid, ntrunc, Ptrunc_valid,
                                                                    E3, var_dict, sic_separate=sic_separate)

        [P_train_valid, Fvar_valid, 
         E_valid] = limkb.step2_multivariate_compress(Ptrunc_all_valid,nmodes, E3_all_valid, Ptrunc_sic_valid,
                                                      sic_separate=sic_separate, Trunc_truth=False)                                                   
        # LIM training ================================================
        print('Training LIM...')
        nmo = int(P_train.shape[1]/nyears_train)
        P_train_3d = np.reshape(P_train, (P_train.shape[0],nyears_train,nmo))

        if mo is 'all':
            LIMd2, G2 = lim.LIM_train(tau,P_train)
            print('Training LIM with tau = '+str(tau)+', all months')
        else: 
            LIMd2, G2 = lim.LIM_train_flex(tau,P_train_3d[:,:,0], P_train_3d[:,:,1])
            print('Training LIM with tau = '+str(tau)+', month = '+str(mo))


        # Run Forecast ==================================================
        print('Running forecast...')
        ntims = len(lags)

        if mo == 'all':
            if Insamp==True: 
                LIM_fcast = limkb.LIM_forecast(LIMd2,P_train[:,0:nyears_train],lags)
            else: 
                LIM_fcast = limkb.LIM_forecast(LIMd2,P_train_valid,lags)
        else: 
            if Insamp==True: 
                P_train_2d = np.reshape(P_train, (P_train.shape[0],int(P_train.shape[1]/2),2))
                LIM_fcast = limkb.LIM_forecast(LIMd2,P_train_2d[:,:,0],lags)
            else: 
                LIM_fcast = limkb.LIM_forecast(LIMd2,P_train_valid,lags)


        # Decompress forecast ==================================================
        print('Decompressing forecast...')
        if len(lags)>1:
            x_forecast_dcomp = np.zeros((len(lags),E.shape[0]+E_sic.shape[0],
                                         LIM_fcast['x_forecast'].shape[2]))
            
            for i,lag in enumerate(lags):
                print('Lag '+ str(lag))
                x_forecast_dcomp[i,:,:] = limkb.decompress_eof_separate_sic(LIM_fcast['x_forecast'][i,:,:],
                                                                            nmodes,nmodes_sic,E,
                                                                            E_sic,limvars,var_dict,
                                                                            W_all,Weights=True,
                                                                            sic_separate=sic_separate)
        else: 
            x_forecast_dcomp = np.zeros((E.shape[0]+E_sic.shape[0],
                                         LIM_fcast['x_forecast'].shape[1]))

            print('Lag '+ str(lags))
            x_forecast_dcomp = limkb.decompress_eof_separate_sic(LIM_fcast['x_forecast'],
                                                                 nmodes,nmodes_sic,E,
                                                                 E_sic,limvars,var_dict,
                                                                 W_all,Weights=True,
                                                                 sic_separate=sic_separate)
            
    
        fcast['forecast'] = LIM_fcast
        fcast['forecast_dcomp'] = x_forecast_dcomp
        fcast['LIMd'] = LIMd2
        fcast['var_dict'] = var_dict
        
        Fcast_out[mo] = fcast
        
    return Fcast_out

            
def run_validation(Fcast_out): 
    print('Validating forecast...')
    validvars = ['tas','tos','sic']
    v = {}
    gm_variance_mo = {}
    valid_stats_allmo = {}
    
    for k, var in enumerate(validvars):
        print(var)
        X_var, _ = limkb.load_data(var, v, fdic_train, remove_climo=True, detrend=True, verbose=False)
        
        var_3d = np.reshape(X_var,(X_var.shape[0],1000,12))
        x_anom_var = np.nanvar(var_3d,axis=1,ddof=1)
        gm_x_var = statskb.global_mean(x_anom_var,areacell[areawt_name[var]])
        gm_variance_mo[var] = gm_x_var
        
        for mo in months:
            print('Working on month '+str(mo))
            validation_stats = {}
            x_forecast_dcomp = Fcast_out[mo]['forecast_dcomp']
            var_dict = Fcast_out[mo]['var_dict']

            corr_tot = np.zeros((len(lags)))
            ce_tot = np.zeros((len(lags)))
            gm_var_ratio = np.zeros((len(lags)))
            valid_stats = {}

            for i,lag in enumerate(lags):
                print('Lag '+str(lag))
                print(mo)
                [truth_anom, forecast_anom] = limbuild.gather_truth_forecast(lag,var,mo,X_var,x_forecast_dcomp,
                                                                             var_dict,ntrain,insamp=Insamp)
                print('Truth_anom shape: '+str(truth_anom.shape))
                print('Forecast_anom shape: '+str(forecast_anom.shape))

                [corr_tot[i], ce_tot[i], gm_var_ratio[i], tot_var_forecast, 
                 tot_var_truth] = limbuild.calc_validataion_stats(var, truth_anom, forecast_anom, var_dict,
                                                                  areacell,areacell_dict,
                                                                  areawt_name,month_names,iplot=True)

            valid_stats['corr_tot'] = corr_tot
            valid_stats['ce_tot'] = ce_tot
            valid_stats['gm_var_ratio'] = gm_var_ratio

            validation_stats[mo] = valid_stats

        valid_stats_allmo[var] = validation_stats   

    return valid_stats_allmo
