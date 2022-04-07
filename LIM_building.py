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

sys.path.append("/home/disk/kalman2/mkb22/pyLMR/")
import LMR_utils

sys.path.append("/home/disk/p/mkb22/Documents/si_analysis_kb/LIMs/SI_LIMs/")
import LIM_utils as lim
import LIM_utils_kb as limkb
import LIM_stats_kb as statskb
import LIM_plot_kb as plotkb


def load_full_names(dsource):
    full_varnames = {'tas':'Surface air temperature',
                      'psl':'Sea level Pressure',
                      'sic':'Sea ice concentration', 
                      'sit':'Sea ice thickness',
                      'tos':'Sea surface temperature',
                      'zg': '500hPa geopotential height'}

    if 'Amon' in dsource: 
        areawt_name = {'tas':'areacella',
                       'psl':'areacella',
                       'sic':'areacella', 
                       'sit':'areacella',
                       'tos':'areacella',
                       'zg': 'areacella'}
    else: 
        areawt_name = {'tas':'areacella',
                       'psl':'areacella',
                       'sic':'areacello', 
                       'sit':'areacello',
                       'tos':'areacello',
                       'zg': 'areacella'}

    month_names = ['January','Februrary','March','April','May','June','July','August',
                   'September','October','November','December']
    
    return full_varnames, areawt_name, month_names


def load_areacell_dict(fdic_ccsm4, lat_cutoff=False, remove_climo=False, 
                       detrend=False, verbose=False):
    areacell = {}
    areacella_dict = {}
    
    if fdic_ccsm4['areacella'] is None: 
        areacell['areacella'] = 'None'
    else: 
        areacell['areacella'], areacella_dict = limkb.load_data('areacella', areacella_dict, fdic_ccsm4, 
                                                                remove_climo=False, detrend=False, verbose=False, 
                                                                lat_cutoff=lat_cutoff)

    areacello_dict = {}
    
    if fdic_ccsm4['areacello'] is None:
        areacell['areacello'] = 'None'
    else: 
        areacell['areacello'], areacello_dict = limkb.load_data('areacello', areacello_dict, fdic_ccsm4, 
                                                                remove_climo=False, detrend=False, verbose=False,
                                                                lat_cutoff=lat_cutoff)

    areacell_dict = {}
    areacell_dict['areacello'] = areacello_dict
    areacell_dict['areacella'] = areacella_dict
    
    return areacell, areacell_dict


def load_training_valid_data_full(limvars, fdic_train, mo, areacell, ntrain, areawt_name, 
                                  ntrunc,nmodes_sic, sic_separate=False,
                                  remove_climo=True, detrend=True, wt=True, verbose=True): 

    var_dict = {}
    v = {}
    tot_var = {}
    tot_var_eig = {}
    W_all = {}
    E3 = {}
    Ptrunc = {}
    Ptrunc_valid = {}
    standard_factor = {}

    for k, var in enumerate(limvars): 

        X_var, v = limkb.load_data(var, var_dict, fdic_train, remove_climo=True, 
                                   detrend=True, verbose=True)
        var_dict[var] = v[var]
        
        if mo is 'all':
            X_t = X_var
            ntime = X_t.shape[1]
            nyears_train = int(ntime*ntrain)
            nyears_valid = ntime - nyears_train

            X_train = X_t[:,0:nyears_train]
            X_train_2d = X_train
            X_valid = X_t[:,nyears_train:]
        else: 
            tsamp = X_var.shape[1]
            nyears_train = int((tsamp*ntrain)/12)
            nyears_valid = int((tsamp*(1-ntrain))/12)

            X_t = np.reshape(X_var,(X_var.shape[0],int(tsamp/12),12))

            X_train = X_t[:,0:nyears_train,mo:mo+2]
            X_train_2d = np.reshape(X_train,(X_train.shape[0],nyears_train*2))
            X_valid = X_t[:,nyears_train:,mo]

        acell = areacell[areawt_name[var]]
        if len(acell.shape)>1:
            acell_1d = np.reshape(acell,(acell.shape[0]*acell.shape[1]))
        else: 
            acell_1d = acell
            
        [Ptrunc[var], E3[var], tot_var[var],
         tot_var_eig[var], W_all[var], 
         standard_factor[var]] = limkb.step1_compress_individual_var(X_train_2d, var, ntrunc, nmodes_sic, 
                                                                     var_dict, areawt=acell_1d,
                                                                     wt=wt, sic_separate=sic_separate)
        
        Ptrunc_valid[var] = limkb.step1_projection_validation_var(X_valid, E3[var], standard_factor[var], 
                                                                  W_all[var])

        del X_var
    
    return Ptrunc, Ptrunc_valid, E3, tot_var, tot_var_eig, W_all, standard_factor, nyears_train, var_dict

def load_training_data_truncated(limvars, mod_folder, mod_sic_filename, mod_filename,
                                 mo, nyearstot, nyearstrain, yrstart_train, nyearsvalid, 
                                 ind_month_trunc = False): 

    var_dict = {}
    v = {}
    tot_var = {}
    tot_var_eig = {}
    W_all = {}
    E3 = {}
    Ptrunc = {}
    Ptrunc_valid = {}
    standard_factor = {}
    
    for k, var in enumerate(limvars): 
#         if var == 'sic': 
#             mod_fname = mod_sic_filename
#         else: 
#             mod_fname = mod_filename
            
        if var == 'sic': 
            mod_fname = mod_sic_filename
        else: 
            mod_fname = mod_filename[var]
    
        [X_var, v, E3[var], standard_factor[var], 
         W_all[var]] = limkb.load_truncated_data(var, mod_folder, mod_fname)
        
        var_dict[var] = v[var]
    
        if mo is 'all':
            print('Month is '+str(mo))
            X_t = X_var
            ntime = X_t.shape[1]
            nyears_train = (nyearstrain)*12
            nyears_valid = ntime - nyears_train

            X_train = X_t[:,yrstart_train:nyears_train]
            X_train_2d = X_train
            X_valid = X_t[:,nyears_train:]
        elif ind_month_trunc is True: 
            print('Month is '+str(mo)+', ind_month_trunc=True')
            ntime = X_var.shape[1]
            nyears_train = ntime/2
            nyears_valid = nyearsvalid
            
            X_train = np.reshape(X_var,(X_var.shape[0],int(ntime/2),2))
            X_train_2d = X_var
            
#             X_t = np.reshape(X_var,(X_var.shape[0],int(ntime/2),2))

#             X_train = X_t
#             X_train_2d = np.reshape(X_train,(X_train.shape[0],nyears_train*2))
            X_valid = X_train[:,:,1]
        else: 
            print('Month is '+str(mo))
            ntime = X_var.shape[1]
            nyears_train = int((ntime*ntrain)/12)
            nyears_valid = int((ntime*(1-ntrain))/12)

            X_t = np.reshape(X_var,(X_var.shape[0],int(ntime/12),12))

            X_train = X_t[:,0:nyears_train,mo:mo+2]
            X_train_2d = np.reshape(X_train,(X_train.shape[0],nyears_train*2))
            X_valid = X_t[:,nyears_train:,mo]

        Ptrunc[var] = X_train_2d 
        Ptrunc_valid[var] = X_valid
#         Ptrunc_valid[var] = limkb.step1_projection_validation_var(X_valid, E3[var], standard_factor[var], 
#                                                                   W_all[var])

        del X_var
    
    return Ptrunc, Ptrunc_valid, E3, tot_var, tot_var_eig, W_all, standard_factor, nyears_train, var_dict

def load_training_data_truncated_og(limvars, mod_folder, mod_sic_filename, mod_filename,
                                  mo, ntrain, ind_month_trunc = False): 

    var_dict = {}
    v = {}
    tot_var = {}
    tot_var_eig = {}
    W_all = {}
    E3 = {}
    Ptrunc = {}
    Ptrunc_valid = {}
    standard_factor = {}
    
    for k, var in enumerate(limvars): 
        if var == 'sic': 
            mod_fname = mod_sic_filename
        else: 
            mod_fname = mod_filename
    
        [X_var, v, E3[var], standard_factor[var], 
         W_all[var]] = limkb.load_truncated_data(var, mod_folder, mod_fname)
        
        var_dict[var] = v[var]
    
        if mo is 'all':
            X_t = X_var
            ntime = X_t.shape[1]
            nyears_train = int(ntrain*ntime)
            nyears_valid = ntime - nyears_train

            X_train = X_t[:,0:nyears_train]
            X_train_2d = X_train
            X_valid = X_t[:,nyears_train:]
        elif ind_month_trunc is True: 
            ntime = X_var.shape[1]
            nyears_train = int((ntime/2)*ntrain)
            nyears_valid = ntime - nyears_train
            
            X_t = np.reshape(X_var,(X_var.shape[0],int(ntime/2),2))

            X_train = X_t[:,0:nyears_train,:]
            X_train_2d = np.reshape(X_train,(X_train.shape[0],nyears_train*2))
            X_valid = X_t[:,nyears_train:,:]
        else: 
            ntime = X_var.shape[1]
            nyears_train = int((ntime*ntrain)/12)
            nyears_valid = int((ntime*(1-ntrain))/12)

            X_t = np.reshape(X_var,(X_var.shape[0],int(ntime/12),12))

            X_train = X_t[:,0:nyears_train,mo:mo+2]
            X_train_2d = np.reshape(X_train,(X_train.shape[0],nyears_train*2))
            X_valid = X_t[:,nyears_train:,mo]

        Ptrunc[var] = X_train_2d 
        Ptrunc_valid[var] = X_valid
#         Ptrunc_valid[var] = limkb.step1_projection_validation_var(X_valid, E3[var], standard_factor[var], 
#                                                                   W_all[var])

        del X_var
    
    return Ptrunc, Ptrunc_valid, E3, tot_var, tot_var_eig, W_all, standard_factor, nyears_train, var_dict


def get_var_indices(limvars, var_dict): 
    """Get indices for each variable
    """
    start = 0
    for k, var in enumerate(limvars): 
        print('working on '+var)
        inds = var_dict[var]['var_ndof']
        var_inds = np.arange(start,start+inds,1)
        start = inds+start

        var_dict[var]['var_inds'] = var_inds
        
    return var_dict


def calc_validataion_stats(var, truth_anom, forecast_anom, var_dict,areacell,areacell_dict,
                           areawt_name,LIMd,lat_cutoff=False,iplot=False):
    """
    """ 
#     units = areacell_dict[areawt_name[var]][areawt_name[var]]['units']
    
    #Check cell area units: change all to km^2
#     if 'km' in units:
#         acell = areacell[areawt_name[var]]
#     else: 
#         print('changing cellarea units from '+
#               str(areacell_dict[areawt_name[var]][areawt_name[var]]['units'])+' to km^2')
#         acell = areacell[areawt_name[var]]*(1e-6)
#         units = 'km^2'
    units = areacell_dict[var][areawt_name[var]]['units']

    if 'km' in units:
        acell = areacell[var]
    elif 'centi' in units: 
        print('changing cellarea units from '+
              str(areacell_dict[var][areawt_name[var]]['units'])+' to km^2')
        acell = areacell[var]*(1e-10)
        units = 'km^2'
    else: 
        print('changing cellarea units from '+
              str(areacell_dict[var][areawt_name[var]]['units'])+' to km^2')
        acell = areacell[var]*(1e-6)
        units = 'km^2'
        
    if var == 'sic':
        nlon = int(var_dict[var]['var_ndof']/var_dict[var]['lat'].shape[0])
        tot_var_forecast = statskb.calc_tot_si_checks(forecast_anom,acell,units,var_dict[var]['lat'],nlon,lat_cutoff=0.0)
        tot_var_truth = statskb.calc_tot_si_checks(truth_anom,acell,units,var_dict[var]['lat'],nlon,lat_cutoff=0.0)
#     elif var == 'sit':
#         tot_var_forecast = statskb.calc_tot_si(forecast_anom, areacell[areawt_name[var]], 
#                                                   units, var_dict[var]['lat'], lat_cutoff=0.0)
#         tot_var_truth = statskb.calc_tot_si(truth_anom, areacell[areawt_name[var]], 
#                                                units, var_dict[var]['lat'],lat_cutoff=0.0)
    else: 
        tot_var_forecast = statskb.global_mean(forecast_anom,acell)
        tot_var_truth = statskb.global_mean(truth_anom,acell)
    
    
    if iplot==True: 
        time = var_dict[var]['time']
        ntime = tot_var_truth.shape[0]
        ttime = time.shape[0]
        
        plt.figure(figsize=(6,4))
        plt.plot(tot_var_truth,label='truth')
        plt.plot(tot_var_forecast,label='forecast')
#         plt.plot(time[-ntime:],tot_var_truth,label='truth')
#         plt.plot(time[-ntime:],tot_var_forecast,label='forecast')
        plt.xlim(0,20)
#         plt.xlim(time[ttime-ntime],time[ttime-ntime+20])
#         plt.xticks(rotation = 45)
       # plt.ylim(-1.5,1.5)
        plt.legend()
        plt.show()

    corr_tot = np.corrcoef(tot_var_truth,np.nan_to_num(tot_var_forecast))[0,1]
    ce_tot = LMR_utils.coefficient_efficiency(tot_var_truth,np.nan_to_num(tot_var_forecast))

    #error_var = np.nanvar(truth_anom-forecast_anom,axis=1,ddof=1)
#    truth_error_var = np.nanvar(truth_anom,axis=1,ddof=1)

    ## Error var calculations: prior to 09/23/21
#     error_var = np.nansum((truth_anom-forecast_anom)**2,axis=1)
#     truth_error_var = np.nansum((truth_anom - np.nanmean(truth_anom,axis=1)[:,np.newaxis])**2,axis=1)
    
#     gm_error_var = statskb.global_mean(error_var,acell)
#     gm_truth_var = statskb.global_mean(truth_error_var,acell)
#     gm_var_ratio=gm_error_var/gm_truth_var
    
    ## New error var calculations: 09/23/21 locally normalized
#     error_var = np.nanvar((truth_anom-forecast_anom),axis=1,ddof=1)
#     truth_error_var = np.nanvar((truth_anom),axis=1,ddof=1)

#     error_var_nonan = np.where(np.isclose(error_var,0),np.nan,error_var)
#     truth_error_var_nonan = np.where(np.isclose(truth_error_var,0),np.nan,truth_error_var)

#     gm_var_ratio = error_var/truth_error_var_nonan
#     gm_error_var = statskb.global_mean(gm_var_ratio,acell)
    
     ## New error var calculations: 10/06/21
    forecast_nan_mask = np.where(np.isclose(np.nanvar(LIMd['E3'][var],axis=1),0,atol=1e-5),np.nan,1)

    rmse = np.sqrt(np.nanmean((truth_anom-forecast_anom)**2,axis=1))
    gm_rmse = statskb.global_mean(rmse*forecast_nan_mask,acell)
    gsum_rmse = np.nansum(gm_rmse*forecast_nan_mask)
    
    return corr_tot, ce_tot, gm_rmse, gsum_rmse, rmse


def gather_truth_forecast(lag,var,mo,X_var,x_forecast_dcomp,
                          nvalidyrs,var_dict,ntrain,insamp=True):     
    """Gathers truth and forecast data for a LIM trained on all months or individual month
       for a given lag. 
       
       INPUTS:
       =======
       lag:      scalar value representing forecast lag of interest
       var:      string with variable name
       mo:       either month data was trained on, integer value (0 to 11) OR 
                 'all' indicating the training of the LIM was performed on all months together. 
       X_var:    training/truth data output from load_data(), shape is (ndof, ntime)
       x_forecast_dcomp: forecast data, shape is (nlags, ndof, nyears)
       nvalidyrs: total timesteps used for validation.
       var_dict: dictionary with variable information output from load_data()
       ntrain:   scalar value between 0 to 1 representing fraction of timesteps 
                 in X_var used for training
       insamp:   Boolean value, True indicates insample validation (training data = validation data)
                 False indicates out of sample validation
       
       OUTPUTS: 
       ========
       truth_anom:    Truth data for variable and lag of interest, shape is (ndof, nyears)
       forecast_anom: Forecast data for variable and lag of interest, shape is (ndof, nyears)
    """
    if len(x_forecast_dcomp.shape)>2: 
        x_forecast = x_forecast_dcomp[lag,var_dict[var]['var_inds'],:] 
    else: 
        x_forecast = x_forecast_dcomp[var_dict[var]['var_inds'],:] 

    tsamp = X_var.shape[1]

    if mo == 'all':
        print('Trained using all months...')
        nyears_train = int(np.floor(tsamp*ntrain))
        if insamp==True: 
            x_truth = X_var[:,lag:nyears_train]
            truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
        else: 
            x_truth = X_var[:,(nyears_train+lag):(nyears_train+lag+nvalidyrs)]
            truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
            
        forecast_anom = x_forecast[:,lag:] - np.nanmean(x_forecast[:,lag:],axis=1)[:,np.newaxis]

    else: 
        print('Trained using month '+str(mo)+'...')
        nyears_train = int((tsamp*ntrain)/12)
        nyears_valid = int((tsamp*(1-ntrain))/12)

        X_t = np.reshape(X_var,(X_var.shape[0],int(tsamp/12),12))
        X_valid = X_t[:,nyears_train:,mo]

        step = mo+lag
        print('Validating against month '+str(step))
        if step>11:
            step = step-12
            start_yr = 0+1
        else: 
            start_yr = 0
            
        if insamp==True: 
            x_truth = X_t[:,start_yr:(nyears_train-lag),step]
            truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
        else: 
            if lag == 0: 
                x_truth = X_t[:,nyears_train:,step]
                truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
            else: 
                x_truth = X_t[:,nyears_train:-lag,step]
                truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]

        x_forecast_new = x_forecast[:,lag:]
        forecast_anom = x_forecast_new - np.nanmean(x_forecast_new,axis=1)[:,np.newaxis]
        
    return truth_anom, forecast_anom

def gather_truth_forecast2(lag,var,mo,X_var,x_forecast_dcomp,
                          nvalidyrs,var_dict,ntrain, nvalid_startyr,insamp=True):     
    """Gathers truth and forecast data for a LIM trained on all months or individual month
       for a given lag. 
       
       INPUTS:
       =======
       lag:      scalar value representing forecast lag of interest
       var:      string with variable name
       mo:       either month data was trained on, integer value (0 to 11) OR 
                 'all' indicating the training of the LIM was performed on all months together. 
       X_var:    training/truth data output from load_data(), shape is (ndof, ntime)
       x_forecast_dcomp: forecast data, shape is (nlags, ndof, nyears)
       nvalidyrs: total timesteps used for validation.
       var_dict: dictionary with variable information output from load_data()
       ntrain:   scalar value between 0 to 1 representing fraction of timesteps 
                 in X_var used for training
       nvalid_startyr: index (integer) to start validation data
       insamp:   Boolean value, True indicates insample validation (training data = validation data)
                 False indicates out of sample validation
       
       OUTPUTS: 
       ========
       truth_anom:    Truth data for variable and lag of interest, shape is (ndof, nyears)
       forecast_anom: Forecast data for variable and lag of interest, shape is (ndof, nyears)
    """
    if len(x_forecast_dcomp.shape)>2: 
        x_forecast = x_forecast_dcomp[lag,var_dict[var]['var_inds'],:] 
    else: 
        x_forecast = x_forecast_dcomp[var_dict[var]['var_inds'],:] 

    tsamp = X_var.shape[1]

    if mo == 'all':
        print('Trained using all months...')
        nyears_train = int(np.floor(tsamp*ntrain))
        if insamp==True: 
            x_truth = X_var[:,lag:nyears_train]
            truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
        else: 
            x_truth_valid = X_var[:,nvalid_startyr:(nvalid_startyr+nvalidyrs)]
            x_truth = x_truth_valid[:,lag:]
            truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
            
        forecast_anom = x_forecast[:,lag:] - np.nanmean(x_forecast[:,lag:],axis=1)[:,np.newaxis]

    else: 
        print('Trained using month '+str(mo)+'...')
        nyears_train = int((tsamp*ntrain)/12)
        nyears_valid = int(exp_setup['nyearsvalid']/12)

        X_t = np.reshape(X_var,(X_var.shape[0],int(tsamp/12),12))
        X_valid = X_t[:,nyears_train:,mo]
        
        start_yr = 0
        step = mo+lag
        print('Validating against month '+str(step))
        if step>11:
            step = step-12
            start_yr = start_yr+1
        else: 
            start_yr = 0
            
        if insamp==True: 
            x_truth = X_t[:,start_yr:(nyears_train-lag),step]
            truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
        else: 
            if lag == 0: 
                x_truth = X_t[:,nvalid_startyr:(nvalid_startyr+nyears_valid) ,step]
                truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
            else: 
                x_truth = X_t[:,nvalid_startyr+start_yr:(nvalid_startyr+nyears_valid),step]
                truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]

        x_forecast_new = x_forecast[:,lag:]
        forecast_anom = x_forecast_new - np.nanmean(x_forecast_new,axis=1)[:,np.newaxis]
        
    return truth_anom, forecast_anom

def gather_truth_forecast_notime(lag,var,mo,X_var,x_forecast_dcomp,var_dict,insamp=True):     
    """Gathers truth and forecast data for a LIM trained on all months or individual month
       for a given lag. 
       
       INPUTS:
       =======
       lag:      scalar value representing forecast lag of interest
       var:      string with variable name
       mo:       either month data was trained on, integer value (0 to 11) OR 
                 'all' indicating the training of the LIM was performed on all months together. 
       X_var:    training/truth data output from load_data(), shape is (ndof, ntime), ntime is times of interest
       x_forecast_dcomp: forecast data, shape is (nlags, ndof, nyears)
       var_dict: dictionary with variable information output from load_data()
       insamp:   Boolean value, True indicates insample validation (training data = validation data)
                 False indicates out of sample validation
       
       OUTPUTS: 
       ========
       truth_anom:    Truth data for variable and lag of interest, shape is (ndof, nyears)
       forecast_anom: Forecast data for variable and lag of interest, shape is (ndof, nyears)
    """
    if len(x_forecast_dcomp.shape)>2: 
        x_forecast = x_forecast_dcomp[lag,var_dict[var]['var_inds'],:] 
    else: 
        x_forecast = x_forecast_dcomp[var_dict[var]['var_inds'],:] 

    tsamp = X_var.shape[1]

    if mo == 'all':
        print('Trained using all months...')
        x_truth = X_var[:,lag:]
        truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
            
        forecast_anom = x_forecast[:,lag:] - np.nanmean(x_forecast[:,lag:],axis=1)[:,np.newaxis]

#     else: 
#         print('Trained using month '+str(mo)+'...')

#         X_t = np.reshape(X_var,(X_var.shape[0],int(tsamp/12),12))
#         X_valid = X_t[:,:,mo]
        
#         start_yr = 0
#         step = mo+lag
#         print('Validating against month '+str(step))
#         if step>11:
#             step = step-12
#             start_yr = start_yr+1
#         else: 
#             start_yr = 0
            
#         if insamp==True: 
#             x_truth = X_t[:,:-lag,step]
#             truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
#         else: 
#             if lag == 0: 
#                 x_truth = X_t[:,:,step]
#                 truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
#             else: 
#                 x_truth = X_t[:,start_yr:,step]
#                 truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]

#         x_forecast_new = x_forecast[:,lag:]
#         forecast_anom = x_forecast_new - np.nanmean(x_forecast_new,axis=1)[:,np.newaxis]
        
    return truth_anom, forecast_anom

def gather_truth_forecast_allmo(lag,var,mo,X_var,x_forecast_dcomp,
                                var_dict,ntrain,nyears_startvalid):
    """Gathers truth and forecast data for a LIM trained on all months, but wanting 
       to validate across individual months. 
       
       INPUTS:
       =======
       lag: scalar value representing forecast lag of interest
       var: string with variable name
       mo: forecasted month of interest, integer value (0 to 11)
       X_var: training/truth data output from load_data(), shape is (ndof, ntime)
       x_forecast_dcomp: 
       var_dict: dictionary with variable information output from load_data()
       ntrain: scalar value between 0 to 1 representing fraction of timesteps 
               in X_var used for training
       nyears_train: integer value indicating the start year of validation data
       insamp: Boolean value, True indicates insample validation (training data = validation data)
               False indicates out of sample validation
       
       OUTPUTS: 
       ========
       truth_anom: Truth data for variable and month of interest, shape is (ndof, nyears)
       forecast_anom:  Forecast data for variable and month of interest, shape is (ndof, nyears)
    """
    
    if lag is None: 
        x_forecast = x_forecast_dcomp[var_dict[var]['var_inds'],:] 
        nvalidyrs = x_forecast_dcomp.shape[1]
    else: 
        x_forecast = x_forecast_dcomp[lag,var_dict[var]['var_inds'],:]
        nvalidyrs = x_forecast_dcomp.shape[2]

    tsamp = X_var.shape[1]
 
    print('Trained using month '+str(mo)+'...')
    nyears_train = int((tsamp*ntrain)/12)
#     nyears_valid = int((tsamp*(1-ntrain))/12)
#     nyears_valid = int((tsamp*nvalid)/12)

    X_t = np.reshape(X_var,(X_var.shape[0],int(tsamp/12),12))
#     X_valid = X_t[:,nyears_train:,mo]

    step = mo
    print('Validating against month '+str(step))
    if step>11:
        step = step-11
        start_yr = 1
    else: 
        start_yr = 0

    if lag == 0: 
        x_truth = X_t[:,nyears_startvalid:(nyears_startvalid+nvalidyrs),step]
        truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
    else: 
        x_truth = X_t[:,(nyears_startvalid+start_yr):(nyears_startvalid+nvalidyrs),step]
        truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]

    forecast_anom = x_forecast - np.nanmean(x_forecast,axis=1)[:,np.newaxis]
        
    return truth_anom, forecast_anom

def gather_truth_forecast_allmo_notime(lag,var,mo,X_var,x_forecast_dcomp,var_dict):
    """Gathers truth and forecast data for a LIM trained on all months, but wanting 
       to validate across individual months. 
       
       INPUTS:
       =======
       lag: scalar value representing forecast lag of interest
       var: string with variable name
       mo: forecasted month of interest, integer value (0 to 11)
       X_var: from load_data(), shape is (ndof, ntime) already validation years 
       x_forecast_dcomp: 
       var_dict: dictionary with variable information output from load_data()
       ntrain: scalar value between 0 to 1 representing fraction of timesteps 
               in X_var used for training
       nyears_train: integer value indicating the start year of validation data
       insamp: Boolean value, True indicates insample validation (training data = validation data)
               False indicates out of sample validation
       
       OUTPUTS: 
       ========
       truth_anom: Truth data for variable and month of interest, shape is (ndof, nyears)
       forecast_anom:  Forecast data for variable and month of interest, shape is (ndof, nyears)
    """
    start_yr =0
    if lag is None: 
        x_forecast = x_forecast_dcomp[var_dict[var]['var_inds'],:] 
        nvalidyrs = x_forecast_dcomp.shape[1]
        step = mo      # Changed 11/23/21 to add lag 
    else: 
        x_forecast = x_forecast_dcomp[lag,var_dict[var]['var_inds'],:]
        nvalidyrs = x_forecast_dcomp.shape[2]
        step = mo+lag      # Changed 11/23/21 to add lag 

    tsamp = X_var.shape[1]
 
    print('Trained using month '+str(mo)+'...')

    X_t = np.reshape(X_var,(X_var.shape[0],int(tsamp/12),12))

#    step = mo     # Changed 11/23/21 to add lag 
    print('Validating against month '+str(step))
    if step>10:
        step = step-11
        start_yr = start_yr+1
        x_forecast = x_forecast[:,:-start_yr]
    else: 
        start_yr = start_yr

    if lag == 0: 
        x_truth = X_t[:,:,step]
        truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
    else: 
        x_truth = X_t[:,start_yr:,step]
        truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]

    forecast_anom = x_forecast - np.nanmean(x_forecast,axis=1)[:,np.newaxis]
        
    return truth_anom, forecast_anom

def gather_truth_forecast_allmo_notime_test(lag,var,mo,X_var,x_forecast_dcomp,var_dict):
    """Gathers truth and forecast data for a LIM trained on all months, but wanting 
       to validate across individual months. 
       
       INPUTS:
       =======
       lag: scalar value representing forecast lag of interest
       var: string with variable name
       mo: forecasted month of interest, integer value (0 to 11)
       X_var: from load_data(), shape is (ndof, ntime) already validation years 
       x_forecast_dcomp: 
       var_dict: dictionary with variable information output from load_data()
       ntrain: scalar value between 0 to 1 representing fraction of timesteps 
               in X_var used for training
       nyears_train: integer value indicating the start year of validation data
       insamp: Boolean value, True indicates insample validation (training data = validation data)
               False indicates out of sample validation
       
       OUTPUTS: 
       ========
       truth_anom: Truth data for variable and month of interest, shape is (ndof, nyears)
       forecast_anom:  Forecast data for variable and month of interest, shape is (ndof, nyears)
    """
    start_yr =0
    
    if lag is None: 
        x_forecast = x_forecast_dcomp[var_dict[var]['var_inds'],:] 
        nvalidyrs = x_forecast_dcomp.shape[1]
        step = mo      # Changed 11/23/21 to add lag 
    else: 
        x_forecast = x_forecast_dcomp[lag,var_dict[var]['var_inds'],:]
        nvalidyrs = x_forecast_dcomp.shape[2]
        step = mo      # Changed 11/23/21 to add lag 

    tsamp = X_var.shape[1]
 
    print('Trained using month '+str(mo)+'...')

    X_t = np.reshape(X_var,(X_var.shape[0],int(tsamp/12),12))

#    step = mo     # Changed 11/23/21 to add lag 
    print('Validating against month '+str(step))
    if step>10:
        step = step-11
        start_yr = start_yr+1
        x_forecast = x_forecast[:,start_yr:]
    else: 
        start_yr = start_yr

    if lag == 0: 
        x_truth = X_t[:,:,step]
        truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
    else: 
        x_truth = X_t[:,start_yr:,step]
        truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]

    forecast_anom = x_forecast - np.nanmean(x_forecast,axis=1)[:,np.newaxis]
        
    return truth_anom, forecast_anom

def gather_truth_forecast_lag(lag,var,mo,X_var,x_forecast_dcomp,var_dict,insamp=True):     
    """Gathers truth and forecast data for a LIM trained on all months or individual month
       for a given lag. 
       
       INPUTS:
       =======
       lag:      scalar value representing forecast lag of interest
       var:      string with variable name
       mo:       either month data was trained on, integer value (0 to 11) OR 
                 'all' indicating the training of the LIM was performed on all months together. 
       X_var:    training/truth data output from load_data(), shape is (ndof, ntime), ntime is times of interest
       x_forecast_dcomp: forecast data, shape is (nlags, ndof, nyears)
       var_dict: dictionary with variable information output from load_data()
       insamp:   Boolean value, True indicates insample validation (training data = validation data)
                 False indicates out of sample validation
       
       OUTPUTS: 
       ========
       truth_anom:    Truth data for variable and lag of interest, shape is (ndof, nyears)
       forecast_anom: Forecast data for variable and lag of interest, shape is (ndof, nyears)
    """
    if len(x_forecast_dcomp.shape)>2: 
        x_forecast = x_forecast_dcomp[lag,var_dict[var]['var_inds'],:] 
    else: 
        x_forecast = x_forecast_dcomp[var_dict[var]['var_inds'],:] 

    tsamp = X_var.shape[1]

    if mo == 'all':
        print('Trained using all months...')
        x_truth = X_var[:,lag:]
        truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
            
        forecast_anom = x_forecast[:,lag:] - np.nanmean(x_forecast[:,lag:],axis=1)[:,np.newaxis]
    else: 
        print('Currently not supported unless all months included.')
        
    return truth_anom, forecast_anom

def gather_truth_forecast_monthly(var,mo,X_var,x_forecast_dcomp,var_dict):
    """Gathers truth and forecast data for a LIM trained on all months, but wanting 
       to validate across individual months. Assumes lag is already built in, i.e. if 
       month 0 is given it assumes that that is the forecast for January and should be 
       validated against January (month=0). 
       
       INPUTS:
       =======
       var: string with variable name
       mo: forecasted month of interest, integer value (0 to 11)
       X_var: from load_data(), shape is (ndof, ntime) already validation years 
       x_forecast_dcomp: 
       var_dict: dictionary with variable information output from load_data()
       ntrain: scalar value between 0 to 1 representing fraction of timesteps 
               in X_var used for training
       nyears_train: integer value indicating the start year of validation data
       insamp: Boolean value, True indicates insample validation (training data = validation data)
               False indicates out of sample validation
       
       OUTPUTS: 
       ========
       truth_anom: Truth data for variable and month of interest, shape is (ndof, nyears)
       forecast_anom:  Forecast data for variable and month of interest, shape is (ndof, nyears)
    """
    
    x_forecast = x_forecast_dcomp[var_dict[var]['var_inds'],:] 
    nvalidyrs = x_forecast_dcomp.shape[1]
    step = mo
    if mo ==0:
        start_yr = 1
    else: 
        start_yr = 0

    tsamp = X_var.shape[1]
 
    print('Trained using month '+str(mo)+'...')

    X_t = np.reshape(X_var,(X_var.shape[0],int(tsamp/12),12))

    print('Validating against month '+str(step))

    x_forecast = x_forecast[:,start_yr:]
    forecast_anom = x_forecast - np.nanmean(x_forecast,axis=1)[:,np.newaxis]

    x_truth = X_t[:,start_yr:,step]
    truth_anom = x_truth - np.nanmean(x_truth,axis=1)[:,np.newaxis]
        
    return truth_anom, forecast_anom

def detrend_lagged_truth_anom(truth_anom, atol=False, remove_mn=False):
    print('Detrending truth_anom.')
    truth_anom_dt = np.zeros((truth_anom.shape))
    for i in range(12):
        print('month...'+str(i))
        y = truth_anom[:,i::12]
        if np.isnan(y).sum()>0:
            print('Found some nans, going to fill with previous timestep...')
            inds = np.where(np.isnan(y))
            ind_int = int(inds[1].min()-1)
            fill = np.ones(y.shape)*y[:,ind_int][:,np.newaxis]
            var_nans_mask = np.where(np.isnan(y),np.nan,1)
            Y = np.where(np.isnan(y),fill,y)
        else: 
            Y = y
#                print('Y = '+ str(Y.shape))
        X = np.arange(0,int(Y.shape[1]))
        [var_dt,_,intercept] = statskb.multi_linear_detrend(X,Y,axis=1,atol=atol,remove_mn=remove_mn)
        if np.isnan(y).sum()>0:
            truth_anom_dt[:,i::12] = var_dt*var_nans_mask
        else: 
            truth_anom_dt[:,i::12] = var_dt
    return truth_anom_dt
    