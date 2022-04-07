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
import LIM_utils_kb as limkb
import LIM_stats_kb as statskb
import LIM_plot_kb as plotkb
import LIM_building as limbuild

sys.path.append("/home/disk/kalman2/mkb22/pyLMR/")
import LMR_utils

from datetime import date

today = date.today()

#Year-month-day
today_date = today.strftime("%Y%m%d")

import warnings
warnings.filterwarnings("ignore")

pi = np.pi

    
def build_L(exp_setup,  L_folder, save=False):
    """This builds and saves L.
    """

#     fdic_train = limkb.build_training_dic(exp_setup['train_dsource'])

    full_names, areawt_name, month_names = limbuild.load_full_names(exp_setup['train_dsource'])

    #--------------------------------------------------
    # BUILD L FROM PRE-TRUNCATED DATA: 
    #--------------------------------------------------
    
#     [Ptrunc, _, E3, tot_var, 
#      tot_var_eig, W_all, standard_factor, 
#      nyears_train, var_dict] = limbuild.load_training_data_truncated(exp_setup['limvars'], exp_setup['mod_folder'], 
#                                                                      exp_setup['mod_sic_filename'], 
#                                                                      exp_setup['mod_filename'], 
#                                                                      exp_setup['mo'],exp_setup['nyearstot'], 
#                                                                      exp_setup['nyearstrain'],exp_setup['nyearsvalid'],
#                                                                      ind_month_trunc=exp_setup['ind_month_trunc'])
    [Ptrunc, _, E3, tot_var, 
     tot_var_eig, W_all, standard_factor, 
     nyears_train, var_dict] = limbuild.load_training_data_truncated(exp_setup['limvars'], exp_setup['mod_folder'], 
                                                                     exp_setup['mod_filename']['sic'], 
                                                                     exp_setup['mod_filename'], 
                                                                     exp_setup['mo'],exp_setup['nyearstot'], 
                                                                     exp_setup['nyearstrain'],exp_setup['nyears_starttrain'],
                                                                     exp_setup['nyearsvalid'],
                                                                     ind_month_trunc=exp_setup['ind_month_trunc'])
    print('nyears_train = '+str(nyears_train))
    
    var_dict = limbuild.get_var_indices(exp_setup['limvars'], var_dict)

    ndof_all = limkb.count_ndof_all(exp_setup['limvars'], E3, sic_separate=exp_setup['sic_separate'])
    

    if len(exp_setup['limvars'])<=1:
        print('Only one variable detected...')
        Ptrunc_all = []
        E3_all = []
        Ptrunc_sic = Ptrunc['sic']
        E_sic = E3['sic']
        
        P_train = Ptrunc_sic
    else: 
        print('Multiple variables detected...')
        [Ptrunc_all, E3_all, 
        Ptrunc_sic,E_sic] = limkb.stack_variable_eofs(exp_setup['limvars'], ndof_all, 
                                                      exp_setup['ntrunc'], Ptrunc, E3,var_dict,
                                                      sic_separate=exp_setup['sic_separate'])

        P_train = np.concatenate((Ptrunc_all, Ptrunc_sic),axis=0)

    # TRAIN LIM: 
    #--------------------------------------------------

    if exp_setup['mo'] is 'all':
        LIMd2, G2 = limkb.LIM_train(exp_setup['tau'],P_train)
        print('Training LIM with tau = '+str(exp_setup['tau']))
    else: 
        nmo = int(P_train.shape[1]/nyears_train)
    #    nmo = int(P_train.shape[1]/exp_setup['nyearstrain'])
        # nmo = 2
        P_train_3d = np.reshape(P_train, (P_train.shape[0],nyears_train,nmo))
        
        LIMd2, G2 = limkb.LIM_train_flex(exp_setup['tau'],P_train_3d[:,:,0], P_train_3d[:,:,1])
        print('Training LIM with tau = '+str(exp_setup['tau']))

    #--------------------------------------------------

    max_eigenval = np.real(LIMd2['lam_L']).max()

    if max_eigenval >0: 
        LIMd2['lam_L_adj'] = LIMd2['lam_L'] - (max_eigenval+0.01)
    else: 
        LIMd2['lam_L_adj'] = LIMd2['lam_L']

    LIMd2['npos_eigenvalues'] = (LIMd2['lam_L']>0).sum()/(LIMd2['lam_L'].shape[0])
    print('Number of positive eigenvalues = '+ str((LIMd2['lam_L']>0).sum()/(LIMd2['lam_L'].shape[0])))
    
    LIMd2['E3'] = E3
    LIMd2['W_all'] = W_all
    LIMd2['standard_factor'] = standard_factor
    LIMd2['E3_all'] = E3_all
    LIMd2['E_sic'] = E_sic
    LIMd2['var_dict'] = var_dict
    LIMd2['P_train'] = P_train
    LIMd2['exp_setup'] = exp_setup

    #--------------------------------------------------
    if save is True: 

        L_filename = ('L_'+exp_setup['train_dsource']+'_ntrain_'+exp_setup['mod_filename'][-22:-13]+'_'+
                      (str(exp_setup['ntrunc'])+"_").join(exp_setup['limvars'])+
                      str(exp_setup['nmodes_sic'])+'_'+today_date+'.pkl')

        print('saving in: '+L_folder+L_filename)
        pickle.dump(LIMd2, open(L_folder+L_filename, "wb" ) )
        
    return LIMd2


def run_forecast(LIMd, exp_setup,  f_folder=None, verbose=True, save=False, save_decomp=False):
    """Projects forecast initialization data into eof space and performs forecast. 
    """
    print('------------------------------------------------')
    print('------------------------------------------------')
    print('STARTING run_forecast()')
    print('------------------------------------------------')
    print('------------------------------------------------')
    
    fdic_valid = limkb.build_training_dic(exp_setup['valid_dsource'])
    
    Ptrunc_valid = {}
    var_dict_valid = {}
    ntims = len(exp_setup['lags'])
    print('Validation data: '+exp_setup['valid_dsource'])

    for k, var in enumerate(exp_setup['limvars']): 
        tecut = int(exp_setup['nyears_startvalid']/12)+(exp_setup['nyearsvalid'])
#        tecut = exp_setup['nyearstot'] - (int(exp_setup['nyears_startvalid']/12)+(exp_setup['nyearsvalid']))
                
        print('tecut = '+str(tecut) +', tscut = '+str(exp_setup['nyears_startvalid']/12))
    
        X_var_valid, var_dict_valid = limkb.load_data(var, var_dict_valid, fdic_valid, 
                                                      remove_climo=exp_setup['remove_climo'], 
                                                      detrend=exp_setup['detrend'], verbose=verbose, 
                                                      tscut=int(exp_setup['nyears_startvalid']/12), 
                                                      tecut=tecut, lat_cutoff=exp_setup['lat_cutoff'][var])
        if 'sic' in var:
            if np.nanmax(X_var_valid)>1:
                print('Changing units of sic be a between 0 to 1')
                X_var_valid = X_var_valid/100

        print('Validation shape: '+str(X_var_valid.shape))
   
        if 'datetime64' in str(type(var_dict_valid[var]['time'][0])):
            print('time dimension: '+str(var_dict_valid[var]['time'][0].astype('M8[Y]'))+' - '+
                  str(var_dict_valid[var]['time'][-1].astype('M8[Y]')))
            print(var_dict_valid[var]['time'].shape)
        else: 
            print('time dimension: '+str(var_dict_valid[var]['time'][0].year)+' - '+
                  str(var_dict_valid[var]['time'][-1].year))
            print(var_dict_valid[var]['time'].shape)
        
        Ptrunc_valid[var] = limkb.step1_projection_validation_var(X_var_valid, LIMd['E3'][var], 
                                                                  LIMd['standard_factor'][var],
                                                                  LIMd['W_all'][var], Weights=exp_setup['Weight'])

    var_dict_valid = limbuild.get_var_indices(exp_setup['limvars'], var_dict_valid)
    ndof_all_valid = limkb.count_ndof_all(exp_setup['limvars'], LIMd['E3'], sic_separate=exp_setup['sic_separate'])

    if len(exp_setup['limvars'])<=1:
        print('Only one variable detected...')
        Ptrunc_all_valid = []
        E3_all_valid = []
        Ptrunc_sic_valid = Ptrunc_valid['sic']
        E_sic_valid = LIMd['E3']['sic']
        
        P_train_valid = Ptrunc_sic_valid
    else: 
        print('Multiple variables detected...')
        if exp_setup['sic_separate']:
            [Ptrunc_all_valid, E3_all_valid,
             Ptrunc_sic_valid, E_sic_valid] = limkb.stack_variable_eofs(exp_setup['limvars'], ndof_all_valid, 
                                                                        exp_setup['ntrunc'],Ptrunc_valid, 
                                                                        LIMd['E3'], var_dict_valid, 
                                                                        sic_separate=exp_setup['sic_separate'])
            if exp_setup['step2_trunc']:
                [P_train_valid, Fvar, 
                 E_train_valid] = limkb.step2_multivariate_compress(Ptrunc_all_valid,exp_setup['nmodes'], E3_all_valid, 
                                                                         Ptrunc_sic_valid,
                                                                         sic_separate=exp_setup['sic_separate'],
                                                                         Trunc_truth=False)
            else: 
                P_train_valid = np.concatenate((Ptrunc_all_valid, Ptrunc_sic_valid),axis=0)    
                
        else: 
            [Ptrunc_all_valid, E3_all_valid] = limkb.stack_variable_eofs(exp_setup['limvars'], ndof_all_valid, 
                                                                         exp_setup['ntrunc'],Ptrunc_valid, 
                                                                         LIMd['E3'], var_dict_valid, 
                                                                         sic_separate=exp_setup['sic_separate'])
            if exp_setup['step2_trunc']:
                [P_train_valid, Fvar, 
                 E_train_valid] = limkb.step2_multivariate_compress(Ptrunc_all_valid,exp_setup['nmodes'], 
                                                                         E3_all_valid,0,
                                                                         sic_separate=exp_setup['sic_separate'],
                                                                         Trunc_truth=False)
            else: 
                P_train_valid = Ptrunc_all_valid   

#        P_train_valid = np.concatenate((Ptrunc_all_valid, Ptrunc_sic_valid),axis=0)
    print('P_train_valid: '+str(P_train_valid.shape))

    # LIM forecasts for a range of monthly values (can specify a list of arbitrary values too)    
#     nvalidtimes = P_train_valid.shape[1]
    
    print('Running a forecast!')
    if exp_setup['mo'] == 'all':
        print('Using all months')
        if exp_setup['Insamp']==True: 
            print('Performing in sample forecast')
            LIM_fcast = limkb.LIM_forecast(LIMd,P_train[:,0:nyr_train],exp_setup['lags'],adjust=exp_setup['adj'])
        else: 
            print('Performing out of sample forecast')
            LIM_fcast = limkb.LIM_forecast(LIMd,P_train_valid,exp_setup['lags'],adjust=exp_setup['adj'])
    else: 
        print('Using individual months')
        if exp_setup['Insamp']==True: 
            print('Performing in sample forecast')
            P_train_2d = np.reshape(P_train, (P_train.shape[0],int(P_train.shape[1]/2),2))
            LIM_fcast = limkb.LIM_forecast(LIMd,P_train_2d[:,:,0],exp_setup['lags'],adjust=exp_setup['adj'])
        else: 
            print('Performing out of sample forecast')
    #        LIM_fcast = limkb.LIM_forecast(LIMd2,P_train_valid,lags,adjust=adj)
            LIM_fcast = limkb.LIM_forecast(LIMd,P_train_valid,exp_setup['lags'],adjust=exp_setup['adj'])

    print('LIM_fcast: '+str(LIM_fcast['x_forecast'].shape))
    
    # Decompress LIM forecast: 
    if len(exp_setup['limvars'])<=1:
        x_forecast_dcomp = np.zeros((len(exp_setup['lags']),LIMd['E_sic'].shape[0],
                                     LIM_fcast['x_forecast'].shape[2]))
        nmodes = 0
    else: 
        x_forecast_dcomp = np.zeros((len(exp_setup['lags']),LIMd['E3_all'].shape[0]+LIMd['E_sic'].shape[0],
                                     LIM_fcast['x_forecast'].shape[2]))
        nmodes = LIMd['E3_all'].shape[1]
    
    print('Decompressing forecasted fields out of eof space.')
    print('Working on...')
    for i,lag in enumerate(exp_setup['lags']):
        print('Lag '+ str(lag))
        x_forecast_dcomp[lag,:,:] = limkb.decompress_eof_separate_sic(LIM_fcast['x_forecast'][lag,:,:],
                                                                      nmodes,exp_setup['nmodes_sic'],LIMd['E3_all'],
                                                                      LIMd['E_sic'],exp_setup['limvars'],LIMd['var_dict'],
                                                                      LIMd['W_all'],Weights=exp_setup['Weight'],
                                                                      sic_separate=exp_setup['sic_separate'])
        
        
    forecast = {}
    forecast['P_train_valid'] = P_train_valid
    forecast['exp_setup'] = exp_setup
    forecast['var_dict_valid'] = var_dict_valid
    forecast['x_forecast'] = LIM_fcast['x_forecast']
    if save_decomp is True: 
        forecast['x_forecast_dcomp'] = x_forecast_dcomp
        
    #--------------------------------------------------
    if save is True: 

        f_filename = ('Forecast_'+exp_setup['train_dsource']+'_ntrain_'+exp_setup['mod_filename'][-22:-13]+'_'+
                      (str(exp_setup['ntrunc'])+"_").join(exp_setup['limvars'])+
                      str(exp_setup['nmodes_sic'])+'_'+today_date+'.pkl')

        print('saving in: '+f_folder+f_filename)
        pickle.dump(forecast, open(f_folder+f_filename, "wb" ) )
        
    forecast['x_forecast_dcomp'] = x_forecast_dcomp

    return forecast


def validate_forecast_monthly(forecast, valid_vars, lag, exp_setup, LIMd, f_folder, 
                              iplot=False, save=False):
    print('------------------------------------------------')
    print('------------------------------------------------')
    print('STARTING validate_forecast_monthly()')
    print('------------------------------------------------')
    print('------------------------------------------------')
    
    fdic_train = limkb.build_training_dic(exp_setup['train_dsource'])
    fdic_valid = limkb.build_training_dic(exp_setup['valid_dsource'])

    full_names, areawt_name, month_names = limbuild.load_full_names(exp_setup['train_dsource'])
    
    if 'Amon' in exp_setup['train_dsource']:
        areawt_name['sit'] = 'areacella'
        areawt_name['sic'] = 'areacella'
        areawt_name['tos'] = 'areacella'
    
    areacell_dict_all = {}
    areacell = {}
    for var in exp_setup['limvars']:
        areacell_dict = {}
        areacell[var], areacell_dict_all[var] = limkb.load_data(areawt_name[var], areacell_dict, fdic_train, 
                                                       remove_climo=False, detrend=False, verbose=False, 
                                                       lat_cutoff=exp_setup['lat_cutoff'][var])
        
#     areacell, areacell_dict = limbuild.load_areacell_dict(fdic_train, lat_cutoff=exp_setup['lat_cutoff'],
#                                                           remove_climo=False, detrend=False, verbose=False )
    

    x_forecast_dcomp_mo = np.reshape(forecast['x_forecast_dcomp'], 
                                     (forecast['x_forecast_dcomp'].shape[0], 
                                      forecast['x_forecast_dcomp'].shape[1],
                                      int(forecast['x_forecast_dcomp'].shape[2]/12),12))

    months=[0,1,2,3,4,5,6,7,8,9,10,11]
    validation_stats = {}

    nyear_valid = exp_setup['nyears_startvalid']

    for k, var in enumerate(valid_vars):
        valid_stats = {}
        v = {}
        tecut = int(exp_setup['nyears_startvalid']/12)+(exp_setup['nyearsvalid'])
#        tecut = exp_setup['nyearstot'] - (int(exp_setup['nyears_startvalid']/12)+(exp_setup['nyearsvalid']))

        if tecut <=0: 
            tecut = False
        print('tecut = '+str(tecut) +', tscut = '+str(exp_setup['nyears_startvalid']/12))
        
        X_var, _ = limkb.load_data(var, v, fdic_valid, remove_climo=exp_setup['remove_climo'], 
                                   detrend=exp_setup['detrend'], verbose=True, 
                                   tscut=int(exp_setup['nyears_startvalid']/12), tecut=tecut, 
                                   lat_cutoff=exp_setup['lat_cutoff'][var])

        if var == 'sic':
            if np.nanmax(X_var)>1:
                print('Changing units of sic be a between 0 to 1')
                X_var = X_var/100
                
        corr_tot = np.zeros((len(months)))
        ce_tot = np.zeros((len(months)))
        gm_rmse = np.zeros((len(months)))
        gsum_rmse = np.zeros((len(months)))
        rmse = np.zeros((X_var.shape[0],len(months)))

        for i,m in enumerate(months):
            print('Month '+str(m))
#             [truth_anom, forecast_anom] = limbuild.gather_truth_forecast_allmo(lag,var,m,X_var,
#                                                                                x_forecast_dcomp_mo[:,:,:,m],
#                                                                                LIMd['var_dict'],1,0,
#                                                                                insamp=exp_setup['Insamp'])
 
            [truth_anom, 
             forecast_anom] = limbuild.gather_truth_forecast_monthly(var,m,X_var,
                                                                     x_forecast_dcomp_mo[lag,:,:,m],
                                                                     LIMd['var_dict'])
                
            print('Truth_anom shape: '+str(truth_anom.shape))
            print('Forecast_anom shape: '+str(forecast_anom.shape))

            [corr_tot[i], ce_tot[i], 
             gm_rmse[i], gsum_rmse[i], 
             rmse[:,i]] = limbuild.calc_validataion_stats(var, truth_anom, forecast_anom, LIMd['var_dict'],
                                                          areacell,areacell_dict_all,areawt_name,
                                                          LIMd,iplot=iplot)

        valid_stats['corr_tot'] = corr_tot
        valid_stats['ce_tot'] = ce_tot
        valid_stats['gm_rmse'] = gm_rmse
        valid_stats['gsum_rmse'] = gsum_rmse
        valid_stats['rmse'] = rmse

        validation_stats[var] = valid_stats

        del X_var
        
        
    forecast_validation = {}
    forecast_validation['validation_stats'] = validation_stats
    forecast_validation['exp_setup'] = exp_setup
    
    if save is True: 
#         start_tim = exp_setup['nyears_startvalid']
#         end_tim = exp_setup['nyears_startvalid']+(exp_setup['nyearsvalid']*12)
        valid_yrs_str = (str(forecast['var_dict_valid'][var]['time'][0])[:4]+'_'+
                         str(forecast['var_dict_valid'][var]['time'][-1])[:4])
        
        fvalid_filename = ('Forecast_validation_'+exp_setup['train_dsource']+'_ntrain_'+
                            exp_setup['mod_filename'][-22:-13]+'_'+'validation_'+exp_setup['valid_dsource']+'_'+
                            valid_yrs_str+'_'+(str(exp_setup['ntrunc'])+"_").join(exp_setup['limvars'])+
                            str(exp_setup['nmodes_sic'])+'_'+today_date+'.pkl')

        print('saving in: '+f_folder+fvalid_filename)
        pickle.dump(forecast_validation, open(f_folder+fvalid_filename, "wb" ) )
    
    return forecast_validation


def validate_forecast_lagged(forecast, valid_vars, exp_setup, LIMd, f_folder, 
                             iplot=False, save=False, detrend_truth=False):
    
    print('------------------------------------------------')
    print('------------------------------------------------')
    print('STARTING validate_forecast_lagged()')
    print('------------------------------------------------')
    print('------------------------------------------------')
    
    fdic_train = limkb.build_training_dic(exp_setup['train_dsource'])
    fdic_valid = limkb.build_training_dic(exp_setup['valid_dsource'])

#     areacell, areacell_dict = limbuild.load_areacell_dict(fdic_train, lat_cutoff=exp_setup['lat_cutoff'],
#                                                           remove_climo=False, detrend=False, verbose=False )
    full_names, areawt_name, month_names = limbuild.load_full_names(exp_setup['train_dsource'])
    
    areacell_dict_all = {}
    areacell = {}
    for var in exp_setup['limvars']:
        areacell_dict = {}
        areacell[var], areacell_dict_all[var] = limkb.load_data(areawt_name[var], areacell_dict, fdic_train, 
                                                               remove_climo=False, detrend=False, verbose=False, 
                                                               lat_cutoff=exp_setup['lat_cutoff'][var])

    v = {}
    validation_stats_lags = {}

#    nyear_valid = exp_setup['nyearstot'] - exp_setup['nyearsvalid']

    for k, var in enumerate(valid_vars):
        tecut = int(exp_setup['nyears_startvalid']/12)+(exp_setup['nyearsvalid'])
#        tecut = exp_setup['nyearstot'] - (int(exp_setup['nyears_startvalid']/12)+(exp_setup['nyearsvalid']))
        
        if tecut <=0: 
            tecut = False
        print('tecut = '+str(tecut) +', tscut = '+str(exp_setup['nyears_startvalid']/12))

        X_var, _ = limkb.load_data(var, v, fdic_valid, remove_climo=True, detrend=True, verbose=True,
                                   tscut=int(exp_setup['nyears_startvalid']/12), tecut=tecut, 
                                   lat_cutoff=exp_setup['lat_cutoff'][var])
        
        if var == 'sic':
            if np.nanmax(X_var)>1:
                print('Changing units of sic be a between 0 to 1')
                X_var = X_var/100
        
        corr_tot = np.zeros((len(exp_setup['lags'])))
        ce_tot = np.zeros((len(exp_setup['lags'])))
        gm_rmse = np.zeros((len(exp_setup['lags'])))
        gsum_rmse = np.zeros((len(exp_setup['lags'])))
        rmse = np.zeros((X_var.shape[0],len(exp_setup['lags'])))
        valid_stats = {}

        for i,lag in enumerate(exp_setup['lags']):
            print('Lag '+str(lag))
#             [truth_anom, forecast_anom] = limbuild.gather_truth_forecast2(lag,var,exp_setup['mo'],X_var,
#                                                                           forecast['x_forecast_dcomp'],
#                                                                           exp_setup['nyearsvalid']*12,
#                                                                           LIMd['var_dict'],exp_setup['ntrain'],
#                                                                           exp_setup['nyears_startvalid'],
#                                                                           insamp=exp_setup['Insamp'])
            [truth_anom, 
             forecast_anom] = limbuild.gather_truth_forecast_lag(lag,var,exp_setup['mo'],X_var,
                                                                 forecast['x_forecast_dcomp'],
                                                                 LIMd['var_dict'],insamp=exp_setup['Insamp'])
        
                
            print('Truth_anom shape: '+str(truth_anom.shape))
            print('Forecast_anom shape: '+str(forecast_anom.shape))

            [corr_tot[i], ce_tot[i],  gm_rmse[i], gsum_rmse[i], 
             rmse[:,i]] = limbuild.calc_validataion_stats(var, truth_anom, forecast_anom, LIMd['var_dict'],
                                                              areacell,areacell_dict_all,
                                                              areawt_name,LIMd,iplot=iplot)

        valid_stats['corr_tot'] = corr_tot
        valid_stats['ce_tot'] = ce_tot
        valid_stats['gm_rmse'] = gm_rmse
        valid_stats['gsum_rmse'] = gsum_rmse
        valid_stats['rmse'] = rmse

        validation_stats_lags[var] = valid_stats

        del X_var
        
    forecast_validation_lags = {}
    forecast_validation_lags['validation_stats_lags'] = validation_stats_lags
    forecast_validation_lags['exp_setup'] = exp_setup
    
    if save is True: 
#         start_tim = exp_setup['nyears_startvalid']
#         end_tim = exp_setup['nyears_startvalid']+(exp_setup['nyearsvalid']*12)
        valid_yrs_str = (str(forecast['var_dict_valid'][var]['time'][0])[:4]+'_'+
                         str(forecast['var_dict_valid'][var]['time'][-1])[:4])
        
        fvalid_filename = ('Forecast_validation_lagged_'+exp_setup['train_dsource']+'_ntrain_'+
                            exp_setup['mod_filename'][-22:-13]+'_'+'validation_'+exp_setup['valid_dsource']+'_'+
                            valid_yrs_str+'_'+(str(exp_setup['ntrunc'])+"_").join(exp_setup['limvars'])+
                            str(exp_setup['nmodes_sic'])+'_'+today_date+'.pkl')

        print('saving in: '+f_folder+fvalid_filename)
        pickle.dump(forecast_validation_lags, open(f_folder+fvalid_filename, "wb" ) )
    
    return forecast_validation_lags


def ar1_forecast_valid_by_month(P_train, P_train_valid, LIMd, exp_setup, valid_vars, 
                                month_names, ar1f_folder, forecast, lag=None, iplot=False, 
                                save=False, save_decomp=False):
    """Runs and validates ar1 forecast in eof space for each month. 
    """
    print('------------------------------------------------')
    print('------------------------------------------------')
    print('STARTING ar1_forecast_valid_by_month()')
    print('------------------------------------------------')
    print('------------------------------------------------')
    
    fdic_train = limkb.build_training_dic(exp_setup['train_dsource'])
    fdic_valid = limkb.build_training_dic(exp_setup['valid_dsource'])
    
#     areacell, areacell_dict = limbuild.load_areacell_dict(fdic_train, lat_cutoff=exp_setup['lat_cutoff'],
#                                                           remove_climo=False, detrend=False, verbose=False )
    full_names, areawt_name, month_names = limbuild.load_full_names(exp_setup['train_dsource'])
    
    areacell_dict_all = {}
    areacell = {}
    for var in exp_setup['limvars']:
        areacell_dict = {}
        areacell[var], areacell_dict_all[var] = limkb.load_data(areawt_name[var], areacell_dict, fdic_train, 
                                                               remove_climo=False, detrend=False, verbose=False, 
                                                               lat_cutoff=exp_setup['lat_cutoff'][var])
    
    ar1_forecast, ar1_factor = statskb.red_noise_forecast_ar1(P_train.T, P_train_valid.T, lead=1)
    
    if len(valid_vars)<=1:
        nmodes = 0
        ar1_forecast_dcomp = np.zeros((LIMd['E_sic'].shape[0],
                                       exp_setup['nyearsvalid']*12))
    else: 
        nmodes = LIMd['E3_all'].shape[1]
        ar1_forecast_dcomp = np.zeros((LIMd['E3_all'].shape[0]+ LIMd['E_sic'].shape[0],
                                       exp_setup['nyearsvalid']*12))
    
    nmodes = (len(exp_setup['limvars'])-1)*exp_setup['ntrunc']

    ar1_forecast_dcomp[:,1:] = limkb.decompress_eof_separate_sic(ar1_forecast.T, nmodes,
                                                                 exp_setup['nmodes_sic'],LIMd['E3_all'],
                                                                 LIMd['E_sic'],exp_setup['limvars'],
                                                                 LIMd['var_dict'],LIMd['W_all'],
                                                                 Weights=exp_setup['Weight'], 
                                                                 sic_separate=exp_setup['sic_separate'])

    forecast_ar1_mo = np.reshape(ar1_forecast_dcomp, (ar1_forecast_dcomp.shape[0],
                                                      exp_setup['nyearsvalid'],12))
    
    months=[0,1,2,3,4,5,6,7,8,9,10,11]

    validation_stats_ar1 = {}
    nyear_valid = exp_setup['nyears_startvalid']
    st=0

    for k, var in enumerate(valid_vars):
        valid_stats_ar1 = {}
        v = {}
        tecut = int(exp_setup['nyears_startvalid']/12)+(exp_setup['nyearsvalid'])
#        tecut = exp_setup['nyearstot'] - (int(exp_setup['nyears_startvalid']/12)+(exp_setup['nyearsvalid']))

        if tecut <=0: 
            tecut = False
        print('tecut = '+str(tecut) +', tscut = '+str(exp_setup['nyears_startvalid']/12))

        X_var, _ = limkb.load_data(var, v, fdic_valid, remove_climo=True, detrend=True, verbose=True,
                                   tscut=int(exp_setup['nyears_startvalid']/12), tecut=tecut,
                                   lat_cutoff=exp_setup['lat_cutoff'][var])
        if var == 'sic':
            if np.nanmax(X_var)>1:
                print('Changing units of sic be a between 0 to 1')
                X_var = X_var/100

        corr_tot = np.zeros((len(months)))
        ce_tot = np.zeros((len(months)))
        gm_rmse = np.zeros((len(months)))
        gsum_rmse = np.zeros((len(months)))
        rmse = np.zeros((X_var.shape[0],len(months)))

        for i,m in enumerate(months):
            print('Month '+str(m))

#             [truth_anom, 
#              forecast_anom] = limbuild.gather_truth_forecast_allmo(lag,var,m,X_var,forecast_ar1_mo[:,:,m],
#                                                                    LIMd['var_dict'],1,0,
#                                                                    insamp=exp_setup['Insamp'])
            [truth_anom, 
             forecast_anom] = limbuild.gather_truth_forecast_monthly(var,m,X_var,forecast_ar1_mo[:,:,m],
                                                                     LIMd['var_dict'])
            st = st + LIMd['var_dict'][var]['var_ndof']

            print('Truth_anom shape: '+str(truth_anom.shape))
            print('Forecast_anom shape: '+str(forecast_anom.shape))

    #         [corr_tot[i], ce_tot[i], gm_var_ratio[i], tot_var_forecast, 
    #          tot_var_truth] = limbuild.calc_validataion_stats(var, truth_anom[:,70:111], forecast_anom, var_dict,
    #                                                           areacell,areacell_dict,
    #                                                           areawt_name,month_names,iplot=True)
            [corr_tot[i], ce_tot[i],gm_rmse[i], gsum_rmse[i], 
             rmse[:,i]] = limbuild.calc_validataion_stats(var, truth_anom, forecast_anom, 
                                                          LIMd['var_dict'],
                                                          areacell,areacell_dict_all,
                                                          areawt_name,LIMd,iplot=iplot)

        valid_stats_ar1['corr_tot'] = corr_tot
        valid_stats_ar1['ce_tot'] = ce_tot
        valid_stats_ar1['gm_rmse'] = gm_rmse
        valid_stats_ar1['gsum_rmse'] = gsum_rmse
        valid_stats_ar1['rmse'] = rmse
        valid_stats_ar1['ar1_factor'] = ar1_factor

        validation_stats_ar1[var] = valid_stats_ar1

        del X_var
     
    ar1cast = {}
    ar1cast['validation_stats_ar1'] = validation_stats_ar1
    ar1cast['exp_setup'] = exp_setup
    
    if save_decomp is True: 
        ar1cast['ar1_forecast_dcomp'] = ar1_forecast_dcomp
    #--------------------------------------------------
    if save is True: 
#         start_tim = exp_setup['nyears_startvalid']
#         end_tim = exp_setup['nyears_startvalid']+(exp_setup['nyearsvalid']*12)
        valid_yrs_str = (str(forecast['var_dict_valid'][var]['time'][0])[:4]+'_'+
                         str(forecast['var_dict_valid'][var]['time'][-1])[:4])
        
        ar1f_filename = ('AR1_forecast_monthly_'+exp_setup['train_dsource']+'_ntrain_'+
                            exp_setup['mod_filename'][-22:-13]+'_'+'validation_'+exp_setup['valid_dsource']+'_'+
                            valid_yrs_str+'_'+(str(exp_setup['ntrunc'])+"_").join(exp_setup['limvars'])+
                            str(exp_setup['nmodes_sic'])+'_'+today_date+'.pkl')

        print('saving in: '+ar1f_folder+ar1f_filename)
        pickle.dump(ar1cast, open(ar1f_folder+ar1f_filename, "wb" ) )
        
    return ar1cast


def ar1_forecast_valid_by_lag(P_train, P_train_valid, LIMd, exp_setup, valid_vars, 
                              month_names, ar1f_folder, forecast, 
                              iplot=False, save=False, save_decomp=False,detrend_truth=False):
    """Runs and validates ar1 forecast in eof space for each lag. 
    """
    print('------------------------------------------------')
    print('------------------------------------------------')
    print('STARTING ar1_forecast_valid_by_lag()')
    print('------------------------------------------------')
    print('------------------------------------------------')
    
    fdic_train = limkb.build_training_dic(exp_setup['train_dsource'])
    fdic_valid = limkb.build_training_dic(exp_setup['valid_dsource'])
    
#     areacell, areacell_dict = limbuild.load_areacell_dict(fdic_train, lat_cutoff=exp_setup['lat_cutoff'],
#                                                           remove_climo=False, detrend=False, verbose=False )
    full_names, areawt_name, month_names = limbuild.load_full_names(exp_setup['train_dsource'])
    
    areacell_dict_all = {}
    areacell = {}
    for var in exp_setup['limvars']:
        areacell_dict = {}
        areacell[var], areacell_dict_all[var] = limkb.load_data(areawt_name[var], areacell_dict, fdic_train, 
                                                               remove_climo=False, detrend=False, verbose=False, 
                                                               lat_cutoff=exp_setup['lat_cutoff'][var])
    
    if len(valid_vars)<=1:
        ar1_forecast_dcomp_lag = np.zeros((len(exp_setup['lags']),
                                           LIMd['E_sic'].shape[0],
                                           exp_setup['nyearsvalid']*12))
        ar1_factor_lag = np.zeros((len(exp_setup['lags']),P_train.shape[0]))
    else: 
        ar1_forecast_dcomp_lag = np.zeros((len(exp_setup['lags']),
                                           LIMd['E3_all'].shape[0]+LIMd['E_sic'].shape[0],
                                           exp_setup['nyearsvalid']*12))
        ar1_factor_lag = np.zeros((len(exp_setup['lags']),P_train.shape[0]))

    for l,lag in enumerate(exp_setup['lags'][1:]):
        forecast_ar1, ar1_factor = statskb.red_noise_forecast_ar1(P_train.T, P_train_valid.T, lead=lag)

        if len(valid_vars)<=1:
            nmodes=0
            ar1_forecast_dcomp = np.zeros((LIMd['E_sic'].shape[0],exp_setup['nyearsvalid']*12))
        else:
            nmodes = LIMd['E3_all'].shape[1]
            ar1_forecast_dcomp = np.zeros((LIMd['E3_all'].shape[0]+ LIMd['E_sic'].shape[0],exp_setup['nyearsvalid']*12))
            
        ar1_forecast_dcomp_lag[lag,:,lag:] = limkb.decompress_eof_separate_sic(forecast_ar1.T,nmodes,
                                                                             exp_setup['nmodes_sic'],
                                                                             LIMd['E3_all'],LIMd['E_sic'],
                                                                             exp_setup['limvars'],LIMd['var_dict'],
                                                                             LIMd['W_all'],Weights=exp_setup['Weight'],
                                                                             sic_separate=exp_setup['sic_separate'])
        ar1_factor_lag[lag,:] = ar1_factor                 

    validation_stats_ar1_lags = {}
    lag = None
#     nyear_valid = exp_setup['nyearsvalid']*12
    st=0

    for k, var in enumerate(valid_vars):
        valid_stats_ar1_lags = {}
        v = {}
        tecut = int(exp_setup['nyears_startvalid']/12)+(exp_setup['nyearsvalid'])
#        tecut = exp_setup['nyearstot'] - (int(exp_setup['nyears_startvalid']/12)+(exp_setup['nyearsvalid']))

        if tecut <=0: 
            tecut = False
        print('tecut = '+str(tecut) +', tscut = '+str(exp_setup['nyears_startvalid']/12))

        X_var, _ = limkb.load_data(var, v, fdic_valid, remove_climo=True, detrend=True, verbose=True,
                                  tscut=int(exp_setup['nyears_startvalid']/12), tecut=tecut,
                                  lat_cutoff=exp_setup['lat_cutoff'][var])
        
        if var is 'sic':
            if np.nanmax(X_var)>1:
                print('Changing units of sic be a between 0 to 1')
                X_var = X_var/100

        corr_tot = np.zeros((len(exp_setup['lags'])))
        ce_tot = np.zeros((len(exp_setup['lags'])))
        gm_rmse = np.zeros((len(exp_setup['lags'])))
        gsum_rmse = np.zeros((len(exp_setup['lags'])))
        rmse = np.zeros((X_var.shape[0],len(exp_setup['lags'])))

        for i,lag in enumerate(exp_setup['lags'][1:]):
            print('Lag '+str(lag))
#             [truth_anom, 
#              forecast_anom] = limbuild.gather_truth_forecast2(i,var,exp_setup['mo'],X_var,ar1_forecast_dcomp_lag,
#                                                               exp_setup['nyearsvalid']*12,LIMd['var_dict'],
#                                                               exp_setup['ntrain'],exp_setup['nyears_startvalid'],
#                                                               insamp=exp_setup['Insamp'])
            [truth_anom, 
             forecast_anom] = limbuild.gather_truth_forecast_lag(lag,var,exp_setup['mo'],X_var,
                                                                 ar1_forecast_dcomp_lag,
                                                                 LIMd['var_dict'],insamp=exp_setup['Insamp'])

                
            print('Truth_anom shape: '+str(truth_anom.shape))
            print('Forecast_anom shape: '+str(forecast_anom.shape))

            [corr_tot[i], ce_tot[i], gm_rmse[i], gsum_rmse[i], 
             rmse[:,i]] = limbuild.calc_validataion_stats(var, truth_anom, forecast_anom, 
                                                          forecast['var_dict_valid'], #LIMd['var_dict'],
                                                          areacell,areacell_dict_all,
                                                          areawt_name,LIMd,iplot=iplot)

        valid_stats_ar1_lags['corr_tot'] = corr_tot
        valid_stats_ar1_lags['ce_tot'] = ce_tot
        valid_stats_ar1_lags['gm_rmse'] = gm_rmse
        valid_stats_ar1_lags['gsum_rmse'] = gsum_rmse
        valid_stats_ar1_lags['rmse'] = rmse
        valid_stats_ar1_lags['ar1_factor_lag'] = ar1_factor_lag                     

        validation_stats_ar1_lags[var] = valid_stats_ar1_lags

        del X_var
     
    ar1cast_lags = {}
    ar1cast_lags['validation_stats_ar1_lags'] = validation_stats_ar1_lags
    ar1cast_lags['exp_setup'] = exp_setup
    
    if save_decomp is True: 
        ar1cast_lags['ar1_forecast_dcomp_lag'] = ar1_forecast_dcomp_lag
    #--------------------------------------------------
    if save is True: 
#         start_tim = exp_setup['nyears_startvalid']
#         end_tim = exp_setup['nyears_startvalid']+(exp_setup['nyearsvalid']*12)
        valid_yrs_str = (str(forecast['var_dict_valid'][var]['time'][0])[:4]+'_'+
                         str(forecast['var_dict_valid'][var]['time'][-1])[:4])
        
        ar1f_filename = ('AR1_forecast_lagged_'+exp_setup['train_dsource']+'_ntrain_'+
                            exp_setup['mod_filename'][-22:-13]+'_'+'validation_'+exp_setup['valid_dsource']+'_'+
                            valid_yrs_str+'_'+(str(exp_setup['ntrunc'])+"_").join(exp_setup['limvars'])+
                            str(exp_setup['nmodes_sic'])+'_'+today_date+'.pkl')


        print('saving in: '+ar1f_folder+ar1f_filename)
        pickle.dump(ar1cast_lags, open(ar1f_folder+ar1f_filename, "wb" ) )
        
    return ar1cast_lags
