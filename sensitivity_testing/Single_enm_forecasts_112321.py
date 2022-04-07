import sys
import numpy as np
import pickle

sys.path.append("/home/disk/p/mkb22/Documents/si_analysis_kb/LIMs/SI_LIMs/")
import run_forecast_model_data as rf

from datetime import date

today = date.today()
neofs = 30
jacknife = True

## ---------------------------------------------------
## LOAD PREVIOUSLY RUN LIMCAST FILE: 
## ---------------------------------------------------

data_dir = '/home/disk/kalman2/mkb22/SI_LIMs/sensitivity_testing/neofs/last_millennium/'
data_name = ('LIMcast_cesm_lme_002_ntrain_850_1650_validyrs_1651_1850_tas'+str(neofs)+'_psl'+str(neofs)+'_zg'+
             str(neofs)+'_tos'+str(neofs)+'_sit'+str(neofs)+'_sic'+str(neofs)+'_20211113.pkl')

lim_data = pickle.load(open(data_dir+data_name, "rb" ) )

train_dsource = lim_data['LIMd']['exp_setup']['train_dsource']
master_save=False
save_decomp=False

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

save_folder = '/home/disk/kalman2/mkb22/SI_LIMs/enm_analysis/'+folder_add

Nenms = len(lim_data['LIMd']['exp_setup']['limvars'])*(lim_data['LIMd']['exp_setup']['ntrunc'])
exp_setup = lim_data['LIMd']['exp_setup']

LIMd_smode = lim_data['LIMd']

## ---------------------------------------------------
## RUN FORECAST WITH ALL MODES AS TEST: 
## ---------------------------------------------------

forecast_test = rf.run_forecast(LIMd_smode,exp_setup, save_folder, verbose=True, save=master_save, save_decomp=False)

#     forecast_validation = rf.validate_forecast_monthly(forecast, exp_setup['limvars'], 1, exp_setup, LIMd_smode, 
#                                                    save_folder, iplot=False, save=master_save)

forecast_validation_lags_test = rf.validate_forecast_lagged(forecast_test, exp_setup['limvars'], exp_setup, LIMd_smode, 
                                                            save_folder, iplot=False, save=master_save, 
                                                            detrend_truth=True)

today_date = today.strftime("%Y%m%d")
LIMcast = {}

LIMcast['LIMd'] = LIMd_smode
LIMcast['forecast'] = forecast_test
#     LIMcast['forecast_validation'] = forecast_validation
LIMcast['forecast_validation_lags'] = forecast_validation_lags_test

if save_decomp is False: 
    if 'x_forecast_dcomp' in LIMcast['forecast'].keys():
        LIMcast['forecast'].pop('x_forecast_dcomp')

start_yr = str(forecast_test['var_dict_valid']['sic']['time'][0])[0:4]
end_yr = str(forecast_test['var_dict_valid']['sic']['time'][-1])[0:4]

filename_end = (exp_setup['train_dsource']+'_002_ntrain'+exp_setup['mod_filename'][-22:-13]+
                '_validyrs_'+start_yr+'_'+end_yr+'_'+
                (str(exp_setup['ntrunc'])+"_").join(exp_setup['limvars'])+
                str(exp_setup['nmodes_sic'])+'_'+today_date+'.pkl')

print('saving in: '+save_folder+'LIMcast_allmodes_'+filename_end)
pickle.dump(LIMcast, open(save_folder+'LIMcast_allmodes_'+filename_end, "wb" ) )  


## ---------------------------------------------------
## RUN JACK KNIFE EXPERIMENTS: 
## ---------------------------------------------------
if jacknife is True: 
    for n in range(Nenms): 
        lim_data2 = pickle.load(open(data_dir+data_name, "rb" ) )

        LIMd_smode['vec'] = lim_data2['LIMd']['vec']
        LIMd_smode['veci'] = lim_data2['LIMd']['veci']
        LIMd_smode['lam_L'] = lim_data2['LIMd']['lam_L']

        LIMd_smode['vec'][:,n] = np.zeros((Nenms))
        LIMd_smode['veci'][n,:] = np.zeros((Nenms))
        LIMd_smode['lam_L'][n] = 0

        forecast = rf.run_forecast(LIMd_smode,exp_setup, save_folder, verbose=True, save=master_save, save_decomp=False)

    #     forecast_validation = rf.validate_forecast_monthly(forecast, exp_setup['limvars'], 1, exp_setup, LIMd_smode, 
    #                                                    save_folder, iplot=False, save=master_save)

        forecast_validation_lags = rf.validate_forecast_lagged(forecast, exp_setup['limvars'], exp_setup, LIMd_smode, 
                                                           save_folder, iplot=False, save=master_save, 
                                                           detrend_truth=True)    

        #--------------------------------------------------
        ### Save experiment: 
        #--------------------------------------------------

        today_date = today.strftime("%Y%m%d")
        LIMcast = {}

        LIMcast['LIMd'] = LIMd_smode
        LIMcast['forecast'] = forecast
    #     LIMcast['forecast_validation'] = forecast_validation
        LIMcast['forecast_validation_lags'] = forecast_validation_lags

        if save_decomp is False: 
            if 'x_forecast_dcomp' in LIMcast['forecast'].keys():
                LIMcast['forecast'].pop('x_forecast_dcomp')

        start_yr = str(forecast['var_dict_valid']['sic']['time'][0])[0:4]
        end_yr = str(forecast['var_dict_valid']['sic']['time'][-1])[0:4]

        filename_end = (exp_setup['train_dsource']+'_002_ntrain'+exp_setup['mod_filename'][-22:-13]+
                        '_validyrs_'+start_yr+'_'+end_yr+'_'+
                        (str(exp_setup['ntrunc'])+"_").join(exp_setup['limvars'])+
                        str(exp_setup['nmodes_sic'])+'_'+today_date+'.pkl')

        print('saving in: '+save_folder+'LIMcast_mode'+str(n)+'rm_'+filename_end)
        pickle.dump(LIMcast, open(save_folder+'LIMcast_mode'+str(n)+'rm_'+filename_end, "wb" ) )                                                  
