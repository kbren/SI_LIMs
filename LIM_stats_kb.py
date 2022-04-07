import numpy as np
import sys

sys.path.append("/home/disk/kalman2/mkb22/pyLMR/")
import LMR_utils

def global_mean(var, areacell): 
    """Assumes var is dimensions (nlat*nlon,time)
    """
    var_shape = len(var.shape)
    var_nan_mask = np.where(np.isnan(var),np.nan,1)
    
    if (var_shape<2)&(len(areacell.shape)>1): 
        areacell_1d = np.reshape(areacell,(areacell.shape[0]*areacell.shape[1]))
    elif (var_shape>1)&(len(areacell.shape)>1): 
        areacell_1d_temp = np.reshape(areacell,(areacell.shape[0]*areacell.shape[1]))
        areacell_1d = areacell_1d_temp[:,np.newaxis]
    elif (var_shape>1)&(len(areacell.shape)<=1): 
        areacell_1d = areacell[:,np.newaxis]
    else: 
        areacell_1d = areacell
        
    tot_nh_var = var*areacell_1d
    
    tot_var = np.nansum(tot_nh_var,axis=0)
    wt_sum = np.nansum(areacell_1d*var_nan_mask,axis=0)
    
    var_mn = tot_var/wt_sum
    
    return var_mn


def arctic_mean(var, areacell, cutoff=0.0): 
    if len(areacell.shape)>1:
        areacell_1d = np.reshape(areacell,(areacell.shape[0]*areacell.shape[1]))
    else: 
        areacell_1d = areacell
        
    tot_nh_var = var*areacell_1d
    
    if len(lat.shape)<=1:
        lat_inds = np.where(var_dict[var]['lat']>cutoff)
        tot_nh_var = np.nansum(np.nansum(tot_nh_var[:,lat_inds,:],axis=1),axis=1)
    
        wt_sum = np.nansum(np.nansum(cellarea[lat_inds,:],axis=0),axis=0)
    else:
        lat_inds = np.where(var_dict[var]['lat']>cutoff)
        tot_nh_var = np.nansum(np.nansum(tot_nh_var[:,lat_inds],axis=1),axis=1)
    
        wt_sum = np.nansum(np.nansum(cellarea[lat_inds],axis=0),axis=0)
    
    var_mn = tot_nh_var/wt_sum
    
    return var_mn


def calc_tot_si(var, areacell, units, lat, lat_cutoff=0.0): 
    if len(areacell.shape)>1:
        areacell_1d = np.reshape(areacell,(areacell.shape[0]*areacell.shape[1]))
    else: 
        areacell_1d = areacell
    
    if units == 'm2':
        cellarea = (areacell_1d*1e-6)[:,np.newaxis]
    else: 
        cellarea = areacell_1d[:,np.newaxis]
    
        
    if np.nanmax(var)>2:
        print('Dividing concentration by 100...')
        Var = var/100.0
    else: 
        Var = var
        
    nh_var = Var*cellarea
    
    if len(lat.shape)<=1:
        lat_inds = np.where(lat>lat_cutoff)
        tot_nh_var = np.nansum(nh_var[lat_inds,:].squeeze(),axis=0)
        
#     elif len(lat.shape)>=3:
#         lat_1d = np.reshape(lat[:,:,0],(var.shape[0]))
#         lat_inds = np.where(lat_1d>lat_cutoff)
#         tot_nh_var = np.nansum(nh_var[lat_inds,:].squeeze(),axis=0)
    else:
        lat_1d = np.reshape(lat,(var.shape[0]))
        lat_inds = np.where(lat_1d>lat_cutoff)
        tot_nh_var = np.nansum(nh_var[lat_inds,:].squeeze(),axis=0)
    
    return tot_nh_var

def calc_tot_si_checks(var, areacell, units, lat, nlon, lat_cutoff=0.0): 
    """Calculates total NH sea ice area. 
    
    INPUTS: 
    var: sea ice concentration (ndof,ntime), (ndarray)
    areacell: grid cell area, 1D or 2D, (ndarray)
    units: string ('km^2' or 'm2')
    lat: 
    lat_cutoff
    
    OUTPUTS: 
    tot_nh_var
    """
    
    if len(areacell.shape)>1:
        areacell_1d = np.reshape(areacell,(areacell.shape[0]*areacell.shape[1]))
    else: 
        areacell_1d = areacell

    if units == 'm2':
        cellarea = (areacell_1d*1e-6)[:,np.newaxis]
    else: 
        cellarea = areacell_1d[:,np.newaxis]

    if len(var.shape)<=1:
        var_3d = np.reshape(var,(lat.shape[0],nlon))
    elif len(var.shape) ==2:     
        var_3d = np.reshape(var,(lat.shape[0],nlon,var.shape[1]))
    else: 
        var_3d = var

    if len(lat.shape)<=1:
        var_nh_3d = var_3d[(lat>0),:,:]
        test_var = var_nh_3d[np.isfinite(var_nh_3d)]
    elif len(lat.shape)>1:
        var_nh_3d = var_3d[(lat>0),:]
        test_var = var_nh_3d[np.isfinite(var_nh_3d)]

    if np.nanmax(test_var)>5:
        print('Max concentration is '+str(np.round(np.nanmax(test_var),2))+
              ' ...dividing concentration by 100.')
        Var = var/100.0
    else: 
        Var = var

    nh_var = Var*cellarea

    if len(lat.shape)<=1:
        if len(var.shape)<=1:
            nh_var_3d = np.reshape(nh_var,(lat.shape[0],nlon))
            lat_inds = np.where(lat>lat_cutoff)
            tot_nh_var = np.nansum(np.nansum(nh_var_3d[lat_inds,:].squeeze(),axis=0),axis=0)
        else:     
            nh_var_3d = np.reshape(nh_var,(lat.shape[0],nlon,var.shape[1]))
            lat_inds = np.where(lat>lat_cutoff)
            tot_nh_var = np.nansum(np.nansum(nh_var_3d[lat_inds,:,:].squeeze(),axis=0),axis=0)
    else:
        lat_1d = np.reshape(lat,(var.shape[0]))
        lat_inds = np.where(lat_1d>lat_cutoff)
        tot_nh_var = np.nansum(nh_var[lat_inds,:].squeeze(),axis=0)
        
    return tot_nh_var


def calc_gm_polar_variance(valid_var,valid_var_mon,fields,lat,lon):
    valid_variance = {}
    gm_mon = np.zeros((12))
    polar_mon = np.zeros((2,12))
    nlat = lat.shape[0]
    nlon = lon.shape[0]

    for v in fields.keys():
        print(v)
        var = np.reshape(valid_var[fields[v]],[nlat,nlon])
        var_mon = np.reshape(valid_var_mon[fields[v],:].T,[12,nlat,nlon])

        gm,_,_ = LMR_utils.global_hemispheric_means(var,lat)
        polar = polar_regional_means(var,lat,lon)
        polar_mon = polar_regional_means(var_mon,lat,lon)
        for m in range(12):
            gm_mon[m],_,_ = LMR_utils.global_hemispheric_means(var_mon[m,:,:],lat)

        valid_variance[v+'_gm'] = gm
        valid_variance[v+'_gm_mon'] = gm_mon
        valid_variance[v+'_polarm'] = polar
        valid_variance[v+'_polarm_mon'] = polar_mon
        
    return valid_variance


def calc_linear_fit(X,Y):
    X_anom = X - X.mean()
    Y_anom = Y - Y.mean()
    
    cov = np.dot(X_anom, Y_anom)/(len(Y_anom))#-1)
    var = np.var(X)
    
    slope = cov/var
    intercept = Y.mean() - slope*X.mean()
    
    return slope, intercept 


def linear_detrend(X,Y, atol=False, remove_mn=False):
    slope, intercept = calc_linear_fit(X,Y)
    if atol is False: 
        slope_tol = slope
        intercept_tol = intercept
    else:
        slope_tol = np.where(np.abs(slope)>=atol,slope,0)
        intercept_tol = np.where(np.abs(intercept)>=atol,intercept,0) 

        
    lin_fit = X*slope_tol + intercept_tol 
    Y_dt = Y - lin_fit

    if remove_mn is False: 
        Y_dt_out = Y_dt + intercept_tol
    else: 
        Y_dt_out = Y_dt 
        
    return Y_dt_out, slope_tol, intercept_tol 


def multi_linear_detrend(X,Y,axis=1, atol=False,remove_mn=False):
    """
    axis = dimension which to detrend over. 
    """
#    print('atol = '+str(atol))
#    print(Y.shape)
    if len(Y.shape)>2:
        raise ValueError('Too many dimensions in Y.')
    elif len(Y.shape)<2:
        Y_dt_out, slopes, intercepts = linear_detrend(X,Y,atol=atol,remove_mn=remove_mn)
        
        return Y_dt_out, slopes, intercepts 
    else: 
        Y_dt_out = np.zeros_like(Y)
        slopes = np.zeros(Y.shape[0])
        intercepts = np.zeros(Y.shape[0])
        
        if axis==1: 
            dt_axis=0
        else: 
            dt_axis=1

        for i in range(Y.shape[dt_axis]):
            Y_dt_out[i,:], slopes[i], intercepts[i] = linear_detrend(X,Y[i,:],atol=atol,
                                                                     remove_mn=remove_mn)
            
        return Y_dt_out, slopes, intercepts
    
    
def calc_lac(fcast, obs):
    """
    Taken from pyLIM
    Method to calculate the Local Anomaly Correlation (LAC).  Uses numexpr
    for speed over larger datasets.
    Note: If necessary (memory concerns) in the future, the numexpr statements
    can be extended to use pytable arrays.  Would need to provide means to
    function, as summing over the dataset is still very slow it seems.
    Parameters
    ----------
    fcast: ndarray
        Time series of forecast data. M x N where M is the temporal dimension.
    obs: ndarray
        Time series of observations. M x N
    Returns
    -------
    lac: ndarray
        Local anomaly correlations for all locations over the time range.
    """
    # Calculate means of data
    f_mean = fcast.mean(axis=0)
    o_mean = obs.mean(axis=0)
    f_anom = fcast - f_mean
    o_anom = obs - o_mean

    # Calculate covariance between time series at each gridpoint
    cov = (f_anom * o_anom).sum(axis=0)

    # Calculate standardization terms
    f_std = (f_anom**2).sum(axis=0)
    o_std = (o_anom**2).sum(axis=0)
    
    f_std = np.sqrt(f_std)
    o_std = np.sqrt(o_std)

    std = f_std * o_std
    lac = cov / std

    return lac

def red_noise_fit_ar1(data, lead=1):
    lag1_autocorr = calc_lac(data[:-lead], data[lead:])

    white_noise_var = (1 - lag1_autocorr ** 2) * data.var(ddof=1, axis=0)

    return lag1_autocorr, white_noise_var

def red_noise_forecast_ar1(training_data, forecast_data, lead=1):
    print('Lead = '+str(lead))
    ar1_factor, noise_var = red_noise_fit_ar1(training_data, lead=lead)

    forecast = forecast_data[:-lead] * ar1_factor

    return forecast, ar1_factor