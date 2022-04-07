"""
Tools for Linear Inverse Modelling 

Started October 2020 
Many functions taken from Greg Hakim's notebook (LIM_testing.ipynb)
"""

from netCDF4 import Dataset, date2num, num2date
import numpy as np
import ESMF
import sys
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

import time as timestamp

sys.path.append("/home/disk/kalman2/mkb22/pyLMR/")
import LMR_utils

def load_data(var_to_extract,infile):
    
    """This function does all of the gridded data loading and processing"""
    
    print('fetching ',var_to_extract,' from ',infile)
    
    # open the file
    data = Dataset(infile,'r')
    
    # this block hacked from load_gridded_data

    # Dimensions used to store the data
    nc_dims = [dim for dim in data.dimensions]
    dictdims = {}
    for dim in nc_dims:
        dictdims[dim] = len(data.dimensions[dim])

    # Query its dimensions
    vardims = data.variables[var_to_extract].dimensions
    nbdims  = len(vardims)
    # names of variable dims
    vardimnames = []
    for d in vardims:
        vardimnames.append(d)

    # put everything in lower case for homogeneity
    vardimnames = [item.lower() for item in vardimnames]

    # extract info on variable units
#    units = data.variables[var_to_extract].units

    # time variable
    time = data.variables['time']

    # Transform into calendar dates using netCDF4 variable attributes (units & calendar)
    try:
        if hasattr(time, 'calendar'):
            # if time is defined as "months since":not handled by datetime functions
            if 'months since' in time.units:
                new_time = np.zeros(time.shape)
                nmonths, = time.shape
                basedate = time.units.split('since')[1].lstrip()
                new_time_units = "days since "+basedate        
                start_date = pl.datestr2num(basedate)        
                act_date = start_date*1.0
                new_time[0] = act_date
                for i in range(int(nmonths)): #increment months
                    d = pl.num2date(act_date)
                    ndays = monthrange(d.year,d.month)[1] #number of days in current month
                    act_date += ndays
                    new_time[i] = act_date

                time_yrs = num2date(new_time[:],units=new_time_units,
                                    calendar=time.calendar)
            else:                    
                time_yrs = num2date(time[:],units=time.units,
                                calendar=time.calendar)
        else:
            time_yrs = num2date(time[:],units=time.units)
        time_yrs_list = time_yrs.tolist()
        
    except ValueError:
        # num2date needs calendar year start >= 0001 C.E. (bug submitted
        # to unidata about this
        fmt = '%Y-%d-%m %H:%M:%S'
        tunits = time.units
        since_yr_idx = tunits.index('since ') + 6
        year = int(tunits[since_yr_idx:since_yr_idx+4])
        year_diff = year - 1
        new_start_date = datetime(1, 1, 1, 0, 0, 0)

        new_units = tunits[:since_yr_idx] + '0001-01-01 00:00:00'
        if hasattr(time, 'calendar'):
            time_yrs = num2date(time[:], new_units, calendar=time.calendar)
        else:
            time_yrs = num2date(time[:], new_units)

        time_yrs_list = [datetime(d.year + year_diff, d.month, d.day,
                                  d.hour, d.minute, d.second)
                         for d in time_yrs]
        
    # this loads the data
    begin_time = timestamp.time()
    data_var = data.variables[var_to_extract][:]
    elapsed_time = timestamp.time() - begin_time
    print('-----------------------------------------------------')
    print('completed in ' + str(elapsed_time) + ' seconds')
    print('-----------------------------------------------------')
    
    return data, data_var, vardimnames, time_yrs

def set_coord_names(data, data_var, vardimnames, var_to_extract):
    # coordinate setup
    varspacecoordnames = [item for item in vardimnames if item != 'time'] 
    varspacecoordnames = [item for item in varspacecoordnames if item != 'plev'] 
    spacecoords = (varspacecoordnames[0],varspacecoordnames[1])
    spacevar1 = data.variables[spacecoords[0]][:]
    spacevar2 = data.variables[spacecoords[1]][:]
    vardims = data_var.shape
    print(vardims)

    # which dim is lat & which is lon?
    indlat = spacecoords.index('lat')
    indlon = spacecoords.index('lon')
    print('indlat=', indlat, ' indlon=', indlon)

    ntime = len(data.dimensions['time'])

    print(spacecoords)

    if var_to_extract in ['tos','ohc']:
        lat_2d_orig = spacevar1
        lon_2d_orig = spacevar2
    elif len(spacevar1.shape)>1:
        lat_2d_orig = spacevar1
        lon_2d_orig = spacevar2
    else:
        lat_2d_orig = spacevar1[:,np.newaxis]*np.ones(spacevar2[:,np.newaxis].shape[0])
        lon_2d_orig = np.ones([spacevar1[:,np.newaxis].shape[0],1])*spacevar2
    
    return ntime, lat_2d_orig, lon_2d_orig

def regrid_data(data_var, ntime, lat_2d_orig, lon_2d_orig, nlat, nlon):
    #----- regrid the data to lower resolution
    begin_time = timestamp.time()
    
    nlat_orig = lat_2d_orig.shape[0]
    nlon_orig = lon_2d_orig.shape[1]
    print('nlat:',nlat_orig)
    print('nlon:',nlon_orig)

    datax = np.reshape(data_var,[ntime,nlat_orig*nlon_orig])
    # shift to (nx,nens) shaping for regridding function
    tmp = np.moveaxis(datax,0,-1)
    print(tmp.shape)

    # this is the new grid
#     nlat = 45
#     nlon = 72
    ndof = nlat*nlon
    nens = ntime

    # regrid using LMR regrid function, which uses ESMpy
    [data_new,
     lat_2d,
     lon_2d] = LMR_utils.regrid_esmpy(nlat,
                                       nlon,
                                       nens,
                                       tmp,
                                       lat_2d_orig,
                                       lon_2d_orig,
                                       nlat_orig,
                                       nlon_orig)

    elapsed_time = timestamp.time() - begin_time
    print('-----------------------------------------------------')
    print('completed in ' + str(elapsed_time) + ' seconds')
    print('-----------------------------------------------------')
    
    return data_new, lat_2d, lon_2d

def compute_climo(data_new, time_yrs, ntime, nlat, nlon): 
    # compute monthly average and remove from the original data
    dates = time_yrs

    # shape back back to (ntime,nlat,nlon)
    data_new2 = np.reshape(np.moveaxis(data_new,0,-1),[ntime,nlat,nlon])
    climo_month = np.zeros([12,nlat,nlon], dtype=float)
    print('Removing climo from:')
    for i in range(12):
        print('month = '+str(i))
        m = i+1
        indsm = [j for j,v in enumerate(dates) if v.month == m]
        indsmref = indsm

        climo_month[i] = np.nanmean(data_new2[indsmref], axis=0)
        data_new2[indsm] = (data_new2[indsm] - climo_month[i])
        # NEW---option to standardize monthly anomalies
        mostd = np.std(data_new2[indsm],axis=0,ddof=1)
        data_new2[indsm] = data_new2[indsm]/mostd

        # GH: make an array with month and year at some point?
    
    return data_new2

def get_data(var_to_extract,infile, nlat, nlon, regrid=True):
    
    data, data_var, vardimnames, time_yrs = load_data(var_to_extract,infile)
    ntime, lat_2d_orig, lon_2d_orig = set_coord_names(data, data_var, 
                                                      vardimnames, var_to_extract)
    if regrid is True: 
        print('regridding data...')
        data_new, lat_2d, lon_2d = regrid_data(data_var, ntime, 
                                               lat_2d_orig, lon_2d_orig, 
                                               nlat, nlon)
    else: 
        print('skipping regridding...')
        data_new = data_var
        lat_2d = lat_2d_orig
        lon_2d = lat_2d_orig
        nlat = lat_2d.shape[0]
        nlon = lat_2d.shape[1]

    data_new2 = compute_climo(data_new, time_yrs, ntime, nlat, nlon)
    
    return data_new2,lat_2d,lon_2d, time_yrs

def LIM_train(tau,x_train):
    """
    train a LIM, L, given a training dataset
    
    Inputs:
    * tau: the training lag time (unitless) in time steps 
      defined by the format of x_train
    * x_train: ~(nx,nt) state-time matrix
    
    Outputs:
    * LIMd: a dictionary containing the left eigenvectors
      of L, their inverse, and the eigenvalues of L
    
    """
    nx = x_train.shape[0]
    nt = x_train.shape[1]
    
    # estimate of zero-lag covariance
    C_0 = np.matmul(x_train,x_train.T)/(nt-1)

    # lag-tau covariance
    C_1 = np.matmul(x_train[:,tau:],x_train[:,:-tau].T)/(nt-1)

    # lag-tau resolvant
    G = np.matmul(C_1,np.linalg.inv(C_0))

    # solve for L from G
    val,vec = np.linalg.eig(G)
    veci = np.linalg.inv(vec)
    lam_L = np.log(val)/tau
    
    # make a dictionary with the results
    LIMd = {}
    LIMd['vec'] = vec
    LIMd['veci'] = veci
    LIMd['val'] = val
    LIMd['lam_L'] = lam_L
    LIMd['Gt']= G
    
    return LIMd, G


def LIM_train_flex(tau,x_train_t0, x_train_t1):
    """
    train a LIM, L, given a training dataset
    
    Inputs:
    * tau: the training lag time (unitless) in time steps 
      defined by the format of x_train
    * x_train: ~(nx,nt) state-time matrix
    
    Outputs:
    * LIMd: a dictionary containing the left eigenvectors
      of L, their inverse, and the eigenvalues of L
    
    """
    nx = x_train_t1.shape[0]
    nt1 = x_train_t1.shape[1]
    nt0 = x_train_t0.shape[1]
    
    # estimate of zero-lag covariance
    C_0 = np.matmul(x_train_t0,x_train_t0.T)/(nt0-1)

    # lag-tau covariance
    C_1 = np.matmul(x_train_t1,x_train_t0[:,:nt1].T)/(nt1-1)

    # lag-tau resolvant
    G = np.matmul(C_1,np.linalg.inv(C_0))

    # solve for L from G
    val,vec = np.linalg.eig(G)
    veci = np.linalg.inv(vec)
    lam_L = np.log(val)/tau
    
    # make a dictionary with the results
    LIMd = {}
    LIMd['vec'] = vec
    LIMd['veci'] = veci
    LIMd['val'] = val
    LIMd['lam_L'] = lam_L
    LIMd['Gt']= G
    
    return LIMd, G

def LIM_forecast_test(G,x,lags,E,truth,yrs_valid,nvar,nlalo,
                      nmodes=None,nmodes_sic=None,E_sic=None,
                      sic_separate=False):
    """
    deterministic forecasting experiments for states in x and time lags in lags.

    Inputs:
    * LIMd: a dictionary with LIM attributes
    * x: a state-time matrix for initial conditions and verification ~(ndof,ntims)
    * lags: list of time lags for deterministic forecasts
    * E: the linear map from the coordinates of the LIM to physical (lat,lon) coordinates ~(nx*ny,ndof)
    
    Outputs (in a dictionary):
    *'error' - error variance as a function of space and forecast lead time (ndof,ntims)
    *'x_forecast' - the forecast states (nlags,ndof,ntims)
    *'x_truth_phys_space' - true state in physical space (nlat*nlon,*ntims)
    *'x_forecast_phys_space' - forecast state in physical space (nlat*nlon,*ntims)
    """
    
    ndof = x.shape[0]
    ntims = x.shape[1]
    nlags = len(lags)
    nx = E.shape[0]
    
    error = np.ones([nvar*nlalo,nlags])*np.nan
    x_predict_save = np.ones([nlags,ndof,ntims])*np.nan
    time_forecast = np.tile(yrs_valid,(13,1))
    
    for k,t in enumerate(lags):
        print('t=',t)       
        
        Gt = np.linalg.matrix_power(G,t)
        # forecast
        if t == 0:
            # need to handle this time separately, or the matrix dimension is off
            x_predict = np.matmul(Gt,x)
            x_predict_save[k,:,:] = x_predict
        else:
            x_predict = np.matmul(Gt,x[:,:-t])
            x_predict_save[k,:,t:] = x_predict

        # physical-space fields for forecast and truth 
        # for this forecast lead time ~(ndof,ntims)
        if sic_separate is True: 
            X_predict = decompress_eof_separate_sic(x_predict,nmodes,nmodes_sic,E,E_sic)
        else: 
            X_predict = np.real(np.matmul(E,x_predict))
        
        #X_truth = np.real(np.matmul(E,x[:,t:]))
        X_truth = truth[:,t:]
        time_forecast[k,0:t] = np.ones((t))*np.nan
       
        # error variance as a function of space and forecast lead time ~(ndof,ntims)
        error[:,k] = np.var(X_predict - X_truth,axis=1,ddof=1)
    
    # return the LIM forecast error dictionary
    LIMfd = {}
    LIMfd['error'] = error
    LIMfd['x_forecast'] = x_predict_save
    LIMfd['time_forecast'] = time_forecast
        
    return LIMfd

def LIM_forecast(LIMd,x,lags,E,truth):
    """
    deterministic forecasting experiments for states in x and time lags in lags.

    Inputs:
    * LIMd: a dictionary with LIM attributes
    * x: a state-time matrix for initial conditions and verification ~(ndof,ntims)
    * lags: list of time lags for deterministic forecasts
    * E: the linear map from the coordinates of the LIM to physical (lat,lon) coordinates ~(nx*ny,ndof)
    
    Outputs (in a dictionary):
    *'error' - error variance as a function of space and forecast lead time (ndof,ntims)
    *'x_forecast' - the forecast states (nlags,ndof,ntims)
    *'x_truth_phys_space' - true state in physical space (nlat*nlon,*ntims)
    *'x_forecast_phys_space' - forecast state in physical space (nlat*nlon,*ntims)
    """
    
    ndof = x.shape[0]
    ntims = x.shape[1]
    nlags = len(lags)
    nx = E.shape[0]
    
    error = np.zeros([nx,nlags])
    x_predict_save = np.zeros([nlags,ndof,ntims])
    
    k = -1
    for t in lags:
        k+=1
        print('t=',t)
        # make the propagator for this lead time
        Gt = np.matmul(np.matmul(LIMd['vec'],np.diag(np.exp(LIMd['lam']*t))),LIMd['veci'])
        
        # forecast
        if t == 0:
            # need to handle this time separately, or the matrix dimension is off
            x_predict = np.matmul(Gt,x)
            x_predict_save[k,:,:] = x_predict
        else:
            x_predict = np.matmul(Gt,x[:,:-t])
            x_predict_save[k,:,:-t] = x_predict

        # physical-space fields for forecast and truth for this forecast lead time ~(ndof,ntims)
        X_predict = np.real(np.matmul(E,x_predict))
        #X_truth = np.real(np.matmul(E,x[:,t:]))
        X_truth = truth[:,t:]
        
        # error variance as a function of space and forecast lead time ~(ndof,ntims)
        error[:,k] = np.var(X_predict - X_truth,axis=1,ddof=1)
    
        # return the LIM forecast error dictionary
        LIMfd = {}
        LIMfd['error'] = error
        LIMfd['x_forecast'] = x_predict_save
        LIMfd['Gt'] = Gt
        
    return LIMfd

def calc_forecast_error_by_mon(forecast,lag, truth, E, yrs_forecast, nvar, ndof,
                               E_sic=None,nmodes=None,sic_separate=False): 
    """
    forecast: (lag,eof,time)
    lag: integer value representing lag of interest
    """
    x_predict = forecast[lag,:,:]
    time = yrs_forecast[lag,:]
    if sic_separate is True: 
        X_predict = decompress_eof_separate_sic(x_predict,nmodes,E,E_sic)
    else: 
        X_predict = np.real(np.matmul(E,x_predict))
    
    ntime = x_predict.shape[1]
    nyears = int(ntime/12)
    nlalo = nvar*ndof

    error = X_predict-truth
    error_rs = np.reshape(error,(nlalo,nyears,12))
    time_rs = np.reshape(time,(nyears,12))
    
    error_mon = np.nanvar(error_rs, axis=1,ddof=1)
    rmse = np.sqrt(np.nanmean(error_rs**2,axis=1))
    
    return error_rs, error_mon, rmse, time_rs

# def calc_forecast_error_by_mon(forecast,lag, truth, E, yrs_forecast): 
#     """
#     forecast: (lag,eof,time)
#     lag: integer value representing lag of interest
#     """
#     x_predict = forecast[lag,:,:]
#     time = yrs_forecast[lag,:]
#     X_predict = np.real(np.matmul(E,x_predict))
    
#     ntime = x_predict.shape[1]
#     nyears = int(np.floor(ntime/12))
#     nlalo = E.shape[0]

#     error = X_predict-truth[:,lag:]
#     error_rs = np.reshape(error[:,:nyears*12],(nlalo,12,nyears))
#     time_rs = np.reshape(time[:nyears*12],(nyears,12))
    
#     error_mon = np.var(error_rs, axis=2,ddof=1)
#     rmse = np.nanmean(error_rs**2,axis=2)
    
#     return error_rs, error_mon, rmse, time_rs

def plot_map_vector(vec,lat,lon,minv=-1,maxv=-1,noland=False,cmap='bwr'):
    nlat = lat.shape[0]
    nlon = lon.shape[1]
    pdat = np.reshape(vec,[nlat,nlon])
    pdat_wrap, lon_wrap = add_cyclic_point(pdat,coord=lon[0,:], axis=1)
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.0),zorder=1)
    ax.coastlines()
    if maxv == -1:
        maxv = np.nanmax(vec)
    if minv == -1:
        minv = -maxv
    cs = ax.pcolormesh(lon_wrap,lat[:,0],pdat_wrap,transform=ccrs.PlateCarree(),
                       cmap=cmap,shading='flat',vmin=minv,vmax=maxv)
    plt.colorbar(cs, extend='both', shrink=0.6)
    if noland:
        ax.add_feature(cartopy.feature.LAND, zorder=2, edgecolor='black',facecolor='black')
    return

def euler_twostep(L,Qsr,dt,y_old):
    # stochastic integration scheme
    n = L.shape[0]
    # this has unit variance
    noise = np.random.randn(n,1)
    y_new = y_old + dt*np.matmul(L,y_old) + np.sqrt(dt)*np.matmul(Qsr,noise)
    x_new = 0.5*(y_new + y_old)
    return x_new,y_new

def calc_eof(var, lat_2d):
    # EOFs
    print('computing EOFs...')
    begin_time = timestamp.time()
    
    nlat = lat_2d.shape[0]
    nlon = lat_2d.shape[1]

    # weight matrix for equal-area covariance normalization
    tmp = np.sqrt(np.cos(np.radians(lat_2d)))
    W = np.reshape(tmp,[nlat*nlon,1])
    # note W*X = X*W

    # EOFs via SVD
    u,s,v = np.linalg.svd(W*var,full_matrices=False)

    elapsed_time = timestamp.time() - begin_time
    print('-----------------------------------------------------')
    print('completed in ' + str(elapsed_time) + ' seconds')
    print('-----------------------------------------------------')

    print(u.shape,s.shape,v.shape)
    
    return u,s,v,W 

def plot_eigenvalues(s,nmodes,var_to_extract='other'):
    if var_to_extract == 'tos':
        # first tos EOF is a regridding artifact; rest look OK
        fi = 1
    else:
        fi = 0
    fvar = 100*s[fi:]*s[fi:]/np.sum(s[fi:]*s[fi:])

    plt.plot(fvar[0:nmodes],'ko-')
    plt.title('EOF fraction of variance')
    plt.ylabel('Percent of variance (%)')
    
    print(np.sum(fvar))
    print('fraction in first ',nmodes,' EOFs = ',np.sum(fvar[:nmodes]))
      
def subplot_map_vector(ax,fig,vec,lat,lon,minv=-1,maxv=-1,noland=False,cmap='bwr'):
    nlat = lat.shape[0]
    nlon = lon.shape[1]
    pdat = np.reshape(vec,[nlat,nlon])
    pdat_wrap, lon_wrap = add_cyclic_point(pdat,coord=lon[0,:], axis=1)
    
    ax.coastlines()
    if maxv == -1:
        maxv = np.nanmax(vec)
    if minv == -1:
        minv = -maxv
    cs = ax.pcolormesh(lon_wrap,lat[:,0],pdat_wrap,transform=ccrs.PlateCarree(),
                       cmap=cmap,shading='flat',vmin=minv,vmax=maxv)
    plt.colorbar(cs, ax=ax, extend='both')
    if noland:
        ax.add_feature(cartopy.feature.LAND, zorder=2, edgecolor='black',facecolor='black')
    return

def sub_arctic_plot(ax,fig,vec,lat,lon,maxv=-1,
                    minv=-1,colorbar=True,extent=True,cmap='bwr'):
    nlat = lat.shape[0]
    nlon = lon.shape[1]
    pdat = np.reshape(vec,[nlat,nlon])
                    
    if maxv == -1:
        maxv = np.nanmax(vec)
    if minv == -1:
        minv = -maxv                
           
    pdat_wrap, lon_wrap = add_cyclic_point(pdat,coord=lon[0,:], axis=1)
#    new_lon2d, new_lat2d = np.meshgrid(lon_wrap, lat)
                    
    if extent is True: 
        ax.set_extent([-150, 140, 50, 90], crs=ccrs.PlateCarree())
    ax.gridlines(linestyle='--')
    ax.add_feature(cfeature.LAND, facecolor=(1, 1, 1))
    cs = ax.pcolormesh(lon_wrap, lat[:,0], pdat_wrap, 
                       vmin=minv, vmax=maxv, cmap=cmap, 
                       transform=ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.5)
    if colorbar is True:
        plt.colorbar(cs, ax=ax)
        
    return 

def build_training_dic(dsource):
    if dsource == 'ccsm4_lm':
        mod = 'CCSM4'
        mod_dir = '/home/disk/kalman3/rtardif/LMR/data/model/ccsm4_last_millenium/'
        
        infile_sic = mod_dir+'sic_sfc_OImon_'+mod+'_past1000_085001-185012.nc'
        infile_tos = mod_dir+'tos_sfc_Omon_'+mod+'_past1000_085001-185012.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_pr = mod_dir+'pr_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_rlut = mod_dir+'rlut_toa_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_psl = mod_dir+'psl_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_sit = mod_dir+'sit_noMV_OImon_'+mod+'_past1000_085001-185012.nc'
        
    elif dsource == 'mpi_lm':
        mod = 'MPI-ESM-P'
        mod_dir_wp = '/home/disk/katabatic/wperkins/data/LMR/data/model/mpi-esm-p_last_millenium/'
        mod_dir = '/home/disk/kalman3/rtardif/LMR/data/model/mpi-esm-p_last_millenium/'
        
        infile_sic = mod_dir+'sic_sfc_OImon_'+mod+'_past1000_085001-185012.nc'
        infile_tos = mod_dir_wp+'tos_sfc_Omon_'+mod+'_past1000_085001-184912.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_pr = mod_dir+'pr_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_rlut = mod_dir+'rlut_toa_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_psl = mod_dir+'psl_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_sit = mod_dir+'sit_sfc_OImon_'+mod+'_past1000_085001-185012.nc'
    
    fdic = {'fpath':mod_dir,
            'tos':infile_tos,
            'tas':infile_tas,
            'zg':infile_zg,
            'rlut':infile_rlut,
            'sic':infile_sic,
            'psl':infile_psl,
            'pr':infile_pr,
            'sit':infile_sit}
        
    return fdic

def LIM_forecast_error(LIMfd,truth,E,lags,fields,lat_2d,
                       E_sic=None,nmodes=None,sic_separate=False):
    """
    verify LIM forecasts against a truth state. extracted from 
    LIM_forecast to verify against different true states (full and EOF).
    
    truth must be in grid point space
    
    finds: dictionary with start:end indices for individual fields
    """
    ntims = LIMfd['x_forecast'].shape[2]
    ntims_truth = truth.shape[1]
    nlat = lat_2d.shape[0]
    nlon = lat_2d.shape[1]
    
    if ntims != ntims_truth:
        print('truth must have same number of times as the forecast')
        return None
    
    nlags = len(lags)
    nx = truth.shape[0]
    # number of fields to verify global means values
    nfields = len(fields.keys())
    
    error = np.zeros([nx,nlags])
    error_gm = np.zeros([nfields,nlags])

    for k,t in enumerate(lags):
        print('lag = ',t)
        # physical-space fields for forecast and 
        # truth for this forecast lead time ~(ndof,ntims)
        if t == 0:
            # need to handle this time separately, or the matrix dimension is off
            if sic_separate is True: 
                X_predict = decompress_eof_separate_sic(LIMfd['x_forecast'][t,:,:],
                                                        nmodes,E,E_sic)
                X_truth = truth[:,:]
            else: 
                X_predict = np.real(np.matmul(E,LIMfd['x_forecast'][t,:,:]))
                X_truth = truth[:,:]
        else:
            if sic_separate is True: 
                X_predict = decompress_eof_separate_sic(LIMfd['x_forecast'][t,:,t:],nmodes,E,E_sic)
                X_truth = truth[:,t:]
            else: 
                X_predict = np.real(np.matmul(E,LIMfd['x_forecast'][t,:,t:]))
                X_truth = truth[:,t:]

        #print(X_predict.shape)
        #print(X_truth.shape)
        
        # error variance as a function of space and forecast lead time ~(ndof,ntims)
        error[:,k] = np.var(X_predict - X_truth,axis=1,ddof=1)
        
    # global mean for each field
    for n,field in enumerate(fields.keys()):
        finds = fields[field]
        hold = np.moveaxis(np.reshape(error[finds,:],[nlat,nlon,nlags]),-1,0)
        error_gm[n,:],_,_ = LMR_utils.global_hemispheric_means(hold,lat_2d[:,0])

    return error,error_gm

def fix_order_forecasts(var): 
    var_fixed = np.zeros(12)
    
    var_fixed[1:] = var[:-1]
    var_fixed[0] = var[-1]
    
    return var_fixed

def polar_regional_means(field,lat,lon,debug=False):

    # number of geographical regions (default, as defined in PAGES2K(2013) paper
    nregions = 2

    # set number of times, lats, lons; array indices for lat and lon    
    if len(np.shape(field)) == 3: # time is a dimension
        ntime,nlat,nlon = np.shape(field)
    else: # only spatial dims
        ntime = 1
        nlat,nlon = np.shape(field)
        field = field[None,:] # add time dim of size 1 for consistent array dims

    if debug:
        print('field dimensions...')
        print(np.shape(field))

    # define regions as in PAGES paper

    # lat and lon range for each region 
    # (first value is lower limit, second is upper limit)
    rlat = np.zeros([nregions,2]); rlon = np.zeros([nregions,2])
    # 1. Arctic: north of 60N 
    rlat[0,0] = 60.; rlat[0,1] = 90.
    rlon[0,0] = 0.; rlon[0,1] = 360.
    # 7. Antarctica: south of 60S (from map)
    rlat[1,0] = -90.; rlat[1,1] = -60.
    rlon[1,0] = 0.; rlon[1,1] = 360.
    # ...add other regions here...

    # latitude weighting 
    lat_weight = np.cos(np.deg2rad(lat))
    tmp = np.ones([nlon,nlat])
    W = np.multiply(lat_weight,tmp).T

    rm  = np.zeros([nregions,ntime])

    # loop over regions
    for region in range(nregions):

        if debug:
            print('region='+str(region))
            print(rlat[region,0],rlat[region,1],rlon[region,0],rlon[region,1])

        # regional weighting (ones in region; zeros outside)
        mask = LMR_utils.regional_mask(lat,lon,rlat[region,0],
                                       rlat[region,1],rlon[region,0],
                                       rlon[region,1])
        if debug:
            print('mask=')
            print(mask)

        # this is the weight mask for the regional domain    
        Wmask = np.multiply(mask,W)

        # make sure data starts at South Pole
        if lat[0] > 0:
            # data has NH -> SH format; reverse
            field = np.flipud(field)

        # Check for valid (non-NAN) values & use numpy average function 
        # (includes weighted avg calculation) 
        # Get arrays indices of valid values
        indok    = np.isfinite(field)
        for t in range(ntime):
            indok_2d = indok[t,:,:]
            field_2d = np.squeeze(field[t,:,:])
            if debug: 
                print('sum of regional weights = '+str(Wmask[indok_2d].sum()))
                
            if np.max(Wmask) >0.:
                rm[region,t] = np.average(field_2d[indok_2d],
                                          weights=Wmask[indok_2d])
            else:
                rm[region,t] = np.nan
    return rm

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

def decompress_eof_separate_sic(x_train,nmodes,nmodes_sic,E,E_sic):
    x_multivar = x_train[0:nmodes,:]
    x_sic = x_train[-nmodes_sic:,:]

    x_train_multi_dcomp = np.matmul(E,x_multivar)
    x_train_sic_dcomp = np.matmul(E_sic,x_sic)
    x_train_dcomp = np.concatenate((x_train_multi_dcomp,x_train_sic_dcomp),axis=0)
    
    return x_train_dcomp

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
        Local anomaly corellations for all locations over the time range.
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

    ar1_factor, noise_var = red_noise_fit_ar1(training_data, lead=lead)

    forecast = forecast_data[:-lead] * ar1_factor

    return forecast, ar1_factor

def calc_ce_corr_lags_gm_pm(truth_state,forecast,limvars,lags):
    ce = {}
    corr = {}

    for n,var in enumerate(limvars):
        print('working on '+str(var))
        ce_gm_lag = np.zeros((len(lags)))
        ce_pm_lag = np.zeros((len(lags),2))
        corr_gm_lag = np.zeros((len(lags)))
        corr_pm_lag = np.zeros((len(lags),2))

        for k,l in enumerate(lags):
            ce_gm_lag[k] = LMR_utils.coefficient_efficiency(truth_state[var+'_gm'][k,l:],
                                                            forecast[var+'_gm'][k,l:])
            corr_gm_lag[k] = np.corrcoef(truth_state[var+'_gm'][k,l:],
                                         forecast[var+'_gm'][k,l:])[0,1]

            for n in range(2):
                ce_pm_lag[k,n] = LMR_utils.coefficient_efficiency(truth_state[var+'_pm'][k,n,l:],
                                                                  forecast[var+'_pm'][k,n,l:])
                corr_pm_lag[k,n] = np.corrcoef(truth_state[var+'_pm'][k,n,l:],
                                               forecast[var+'_pm'][k,n,l:])[0,1]

        ce[var+'_gm'] = ce_gm_lag
        ce[var+'_pm'] = ce_pm_lag
        corr[var+'_gm'] = corr_gm_lag
        corr[var+'_pm'] = corr_pm_lag
        
    return ce, corr

def decompress_eof(forecast,E,E_sic=None,nmodes=None,
                   nmodes_sic=None,sic_separate=False):
    if sic_separate is True: 
        x_forecast = decompress_eof_separate_sic(forecast,nmodes,nmodes_sic,E,E_sic)
    else: 
        x_forecast = np.matmul(E,forecast)
        
    return x_forecast

def calc_lags_gm_pm(truth, lat, lon, fields, tinds, lags, limvars, 
                    decompress=False,E=None,E_sic=None, 
                    Nmodes=None, Nmodes_sic=None, sic_separate=False):
    """
    inputs: 
    =========
    truth: (ndof*nvars, time)
    lat: (nlat)
    lon: (nlon)
    fields: dictionary of indices for each variable
    lags: array of lags
    
    outputs: 
    =========
    truth_state: dictionary with global and polar means for 
                 each variable in limvars
    """
    truth_state = {}
    nlat = lat.shape[0]
    nlon = lon.shape[0]

    for v,var in enumerate(limvars):
        print('working on '+str(var))
        true_all_gm = np.ones((len(lags),tinds))*np.nan
        true_all_pm = np.ones((len(lags),2,tinds))*np.nan
        for k,l in enumerate(lags):
            if decompress is True: 
                if sic_separate is True: 
                    Truth = decompress_eof(truth[k,:,:],E,E_sic=E_sic,
                                           nmodes=Nmodes,nmodes_sic=Nmodes_sic,
                                           sic_separate=sic_separate)
                else: 
                    Truth = np.matmul(E,truth[k,:,:])
            else: 
                Truth=truth
#            print(Truth.shape)
            truth_3d = np.reshape(Truth[fields[var],:].T,[Truth.shape[1],nlat,nlon])
            truth_gm,_,_ = LMR_utils.global_hemispheric_means(truth_3d[l:,:,:],lat)
            truth_pm = polar_regional_means(truth_3d[l:,:,:],lat,lon)

            true_all_gm[k,l:] = truth_gm
            true_all_pm[k,:,l:] = truth_pm

        truth_state[var+'_gm'] = true_all_gm
        truth_state[var+'_pm'] = true_all_pm
        truth_state[var+'_full'] = truth_3d
        
    return truth_state, Truth

def calc_ce_corr_ar_lags(X_valid,X_train,lags,limvars,lat,lon):
    ce_ar1 = {}
    corr_ar1 = {}
    
    nlat=lat.shape[0]
    nlon=lon.shape[0]

    for v,var in enumerate(limvars):
        print('working on '+str(var))
        ar1_gm = np.zeros((len(lags),X_valid.shape[2]))
        ar1_pm = np.zeros((len(lags),2,X_valid.shape[2]))
        true_gm = np.zeros((X_valid.shape[2]))
        true_pm = np.zeros((2,X_valid.shape[2]))
        ce_gm = np.zeros((len(lags)))
        corr_gm = np.zeros((len(lags)))
        ce_pm = np.zeros((len(lags),2))
        corr_pm = np.zeros((len(lags),2))

        for k,lag in enumerate(lags[1:]):
            ar1_forecast, factor = red_noise_forecast_ar1(X_train[v,:].T,X_valid[v,:].T,lead=lag)
            ar1_forecast_3d = np.reshape(ar1_forecast,(ar1_forecast.shape[0],nlat,nlon))
            X_valid_3d = np.reshape(X_valid[v,:].T,(X_valid.shape[2],nlat,nlon))
            ar1_gm[k,lag:],_,_ = LMR_utils.global_hemispheric_means(ar1_forecast_3d,lat)
            ar1_pm[k,:,lag:] = polar_regional_means(ar1_forecast_3d,lat,lon)

            true_gm,_,_ = LMR_utils.global_hemispheric_means(X_valid_3d,lat)
            true_pm = polar_regional_means(X_valid_3d,lat,lon)

            ce_gm[k] = LMR_utils.coefficient_efficiency(true_gm[lag:],ar1_gm[k,lag:])
            corr_gm[k] = np.corrcoef(true_gm[lag:],ar1_gm[k,lag:])[0,1]
            for n in range(2):
                ce_pm[k,n] = LMR_utils.coefficient_efficiency(true_pm[n,lag:],ar1_pm[k,n,lag:])
                corr_pm[k,n] = np.corrcoef(true_pm[n,lag:],ar1_pm[k,n,lag:])[0,1]

        ce_ar1[var+'_gm'] = ce_gm
        corr_ar1[var+'_gm'] = corr_gm
        ce_ar1[var+'_pm'] = ce_pm
        corr_ar1[var+'_pm'] = corr_pm
        
    return ce_ar1, corr_ar1

def calc_ce_corr_ar_mon(X_valid,X_train,lag,limvars,lat,lon):
    ce_ar1 = {}
    corr_ar1 = {}
    
    nlat=lat.shape[0]
    nlon=lon.shape[0]
    
    ntime = X_valid.shape[2]
    nyrs = int(np.floor((ntime/12)))-1

    for v,var in enumerate(limvars):
        ce_gm = np.zeros((12))
        corr_gm = np.zeros((12))
        ce_pm = np.zeros((12,2))
        corr_pm = np.zeros((12,2))

        ar1_forecast,factor = red_noise_forecast_ar1(X_train[v,:].T,X_valid[v,:].T, lead=lag)
        ar1_forecast_3d = np.reshape(ar1_forecast[(12-lag):,:],(nyrs,12,nlat,nlon))
        X_valid_3d = np.reshape(X_valid[v,:].T[12:,:],(nyrs,12,nlat,nlon))
        
        for mon in range(12):                                
            ar1_gm,_,_ = LMR_utils.global_hemispheric_means(ar1_forecast_3d[:,mon,:,:],lat)
            ar1_pm = polar_regional_means(ar1_forecast_3d[:,mon,:,:],lat,lon)

            true_gm,_,_ = LMR_utils.global_hemispheric_means(X_valid_3d[:,mon,:,:],lat)
            true_pm = polar_regional_means(X_valid_3d[:,mon,:,:],lat,lon)

            ce_gm[mon] = LMR_utils.coefficient_efficiency(true_gm,ar1_gm)
            corr_gm[mon] = np.corrcoef(true_gm,ar1_gm)[0,1]
            for n in range(2):
                ce_pm[mon,n] = LMR_utils.coefficient_efficiency(true_pm[n,:],ar1_pm[n,:])
                corr_pm[mon,n] = np.corrcoef(true_pm[n,:],ar1_pm[n,:])[0,1]

        ce_ar1[var+'_gm'] = ce_gm
        corr_ar1[var+'_gm'] = corr_gm
        ce_ar1[var+'_pm'] = ce_pm
        corr_ar1[var+'_pm'] = corr_pm
        
    return ce_ar1, corr_ar1