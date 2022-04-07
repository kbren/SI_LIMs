"""
utility functions for LIM work. Main code is in Jupyter notebooks
"""

import sys,os
#sys.path.append("/Users/hakim/gitwork/LMR_python3")
sys.path.append("/home/disk/kalman2/mkb22/LMR_lite/")
import LMR_utils
import LMR_lite_utils as LMRlite
import LMR_config

import numpy as np
from netCDF4 import Dataset, date2num, num2date
import time as timestamp # avoids conflict with local naming!
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point

def get_data(var_to_extract,infile):
    
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
    #units = data.variables[var_to_extract].units

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

                time_yrs = num2date(new_time[:],units=new_time_units,calendar=time.calendar)
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
    dates = time_yrs

    print(spacecoords)

    if var_to_extract == 'tos':
        lat_2d_orig = spacevar1
        lon_2d_orig = spacevar2
    else:
        lat_2d_orig = spacevar1[:,np.newaxis]*np.ones(spacevar2[:,np.newaxis].shape[0])
        lon_2d_orig = np.ones([spacevar1[:,np.newaxis].shape[0],1])*spacevar2

    nlat_orig = lat_2d_orig.shape[0]
    nlon_orig = lon_2d_orig.shape[1]

    print('nlat:',nlat_orig)
    print('nlon:',nlon_orig)
    
    #----- regrid the data to lower resolution
    begin_time = timestamp.time()

    datax = np.reshape(data_var,[ntime,nlat_orig*nlon_orig])
    # shift to (nx,nens) shaping for regridding function
    tmp = np.moveaxis(datax,0,-1)
    print(tmp.shape)

    # this is the new grid
    nlat = 45
    nlon = 72
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

    # compute monthly average and remove from the original data

    # shape back back to (ntime,nlat,nlon)
    data_new2 = np.reshape(np.moveaxis(data_new,0,-1),[ntime,nlat,nlon])
    climo_month = np.zeros([12,nlat,nlon], dtype=float)
    for i in range(12):
        print(i)
        m = i+1
        indsm = [j for j,v in enumerate(dates) if v.month == m]
        indsmref = indsm

        climo_month[i] = np.nanmean(data_new2[indsmref], axis=0)
        data_new2[indsm] = (data_new2[indsm] - climo_month[i])
        # standardize monthly anomalies
        mostd = np.std(data_new2[indsm],axis=0,ddof=1)
        data_new2[indsm] = data_new2[indsm]/mostd

        # GH: make an array with month and year at some point?
    
    return data_new2,lat_2d,lon_2d

def LIM_train(tau,x_train):
    """
    train a LIM, L, given a training dataset
    
    Inputs:
    * tau: the training lag time (unitless) in time steps defined by the format of x_train
    * x_train: ~(nx,nt) state-time matrix
    
    Outputs:
    * LIMd: a dictionary containing the left eigenvectors of L, their inverse, and the eigenvalues of L
    
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
    lam = np.log(val)/tau
    
    # make a dictionary with the results
    LIMd = {}
    LIMd['vec'] = vec
    LIMd['veci'] = veci
    LIMd['lam'] = lam
    
    return LIMd

def LIM_forecast(LIMd,x,lags,E,truth):
    """
    deterministic forecasting experiments for states in x and time lags in lags.

    Inputs:
    * LIMd: a dictionary with LIM attributes
    * x: a state-time matrix for initial conditions and verification ~(ndof,ntims)
    * lags: list of time lags for deterministic forecasts
    * E: the linear map from the coordinates of the LIM to physical (lat,lon) coordinates ~(nx*ny,ndof)
    
    Outputs (in a dictionary):
    * error variance as a function of space and forecast lead time ~(ndof,ntims)
    * the forecast states ~(nlags,ndof,ntims)
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
        
    return LIMfd

def plot_map_vector(vec,lat,lon,minv=-1,maxv=-1,
                    noland=False,cmap='bwr',ax=None,cbar=True):
    nlat = lat.shape[0]
    nlon = lon.shape[1]
    pdat = np.reshape(vec,[nlat,nlon])
    pdat_wrap, lon_wrap = add_cyclic_point(pdat,coord=lon[0,:], axis=1)
    if ax is None:
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=-90.),zorder=1)
    ax.coastlines()
    if maxv == -1:
        maxv = np.nanmax(vec)
    if minv == -1:
        minv = -maxv
    cs = ax.pcolormesh(lon_wrap,lat[:,0],pdat_wrap,transform=ccrs.PlateCarree(),cmap=cmap,shading='flat',vmin=minv,vmax=maxv)
    if cbar: plt.colorbar(cs, extend='both', shrink=0.6)
    if noland:
        ax.add_feature(cartopy.feature.LAND, zorder=2, edgecolor='black',facecolor='black')
    return ax,cs

def euler_twostep(L,Qsr,dt,y_old):
    # stochastic integration scheme
    n = L.shape[0]
    # this has unit variance
    noise = np.random.randn(n,1)
    y_new = y_old + dt*np.matmul(L,y_old) + np.sqrt(dt)*np.matmul(Qsr,noise)
    x_new = 0.5*(y_new + y_old)
    return x_new,y_new

def ob_network_PAGES2k(lat,lon,cfile='config.yml.pseudoproxy_test'):
    cpath = './config/'
    yaml_file = os.path.join(LMR_config.SRC_DIR,cpath+cfile)
    cfg,cfg_dict = LMRlite.load_config(yaml_file)
    prox_manager = LMRlite.load_proxies(cfg)
    ob_lat = []
    ob_lon = []
    for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
        # these two lines filter by proxy type or id
        #if prox_type not in Y.type: continue    
        #if prox_id not in Y.id: continue    
        #print(Y.id,Y.type,Y.lat,Y.lon)
    
        # nearest prior grid lat,lon array indices for the proxy
        tmp = lat[:,0]-Y.lat
        itlat = np.argmin(np.abs(tmp))
        tmp = lon[0,:]-Y.lon
        itlon = np.argmin(np.abs(tmp))
        ob_lat.append(lat[itlat,0])
        ob_lon.append(lon[0,itlon])        
        #print(Y.lat,Y.lon,lat[itlat,0],lon[0,itlon])
    return ob_lat,ob_lon

def make_H(ob_lat,ob_lon,lat_2d,lon_2d,ndof,lalo_pairs=False):
    "Make the H operator given lists of lat,lons. NOT pairs, unless lalo_pairs=True. like LMR_lite_utils.make_obs, but this makes H, not obs"

    nlat = lat_2d.shape[0]
    nlon = lat_2d.shape[1]

    if lalo_pairs:
        if len(ob_lon) != len(ob_lat):
            raise ValueError('lat and lon vectors must be the same length')
        nobs = len(ob_lat)
    else:
        nobs = len(ob_lat)*len(ob_lon)

    print('making H for ',nobs,' observations')
    H = np.zeros([nobs,ndof])

    if lalo_pairs:
        # this option is used when the lat,lon values are order pairs
        obvec_lat = ob_lat
        obvec_lon = ob_lon
        
        for k in range(len(ob_lon)):           
            dist = LMR_utils.get_distance(ob_lon[k],ob_lat[k],
                                          lon_2d[0,:],lat_2d[:,0])
            jind, kind = np.unravel_index(dist.argmin(),dist.shape)
            Hone = np.zeros(lat_2d.shape)
            Hone[jind,kind] = 1.
            Hv = np.reshape(Hone,[1,nlat*nlon])
            H[k,:] = Hv

    else:
        # this option is used when the lat,lon values specify an ordered grid
        obvec_lat = np.zeros(nobs)
        obvec_lon = np.zeros(nobs)

        k= -1
        for lon in ob_lon:
            for lat in ob_lat:
                k +=1
                obvec_lat[k] = lat
                obvec_lon[k] = lon        
                dist = LMR_utils.get_distance(lon,lat,lon_2d[0,:],lat_2d[:,0])
                jind, kind = np.unravel_index(dist.argmin(),dist.shape)
                Hone = np.zeros(lat_2d.shape)
                Hone[jind,kind] = 1.
                Hv = np.reshape(Hone,[1,nlat*nlon])
                H[k,:] = Hv
            
    return H,obvec_lat,obvec_lon

def kalman_update(xbm,B,H,y,R):
    """
    update of mean and covariance using the Kalman filter update equation
    Input:
    * xbm: prior mean
    * B: prior covariance
    * H: observation operator
    * y: vector of observations
    * R: observation error covariance matrix
    Output:
    * xam: posterior mean
    * A: posterior covariance
    """
    # ob estimates
    ye = np.matmul(H,xbm)
    # innovation covariance
    IC = np.matmul(H,np.matmul(B,H.T)) + R
    # gain
    K = np.matmul(np.matmul(B,H.T),np.linalg.inv(IC))
    # innovation
    I = y - ye
    # update mean and covariance
    xam = xbm + np.matmul(K,I)
    A = B - np.matmul(np.matmul(K,H),B)
    
    return xam,A

def error_analysis(xbm,xam,truth):
    """error statistics. simple for now"""
 
    # prior mean error and sum of squares over all eofs
    berr = xbm - truth
    aerr = xam - truth
    psse = np.sum(berr**2)
    asse = np.sum(aerr**2)
    
    return psse,asse

def ob_network(network_name,dlat,dlon,lat_2d,lon_2d):
    
    lalo_pairs = False
    if network_name == '_global':
        print(network_name[1:]+' network')
        # evenly spaced global network; avoid the poles, and insure NH points are symmetric with SH
        ob_lat = np.arange(dlat-90.,90.1-dlat,dlat)
        ob_lon = np.arange(0.,360.,dlon)
    elif network_name == '_northamerica':
        print(network_name[1:]+' network')
        # North America network
        ob_lat = np.arange(30.,80.1-dlat,dlat)
        ob_lon = np.arange(220.,300.,dlon)
    elif network_name == '_tropicalpacific':
        print(network_name[1:]+' network')
        # tropical Pacific network
        ob_lat = np.arange(-20.,30.1-dlat,dlat)
        ob_lon = np.arange(120.,300.,dlon)
    elif network_name == '_PAGES2k':
        print(network_name[1:]+' network')
        # PAGES2k proxy network network. file generated by make_proxy_latlon.ipynb
        # work in progress!
        lalo_pairs = True
        tmp_lat,tmp_lon = LIM_utils.ob_network_PAGES2k(lat_2d,lon_2d)
        ob_lat = np.array(tmp_lat)
        ob_lon = np.array(tmp_lon)
    else:
        print('no valid network!')

    nlat = lat_2d.shape[0]
    nlon = lat_2d.shape[1]
    H,obvec_lat,obvec_lon = make_H(ob_lat,ob_lon,lat_2d,lon_2d,nlat*nlon,lalo_pairs=lalo_pairs)

    return H,obvec_lat,obvec_lon

def LIM_forecast_error(LIMfd,truth,E,lags,fields,lat_2d):
    """
    verify LIM forecasts against a truth state. 
    extracted from LIM_forecast to verify against
    different true states (full and EOF).
    
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
    
    k = -1
    for t in lags:
        k+=1

        print('lag = ',t)
        # physical-space fields for forecast and truth for this forecast lead time ~(ndof,ntims)
        if t == 0:
            # need to handle this time separately, or the matrix dimension is off
            X_predict = np.real(np.matmul(E,LIMfd['x_forecast'][t,:,:]))
            X_truth = truth[:,:]
        else:
            X_predict = np.real(np.matmul(E,LIMfd['x_forecast'][t,:,:-t]))
            X_truth = truth[:,t:]

        #print(X_predict.shape)
        #print(X_truth.shape)
        
        # error variance as a function of space and forecast lead time ~(ndof,ntims)
        error[:,k] = np.var(X_predict - X_truth,axis=1,ddof=1)
        
    # global mean for each field
    n = -1
    for field in fields.keys():
        n+=1
        finds = fields[field]
        error_gm[n,:],_,_ = LMR_utils.global_hemispheric_means(np.moveaxis(np.reshape(error[finds,:],[nlat,nlon,nlags]),-1,0),lat_2d[:,0])

    return error,error_gm
