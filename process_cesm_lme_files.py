import xarray as xr 

def fixmonth(dfile):

    """Fix CESM months since by default the timestamp is for the first day of
    the next month
    
    author: Andrew Pauling 

    Parameters
    ----------

    dfile : xarray dataset
            Dataset containing time to fix

    Returns
    -------

    dfile : xarray dataset
            Fixed dataset
    """

    mytime = dfile['time'][:].data
    for time in range(mytime.size):
        if mytime[time].month > 1:
            mytime[time] = mytime[time].replace(month=mytime[time].month-1)
        elif mytime[time].month == 1:
            mytime[time] = mytime[time].replace(month=12)
            mytime[time] = mytime[time].replace(year=mytime[time].year-1)

    dfile = dfile.assign_coords(time=mytime)

    return dfile


datadir = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'

## Sea ice concentration: 
filename = 'siconc_SImon_CESM_LME_nh_085001-185012.nc'
filename2 = 'siconc_SImon_CESM_LME_nh_185001-200512.nc'

outfile ='sic_SImon_CESM_LME_nh_085001-200512_2.nc'

print('Loading each individual dataset...')
data = xr.open_dataset(datadir+filename, decode_coords=False, use_cftime=True)
data2 = xr.open_dataset(datadir+filename2, decode_coords=False, use_cftime=True)

print('Concatenating datasets...')
datatot = xr.concat([data,data2],dim='time')

print('Running fix months...')
datatot_rn = fixmonth(datatot)

print('Writing netcdf...')
datatot_rn.to_netcdf(datadir+outfile)