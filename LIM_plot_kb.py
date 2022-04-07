import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point


def sub_arctic_plot(ax,fig,pdat,lat,lon,maxv=-1,
                    minv=-1,colorbar=True,extent=True,cmap='bwr'):
    nlat = lat.shape[0]
    nlon = lon.shape[1]
                    
    if maxv == -1:
        maxv = np.nanmax(pdat)
    if minv == -1:
        minv = -maxv                
           
#    pdat_wrap, lon_wrap = add_cyclic_point(pdat,coord=lon[0,:], axis=1)
#    new_lon2d, new_lat2d = np.meshgrid(lon_wrap, lat)
                    
    if extent is True: 
        ax.set_extent([-150, 140, 50, 90], crs=ccrs.PlateCarree())
    ax.gridlines(linestyle='--')
    ax.add_feature(cfeature.LAND, facecolor=(1, 1, 1))
    cs = ax.pcolormesh(lon, lat, pdat, 
                       vmin=minv, vmax=maxv, cmap=cmap, 
                       transform=ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.5)
    if colorbar is True:
        plt.colorbar(cs, ax=ax)
        
    return 