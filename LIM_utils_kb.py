import xarray as xr
import numpy as np
import scipy as spy
import pickle 

import time as timestamp 
import LIM_stats_kb as kbstats

def build_training_dic(dsource):
    if dsource == 'ccsm4_lm_rt':
        mod = 'CCSM4'
        mod_dir = '/home/disk/kalman3/rtardif/LMR/data/model/ccsm4_last_millenium/'
#         mod_dir_kb = '/home/disk/chaos/mkb22/Documents/SeaIceData/CCSM4/CCSM4_last_millennium/'
        
        infile_sic = mod_dir+'sic_sfc_OImon_'+mod+'_past1000_085001-185012.nc'
        infile_tos = mod_dir+'tos_sfc_Omon_'+mod+'_past1000_085001-185012.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_pr = mod_dir+'pr_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_rlut = mod_dir+'rlut_toa_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_psl = mod_dir+'psl_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_sit = mod_dir+'sit_noMV_OImon_'+mod+'_past1000_085001-185012.nc'
        infile_areacello = 'None'
        infile_areacella = 'None'
        
    elif dsource == 'mpi_lm_rt':
        mod = 'MPI-ESM-P'
        mod_dir_wp = '/home/disk/katabatic/wperkins/data/LMR/data/model/mpi-esm-p_last_millenium/'
        mod_dir = '/home/disk/kalman3/rtardif/LMR/data/model/mpi-esm-p_last_millenium/'
        mod_dir_area = '/home/disk/chaos/mkb22/Documents/SeaIceData/MPI/'
        
        infile_sic = mod_dir+'sic_sfc_OImon_'+mod+'_past1000_085001-185012.nc'
        infile_tos = mod_dir_wp+'tos_sfc_Omon_'+mod+'_past1000_085001-184912.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_pr = mod_dir+'pr_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_rlut = mod_dir+'rlut_toa_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_psl = mod_dir+'psl_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_sit = mod_dir+'sit_sfc_OImon_'+mod+'_past1000_085001-185012.nc'
        infile_areacello = mod_dir_area+'areacello_fx_'+mod+'_past1000_r0i0p0.nc'
        infile_areacella = mod_dir_area+'areacella_fx_'+mod+'_past1000_r0i0p0.nc'
        
    elif dsource == 'ccsm4_lm_kb':
        mod = 'CCSM4'
        mod_dir = '/home/disk/katabatic/wperkins/data/LMR/data/model/ccsm4_last_millenium/'
        mod_dir_sic = '/home/disk/chaos/mkb22/Documents/SeaIceData/CCSM4/CCSM4_last_millennium/'
        
        infile_sic = mod_dir_sic+'sic_sfc_OImon_'+mod+'_past1000_085001-185012.nc'
        infile_tos = mod_dir_sic+'tos_sfc_Omon_'+mod+'_past1000_085001-185012.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_pr = mod_dir+'pr_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_rlut = mod_dir+'rlut_toa_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_psl = mod_dir+'psl_sfc_Amon_'+mod+'_past1000_085001-185012.nc'
        infile_sit = mod_dir_sic+'sit_sfc_OImon_'+mod+'_past1000_085001-185012.nc'
        infile_areacello = mod_dir+'areacello_fx_'+mod+'_past1000_r0i0p0.nc'
        infile_areacella = mod_dir+'areacella_fx_'+mod+'_past1000_r0i0p0.nc'
    
    elif dsource == 'ccsm4_lm_regridlme':
        mod = 'CCSM4'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/ccsm4_last_millenium/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_sfc_OImon_CCSM4_past1000_regridlme_085001-185012.nc'
        infile_tos = mod_dir+'tos_sfc_Omon_CCSM4_past1000_regridlme_085001-185012.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_CCSM4_past1000_regridlme_085001-185012.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_CCSM4_past1000_regridlme_085001-185012.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_sfc_Amon_CCSM4_past1000_regridlme_085001-185012.nc'
        infile_sit = mod_dir+'sit_sfc_OImon_CCSM4_past1000_regridlme_085001-185012.nc'
        infile_areacello = mod_dir2+'areacello_global_LME_02.nc'
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc' 
        
    elif dsource == 'mpi_lm_kb':
        mod = 'MPI-ESM-P'
        mod_dir= '/home/disk/katabatic/wperkins/data/LMR/data/model/mpi-esm-p_last_millenium/'
        mod_dir_sic = '/home/disk/chaos/mkb22/Documents/SeaIceData/MPI/'
        
        infile_sic = mod_dir_sic+'sic_sfc_OImon_'+mod+'_past1000_085001-185012.nc'
        infile_tos = mod_dir+'tos_sfc_Omon_'+mod+'_past1000_085001-184912.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_'+mod+'_past1000_085001-184912.nc'
        infile_zg = mod_dir_sic+'zg_500hPa_Amon_'+mod+'_past1000_085001-184912.nc'
        infile_pr = mod_dir+'pr_sfc_Amon_'+mod+'_past1000_085001-184912.nc'
        infile_rlut = mod_dir+'rlut_toa_Amon_'+mod+'_past1000_085001-184912.nc'
        infile_psl = mod_dir+'psl_sfc_Amon_'+mod+'_past1000_085001-184912.nc'
        infile_sit = mod_dir_sic+'sit_sfc_OImon_'+mod+'_past1000_085001-185012.nc'
        infile_areacello = mod_dir_sic+'areacello_fx_'+mod+'_past1000_r0i0p0.nc'
        infile_areacella = mod_dir_sic+'areacella_fx_'+mod+'_past1000_r0i0p0.nc'
        
    elif dsource == 'mpi_lm_regridlme':
        mod = 'MPI-ESM-P'
        mod_dir= '/home/disk/kalman2/mkb22/LMR/data/model/mpi-esm-p_last_millenium/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_sfc_OImon_MPI-ESM-P_past1000_regridlme_085001-185012.nc'
        infile_tos = mod_dir+'tos_sfc_Omon_MPI-ESM-P_past1000_regridlme_085001-184912.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_MPI-ESM-P_past1000_regridlme_085001-184912.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_MPI-ESM-P_past1000_regridlme_085001-184912.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_sfc_Amon_MPI-ESM-P_past1000_regridlme_085001-184912.nc'
        infile_sit = mod_dir+'sit_sfc_OImon_MPI-ESM-P_past1000_regridlme_085001-185012.nc'
        infile_areacello = mod_dir2+'areacello_global_LME_02.nc'
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'   
        
        
    elif dsource == 'mpi_hist_kb':
        mod = 'MPI-ESM-P'
        mod_dir_rt = '/home/disk/enkf4/rtardif/CMIP_data/CMIP5/mpi-esm-p_historical_r1i1p1/'
        mod_dir_kb = '/home/disk/chaos/mkb22/Documents/SeaIceData/MPI/'
        mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/MPI/'
        
        infile_sic = mod_dir_kb+'sic_OImon_'+mod+'_historical_r1i1p1_185001-200512.nc'
        infile_tos = mod_dir_kb+'tos_Omon_'+mod+'_historical_r1i1p1_185001-200512.nc'
        infile_tas = mod_dir_kb+'tas_Amon_'+mod+'_historical_r1i1p1_185001-200512.nc'
        infile_zg = mod_dir_rt+'zg_500hPa_Amon_'+mod+'_historical_185001-200512.nc'
        infile_pr = mod_dir_rt+'pr_sfc_Amon_'+mod+'_historical_185001-200512.nc'
        infile_rlut = None
        infile_psl = mod_dir_kb+'psl_Amon_'+mod+'_historical_r1i1p1_185001-200512.nc'
        infile_sit = mod_dir_kb+'sit_OImon_'+mod+'_historical_r1i1p1_185001-200512.nc'
        infile_areacello = mod_dir_kb+'areacello_fx_'+mod+'_past1000_r0i0p0.nc'
        infile_areacella = mod_dir_kb+'areacella_fx_'+mod+'_past1000_r0i0p0.nc'
        
    elif dsource == 'ccsm4_hist_kb':
        mod = 'CCSM4'
        mod_dir_rt = '/home/disk/enkf4/rtardif/CMIP_data/CMIP5/ccsm4_historical_r1i1p1/'
        mod_dir_kb = '/home/disk/chaos/mkb22/Documents/SeaIceData/CCSM4/CCSM4_historical/'
        mod_dir = '/home/disk/katabatic/wperkins/data/LMR/data/model/ccsm4_last_millenium/'
        
        infile_sic = mod_dir_kb+'sic_sfc_OImon_'+mod+'_historical_r1i1p1_185001-200512.nc'
        infile_tos = mod_dir_kb+'tos_Omon_'+mod+'_historical_r1i1p1_185001-200512.nc'
        infile_tas = mod_dir_kb+'tas_sfc_Amon_'+mod+'_historical_r1i1p1_185001-200512.nc'
        infile_zg = mod_dir_rt+'zg_500hPa_Amon_'+mod+'_historical_185001-200512.nc'
        infile_pr = mod_dir_rt+'pr_sfc_Amon_'+mod+'_historical_185001-200512.nc'
        infile_rlut = None
        infile_psl = mod_dir_kb+'psl_sfc_Amon_'+mod+'_historical_r1i1p1_185001-200512.nc'
        infile_sit = mod_dir_kb+'sit_sfc_OImon_'+mod+'_historical_r1i1p1_185001-200512.nc'
        infile_areacello = mod_dir+'areacello_fx_'+mod+'_past1000_r0i0p0.nc'
        infile_areacella = mod_dir+'areacella_fx_'+mod+'_past1000_r0i0p0.nc'
        
    elif dsource == 'cmip6_mpi_hist':
        mod = 'MPI-ESM1-2-LR'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-mpi-esm1-2-historical/'
        
        infile_sic = mod_dir+'sic_SImon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_tos = mod_dir+'tos_Omon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_sit = mod_dir+'sit_SImon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_areacello = mod_dir+'areacello_Ofx_'+mod+'_historical_r1i1p1f1_gn.nc'
        infile_areacella = mod_dir+'areacella_fx_'+mod+'_historical_r1i1p1f1_gn.nc'
        
    elif dsource == 'cmip6_mpi_hist_regridlme':
        mod = 'MPI-ESM1-2-LR'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-mpi-esm1-2-historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_SImon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_tos = mod_dir+'tos_Omon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_sit = mod_dir+'sit_SImon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_areacello = mod_dir2+'areacello_global_LME_02.nc'
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cmip6_mpi_hist_regridlme_Amon':
        mod = 'MPI-ESM1-2-LR'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-mpi-esm1-2-historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_tos = mod_dir+'tos_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_sit = mod_dir+'sit_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
#        infile_ice_areacello = mod_dir+'areacello_CESM_LME_nh_001.nc'
        infile_areacello = None
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cmip6_mpi_ssp585':
        mod = 'MPI-ESM1-2-LR'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-mpi-esm1-2-ssp585/'
        
        infile_sic = mod_dir+'sic_SImon_'+mod+'_ssp585_r1i1p1f1_gn_201501-210012.nc'
        infile_tos = mod_dir+'tos_Omon_'+mod+'_ssp585_r1i1p1f1_gn_201501-210012.nc'
        infile_tas = mod_dir+'tas_Amon_'+mod+'_ssp585_r1i1p1f1_gn_201501-210012.nc'
        infile_zg = None
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_'+mod+'_ssp585_r1i1p1f1_gn_201501-210012.nc'
        infile_sit = mod_dir+'sit_SImon_'+mod+'_ssp585_r1i1p1f1_gn_201501-210012.nc'
        infile_areacello = mod_dir+'areacello_Ofx_'+mod+'_ssp585_r1i1p1f1_gn.nc'
        infile_areacella = mod_dir+'areacella_fx_'+mod+'_ssp585_r1i1p1f1_gn.nc'
        
    elif dsource == 'cmip6_mpi_hist2':
        mod = 'MPI-ESM1-2-LR'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-mpi-esm1-2-historical/'
        
        infile_sic = mod_dir+'sic_SImon_'+mod+'_historical_r10i1p1f1_gn_185001-201412.nc'
        infile_tos = mod_dir+'tos_Omon_'+mod+'_historical_r10i1p1f1_gn_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_'+mod+'_historical_r10i1p1f1_gn_185001-201412.nc'
        infile_zg = None
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_'+mod+'_historical_r10i1p1f1_gn_185001-201412.nc'
        infile_sit = mod_dir+'sit_SImon_'+mod+'_historical_r10i1p1f1_gn_185001-201412.nc'
        infile_areacello = mod_dir+'areacello_Ofx_'+mod+'_historical_r1i1p1f1_gn.nc'
        infile_areacella = mod_dir+'areacella_fx_'+mod+'_historical_r1i1p1f1_gn.nc'
        
    elif dsource == 'cmip6_cesm2_hist':
        mod = 'CESM2'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-cesm2-historical/'
        
        infile_sic = mod_dir+'sic_SImon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_tos = mod_dir+'tos_Omon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_sit = mod_dir+'sit_SImon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_areacello = mod_dir+'areacello_Ofx_'+mod+'_historical_r11i1p1f1_gn.nc'
        infile_areacella = mod_dir+'areacella_fx_'+mod+'_historical_r11i1p1f1_gn.nc'
        
    elif dsource == 'cmip6_cesm2_hist_regridlme':
        mod = 'CESM2'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-cesm2-historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_SImon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_tos = mod_dir+'tos_Omon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_'+mod+'_historical_r1i1p1f1_regridLME_185001-201412.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_'+mod+'_historical_r1i1p1f1_regridLME_185001-201412_noplev.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_'+mod+'_historical_r1i1p1f1_regridLME_185001-201412.nc'
        infile_sit = mod_dir+'sit_SImon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_areacello = mod_dir+'areacello_Ofx_'+mod+'_historical_r11i1p1f1_gn.nc'
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cmip6_cesm2_ssp585':
        mod = 'CESM2'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-cesm2-ssp585/'
        
        infile_sic = mod_dir+'sic_SImon_'+mod+'_ssp585_r4i1p1f1_gn_201501-210012.nc'
        infile_tos = mod_dir+'tos_Omon_'+mod+'_ssp585_r4i1p1f1_gn_201501-210012.nc'
        infile_tas = mod_dir+'tas_Amon_'+mod+'_ssp585_r4i1p1f1_gn_201501-210012.nc'
        infile_zg = None
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_'+mod+'_ssp585_r4i1p1f1_gn_201501-210012.nc'
        infile_sit = mod_dir+'sit_SImon_'+mod+'_ssp585_r4i1p1f1_gn_201501-210012.nc'
        infile_areacello = mod_dir+'areacello_Ofx_'+mod+'_ssp585_r4i1p1f1_gn.nc'
        infile_areacella = mod_dir+'areacella_fx_'+mod+'_ssp585_r4i1p1f1_gn.nc'
        
    elif dsource == 'cmip6_gfdl_hist':
        mod = 'GFDL-ESM4'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-gfdl-esm4-historical/'
        
        infile_sic = mod_dir+'sic_SImon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_tos = mod_dir+'tos_Omon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_'+mod+'_historical_r1i1p1f1_gr1_185001-201412.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_'+mod+'_historical_r1i1p1f1_gr1_185001-201412.nc'
        infile_sit = mod_dir+'sit_SImon_'+mod+'_historical_r1i1p1f1_gn_185001-201412.nc'
        infile_areacello = mod_dir+'areacello_Ofx_'+mod+'_historical_r1i1p1f1_gn.nc'
        infile_areacella = mod_dir+'areacella_fx_'+mod+'_historical_r1i1p1f1_gr1.nc'
        
    elif dsource == 'cmip6_gfdl_hist_regridlme':
        mod = 'GFDL-ESM4'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-gfdl-esm4-historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_SImon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_tos = mod_dir+'tos_Omon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_GFDL-ESM4_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_sit = mod_dir+'sit_SImon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_areacello = mod_dir2+'areacello_global_LME_02.nc'
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cmip6_gfdl_hist_regridlme_Amon':
        mod = 'GFDL-ESM4'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-gfdl-esm4-historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_tos = mod_dir+'tos_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_GFDL-ESM4_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_sit = mod_dir+'sit_Amon_'+mod+'_historical_r1i1p1f1_regridlme_185001-201412.nc'
        infile_areacello = None
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cmip6_CanESM_hist':
        mod = 'CanESM5'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6_CanESM5_historical/'
        
        infile_sic = mod_dir+'sic_SImon_CanESM5_historical_r1i1p2f1_gn_185001-201412.nc'
        infile_tos = mod_dir+'tos_Omon_CanESM5_historical_r1i1p2f1_gn_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_CanESM5_historical_r1i1p2f1_gn_185001-201412.nc'
        infile_zg = mod_dir+'zg500hPa_Amon_CanESM5_historical_r1i1p2f1_gn_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_CanESM5_historical_r1i1p2f1_gn_185001-201412.nc'
        infile_sit = mod_dir+'sit_SImon_CanESM5_historical_r1i1p2f1_gn_185001-201412.nc'
        infile_areacello = mod_dir+'areacello_Ofx_CanESM5_historical_r1i1p2f1_gn.nc	'
        infile_areacella = mod_dir+'areacella_fx_CanESM5_historical_r1i1p2f1_gn.nc'
        
    elif dsource == 'cmip6_CanESM_hist_regridlme':
        mod = 'CanESM5'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6_CanESM5_historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_SImon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_tos = mod_dir+'tos_Omon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_zg = mod_dir+'zg500hPa_Amon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_sit = mod_dir+'sit_SImon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_areacello = mod_dir2+'areacello_global_LME_02.nc'
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cmip6_CanESM_hist_regridlme_Amon':
        mod = 'CanESM5'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6_CanESM5_historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_Amon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_tos = mod_dir+'tos_Amon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_tas = mod_dir+'tas_Amon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_zg = mod_dir+'zg500hPa_Amon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_sit = mod_dir+'sit_Amon_CanESM5_historical_r1i1p2f1_regridlme_185001-201412.nc'
        infile_areacello = None
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cmip6_HadGEM3_hist':
        mod = 'HadGEM3-GC31-LL'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-HadGEM3-GC31-LL-historical/'
        
        infile_sic = mod_dir+'sic_SImon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_185001-201412.nc'
        infile_tos = mod_dir+'tos_Omon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_185001-201412.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_185001-201412.nc'
        infile_zg = mod_dir+'zg500hPa_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_sfc_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_185001-201412.nc'
        infile_sit = mod_dir+'sit_SImon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_185001-201412.nc'
        infile_areacello = mod_dir+'areacello_Ofx_HadGEM3-GC31-LL_hist-1950_r1i1p1f1_gn.nc'
        infile_areacella = mod_dir+'areacella_fx_HadGEM3-GC31-LL_hist-1950_r1i1p1f1_gn.nc'
        
    elif dsource == 'cmip6_HadGEM3_hist_regridlme':
        mod = 'HadGEM3-GC31-LL'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-HadGEM3-GC31-LL-historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_SImon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_tos = mod_dir+'tos_Omon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_zg = mod_dir+'zg500hPa_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_sfc_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_sit = mod_dir+'sit_SImon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_areacello = mod_dir2+'areacello_global_LME_02.nc'
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cmip6_HadGEM3_hist_regridlme_Amon':
        mod = 'HadGEM3-GC31-LL'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/cmip6-HadGEM3-GC31-LL-historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_tos = mod_dir+'tos_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_zg = mod_dir+'zg500hPa_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_sfc_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_sit = mod_dir+'sit_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_regridlme_185001-201412.nc'
        infile_areacello = None
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'multimod_CESM2_MPI_GFDL_HadGEM3_hist':
        mod = 'multimod_hist'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/multimodel_historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_SImon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_latcut40_regridlme_185001-250912.nc'
        infile_tos = mod_dir+'tos_Omon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_latcut40_regridlme_185001-250912.nc'
        infile_tas = mod_dir+'tas_Amon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_latcut40_regridlme_185001-250912.nc'
        infile_zg = mod_dir+'zg_Amon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_latcut40_regridlme_185001-250912.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_Amon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_latcut40_regridlme_185001-250912.nc'
        infile_sit = mod_dir+'sit_SImon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_latcut40_regridlme_185001-250912.nc'
        infile_areacello = mod_dir2+'areacello_global_LME_02.nc'
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'multimod_CESM1_MPI_GFDL_HadGEM3_CanESM_hist_Amon':
        mod = 'multimod_hist'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/multimodel_historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = (mod_dir+'sic_Amon_CESM1_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-262912.nc')
        infile_tos = (mod_dir+'tos_Amon_CESM1_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut0.1_regridlme_185001-262912.nc')
        infile_tas = (mod_dir+'tas_Amon_CESM1_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut0.1_regridlme_185001-262912.nc')
        infile_zg = (mod_dir+'zg_Amon_CESM1_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut0.1_regridlme_185001-262912.nc')
        infile_pr = None
        infile_rlut = None
        infile_psl = (mod_dir+'psl_Amon_CESM1_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut0.1_regridlme_185001-262912.nc')
        infile_sit = (mod_dir+'sit_Amon_CESM1_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-262912.nc')
        infile_areacello = None
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'multimod_CESM2_MPI_GFDL_HadGEM3_CanESM_hist':
        mod = 'multimod_hist'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/multimodel_historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = (mod_dir+'sic_SImon_CESM2_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-262912.nc')
        infile_tos = (mod_dir+'tos_Omon_CESM2_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-262912.nc')
        infile_tas = (mod_dir+'tas_Amon_CESM2_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-262912.nc')
        infile_zg = (mod_dir+'zg_Amon_CESM2_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-262912.nc')
        infile_pr = None
        infile_rlut = None
        infile_psl = (mod_dir+'psl_Amon_CESM2_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-262912.nc')
        infile_sit = (mod_dir+'sit_SImon_CESM2_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-262912.nc')
        infile_areacello = mod_dir2+'areacello_global_LME_02.nc'
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'multimod_CESM2_MPI_GFDL_HadGEM3_hist':
        mod = 'multimod_hist'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/multimodel_historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = (mod_dir+'sic_SImon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_'+
                      'latcut40_regridlme_185001-250912.nc')
        infile_tos = (mod_dir+'tos_Omon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_'+
                      'latcut40_regridlme_185001-250912.nc')
        infile_tas = (mod_dir+'tas_Amon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_'+
                      'latcut40_regridlme_185001-250912.nc')
        infile_zg = (mod_dir+'zg_Amon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_'+
                      'latcut40_regridlme_185001-250912.nc')
        infile_pr = None
        infile_rlut = None
        infile_psl = (mod_dir+'psl_Amon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_'+
                      'latcut40_regridlme_185001-250912.nc')
        infile_sit = (mod_dir+'sit_SImon_CESM2_MPI_GFDL_HadGEM3_historical_detrended_'+
                      'latcut40_regridlme_185001-250912.nc')
        infile_areacello = mod_dir2+'areacello_global_LME_02.nc'
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'multimod_MPI_GFDL_HadGEM3_CanESM_hist_Amon':
        mod = 'multimod_hist'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/multimodel_historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = (mod_dir+'sic_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_tos = (mod_dir+'tos_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut0.1_regridlme_185001-247312.nc')
        infile_tas = (mod_dir+'tas_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut0.1_regridlme_185001-247312.nc')
        infile_zg = (mod_dir+'zg_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut0.1_regridlme_185001-247312.nc')
        infile_pr = None
        infile_rlut = None
        infile_psl = (mod_dir+'psl_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut0.1_regridlme_185001-247312.nc')
        infile_sit = (mod_dir+'sit_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_areacello = None
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'multimod_MPI_GFDL_HadGEM3_CanESM_hist_Amon_latcut40':
        mod = 'multimod_hist'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/multimodel_historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = (mod_dir+'sic_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_tos = (mod_dir+'tos_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_tas = (mod_dir+'tas_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_zg = (mod_dir+'zg_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_pr = None
        infile_rlut = None
        infile_psl = (mod_dir+'psl_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_sit = (mod_dir+'sit_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_areacello = None
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'multimod_MPI_GFDL_HadGEM3_CanESM_hist_latcut40':
        mod = 'multimod_hist'
        mod_dir = '/home/disk/kalman2/mkb22/LMR/data/model/multimodel_historical/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = (mod_dir+'sic_SImon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_tos = (mod_dir+'tos_Omon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_tas = (mod_dir+'tas_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_zg = (mod_dir+'zg_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_pr = None
        infile_rlut = None
        infile_psl = (mod_dir+'psl_Amon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_sit = (mod_dir+'sit_SImon_MPI_GFDL_HadGEM3_CanESM_156each_historical_detrended_'+
                      'latcut40_regridlme_185001-247312.nc')
        infile_areacello = mod_dir2+'areacello_global_LME_02.nc'
        infile_areacella = mod_dir2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'satellite':
        mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/Observations/'
        
        infile_sic = mod_dir+'sic_goddard_merged_mon_v03r01_1979_2016.nc'
        infile_tos = None
        infile_tas = None
        infile_zg = None
        infile_pr = None
        infile_rlut = None
        infile_psl = None
        infile_sit = None
        infile_areacello = mod_dir + 'areacello_CDR_NASA-Team_polar_stereo_25km_NH.nc'
        infile_areacella = None
        
    elif dsource == 'satellite_regridlme':
        mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/Observations/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_goddard_merged_mon_v03r01_regridlme_1979_2016.nc'
        infile_tos = None
        infile_tas = None
        infile_zg = None
        infile_pr = None
        infile_rlut = None
        infile_psl = None
        infile_sit = None
        infile_areacello = mod_dir2+'areacello_global_LME_02.nc'
        infile_areacella = None
        
    elif dsource == 'era5':
        mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/ERA5/'
        mod_dir_LME1 = '/home/disk/chaos/mkb22/Documents/SeaIceData/CESM_LE/'
        mod_dir_LME2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_monthly_ERA5_1979_2020_grid05.nc'
        infile_tos = mod_dir+'tos_monthly_ERA5_1979_2020_grid05.nc'
        infile_tas = mod_dir+'tas_monthly_ERA5_1979_2020_grid05.nc'
        infile_zg = None
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_monthly_ERA5_1979_2020_grid05.nc'
        infile_sit = None
        infile_areacello = mod_dir+'areacello_grid05.nc'
        infile_areacella = mod_dir+'areacella_grid05.nc'
        
    elif dsource == 'era5_regridlme':
        mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/ERA5/'
        mod_dir_LME1 = '/home/disk/chaos/mkb22/Documents/SeaIceData/CESM_LE/'
        mod_dir_LME2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_monthly_ERA5_1979_2020_regridlme.nc'
        infile_tos = mod_dir+'tos_monthly_ERA5_1979_2020_regridlme.nc'
        infile_tas = mod_dir+'tas_monthly_ERA5_1979_2020_regridlme.nc'
        infile_zg = None
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_monthly_ERA5_1979_2020_regridlme.nc'
        infile_sit = None
        infile_areacello = mod_dir_LME1+'areacello_CESM_LE_002_tos.nc'
        infile_areacella = mod_dir_LME2+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cesm_le':
        mod = 'CESM_LE'
        mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/CESM_LE/'
        
        infile_sic = mod_dir+'sic_CESM_LE_002_nh_200601-210012.nc'
        infile_tos = mod_dir+'tos_CESM_LE_002_200601_210012_2d.nc'
        infile_tas = mod_dir+'tas_CESM_LE_002_200601-210012.nc'
        infile_zg = mod_dir+'zg_CESM_LE_002_200601-210012.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_CESM_LE_002_200601-210012.nc'
        infile_sit = mod_dir+'sit_CESM_LE_002_nh_200601-210012.nc'
#        infile_ice_areacello = mod_dir+'areacello_CESM_LME_nh_001.nc'
        infile_areacello = mod_dir+'areacello_CESM_LE_002_tos.nc'
        infile_areacella = mod_dir+'areacella_CESM_LE_global.nc'
        
    elif dsource == 'cesm_le_lmegrid':
        mod = 'CESM_LE'
        mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/CESM_LE/'
        mod_dir_LME = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_CESM_LE_002_nh_200601-210012.nc'
        infile_tos = mod_dir+'tos_CESM_LE_002_200601_210012_2d.nc'
        infile_tas = mod_dir+'tas_CESM_LE_002_regrid_cesm_lme_200601-210012.nc'
        infile_zg = mod_dir+'zg_500hPa_CESM_LE_002_regrid_cesm_lme_200601-210012.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_CESM_LE_002_regrid_cesm_lme_200601-210012.nc'
        infile_sit = mod_dir+'sit_CESM_LE_002_nh_200601-210012.nc'
#        infile_ice_areacello = mod_dir+'areacello_CESM_LME_nh_001.nc'
        infile_areacello = mod_dir+'areacello_CESM_LE_002_tos.nc'
        infile_areacella = mod_dir_LME+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cesm_lme':
        mod = 'CESM_LME'
        mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_SImon_CESM_LME_nh_002_085001-200512.nc'
        infile_tos = mod_dir+'tos_sfc_Omon_CESM_LMEallforc_002_085001-200512.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_CESM_LMEallforc_002_085001-200512.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_CESM_LMEallforc_002_085001-200512.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_sfc_Amon_CESM_LMEallforc_002_085001-200512.nc'
        infile_sit = mod_dir+'sit_SImon_CESM_LME_nh_002_085001-200512.nc'
#        infile_ice_areacello = mod_dir+'areacello_CESM_LME_nh_001.nc'
        infile_areacello = mod_dir+'areacello_global_LME_02.nc'
        infile_areacella = mod_dir+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cesm_lme_Amon':
        mod = 'CESM_LME'
        mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        
        infile_sic = mod_dir+'sic_SImon_CESM_LME_nh_002_regrid_Amon_085001-200512.nc'
        infile_tos = mod_dir+'tos_sfc_Omon_CESM_LMEallforc_002_regrid_Amon_085001-200512.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_CESM_LMEallforc_002_085001-200512.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_CESM_LMEallforc_002_085001-200512.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_sfc_Amon_CESM_LMEallforc_002_085001-200512.nc'
        infile_sit = mod_dir+'sit_SImon_CESM_LME_nh_002_regrid_Amon_085001-200512.nc'
#        infile_ice_areacello = mod_dir+'areacello_CESM_LME_nh_001.nc'
        infile_areacello = None
        infile_areacella = mod_dir+'areacella_CESM_LME_global_001.nc'
        
    elif dsource == 'cesm_lme_regrid_mpilm':
        mod = 'CESM_LME'
        mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        mod_dir2 = '/home/disk/chaos/mkb22/Documents/SeaIceData/MPI/'
        
        infile_sic = mod_dir+'sic_SImon_CESM_LME_nh_002_regrid_mpilm_085001-200512.nc'
        infile_tos = mod_dir+'tos_sfc_Omon_CESM_LMEallforc_002_regrid_mpilm_085001-200512.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_CESM_LMEallforc_002_regrid_mpilm_085001-200512.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_CESM_LMEallforc_002_regrid_mpilm_085001-200512.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_sfc_Amon_CESM_LMEallforc_002_regrid_mpilm_085001-200512.nc'
        infile_sit = mod_dir+'sit_SImon_CESM_LME_nh_002_regrid_mpilm_085001-200512.nc'
#        infile_ice_areacello = mod_dir+'areacello_CESM_LME_nh_001.nc'
        infile_areacello = mod_dir2+'areacello_fx_MPI-ESM-P_past1000_r0i0p0.nc'
        infile_areacella = mod_dir2+'areacella_fx_MPI-ESM-P_past1000_r0i0p0.nc'
        
    elif dsource == 'cesm_lme_rgmpi':
        mod = 'CESM_LME'
        mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/LIMs/'
        mod_dir_sic = '/home/disk/chaos/mkb22/Documents/SeaIceData/MPI/'
        
        infile_sic = mod_dir+'sic_SImon_CESM_LME_nh_002_regrid_mpi_lm_085001-200512.nc'
        infile_tos = mod_dir+'tos_sfc_Omon_CESM_LMEallforc_002_regrid_mpi_lm_085001-200512.nc'
        infile_tas = mod_dir+'tas_sfc_Amon_CESM_LMEallforc_002_regrid_mpi_lm_085001-200512.nc'
        infile_zg = mod_dir+'zg_500hPa_Amon_CESM_LMEallforc_002_regrid_mpi_lm_085001-200512.nc'
        infile_pr = None
        infile_rlut = None
        infile_psl = mod_dir+'psl_sfc_Amon_CESM_LMEallforc_002_regrid_mpi_lm_085001-200512.nc'
        infile_sit = mod_dir+'sit_SImon_CESM_LME_nh_002_regrid_mpi_lm_085001-200512.nc'
#        infile_ice_areacello = mod_dir+'areacello_CESM_LME_nh_001.nc'
        infile_areacello = mod_dir_sic+'areacello_fx_'+mod+'_past1000_r0i0p0.nc'
        infile_areacella = mod_dir_sic+'areacella_fx_'+mod+'_past1000_r0i0p0.nc'

    elif dsource == 'cesm_lme_lm_02':
            mod = 'CESM_CAM5_LME'
            mod_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/LME/'

            infile_sic = mod_dir+'sic_global_02_'+mod+'_085001-185012.nc'
            infile_tos = None
            infile_tas = None
            infile_zg = None
            infile_pr = None
            infile_rlut = None
            infile_psl = None
            infile_sit = None
            infile_areacello = mod_dir+'areacello_global_LME_02.nc'
            infile_areacella = None
        
    fdic = {'fpath':mod_dir,
            'tos':infile_tos,
            'tas':infile_tas,
            'zg':infile_zg,
            'rlut':infile_rlut,
            'sic':infile_sic,
            'psl':infile_psl,
            'pr':infile_pr,
            'sit':infile_sit,
            'areacella':infile_areacella,
            'areacello':infile_areacello
           }
        
    return fdic


def load_data_og(var, var_dict, fdic, remove_climo=True, detrend=True, atol=False, 
              verbose=True, cmip6=False, tscut=False, tecut=False, lat_cutoff = False): 
    """
    INPUTS:
    =========
    limvars:   list of strings with variable names 
    fdic:   dictionary of variables names, location and filenames 
            (results from build_training_dic())
    remove_climo: True/False whether anomalies are returned or not
    detrend:  True/False whether detrended data is returned or not 
    verbose:  True/False whether print statements or not
    cmip6: True/False whether to use xr.open_mfdataset() or xr.open_dataset()
    tcut: number of years to exclude from the eof decomposition
    
    OUTPUTS: 
    ==========
    X_all:    concatenated array of all variables in limvars, 
              stacked along 1D spatial dimension
    var_dict: dictionary with variables as keys, contains index location in X_all, 
              lat values, lon values, and 1D spatial dimension (number of DOF) 
              for each variable. 
    """
    
    begin_time = timestamp.time()
    if verbose is True: 
        print('Loading from '+var)
        
    print('Loading from '+ fdic[var][-60:])
    if cmip6 is True: 
        data_in = xr.open_mfdataset(fdic[var])
        data_in = data_in.drop_vars('lon')
        data_in = data_in.drop_vars('lat')
        data_in = data_in.drop_vars('vertices_latitude')
        data_in = data_in.drop_vars('vertices_longitude')
        
        data_one = xr.open_dataset(fdic[var][:-4]+'185001-186912.nc')
        data_in = data_in.assign_coords(lon=data_one['lon'].astype('float32'))
        data_in = data_in.assign_coords(lat=data_one['lat'].astype('float32'))
        data_in.load()
        
    else: 
        data_in = xr.open_dataset(fdic[var])

#     if var is 'zg':
#         data_in = data_in.sel(plev=5e4) 
        
#     if 'LME' in fdic[var]:
#         print('LME detected')
#         data_in = data_in.sel(member=1)
#         if 'tos' in var: 
#             data_in = data_in.isel(time=slice(12,13872))
#         elif 'sic' in var: 
#             data_in = data_in.isel(time=slice(11,11999))
    
    # Fill pole hole for sea ice in satellite data: 
    if (var is 'sic') & ('Observations' in fdic['fpath']):
        data_nans= data_in.where(data_in[var]<1.1)
        data_in = data_nans.where(data_nans.lat<84.2,1.0)
        
    if 'goddard' in fdic[var]: 
        t = data_in.time.values
        t[107] = np.datetime64('1987-12-01')
        t[108] = np.datetime64('1988-01-01')

        #data_in_old['time'] = t
        data_in = data_in.assign_coords(time=t)

        data_in.sic[107,:,:] = np.nan
        data_in.sic[108,:,:] = np.nan
        
    # Select time period of interest: 
    if tscut and tecut:
        tscut_tot = (12*tscut)
        tecut_tot = (12*tecut)
        data_in = data_in.isel(time=slice(tscut_tot,tecut_tot))
        if 'datetime64' in str(type(data_in.time.values[0])):
            print('time dimension: '+str(data_in.time.values[0].astype('M8[Y]'))+' - '+
                  str(data_in.time.values[-1].astype('M8[Y]')))
#            print(data_in.time.shape)
        else: 
            print('time dimension: '+str(data_in.time.values[0].year)+' - '+
                  str(data_in.time.values[-1].year))
#            print(data_in.time.shape)
    elif tecut: 
        tecut_tot = (12*tecut)
        data_in = data_in.isel(time=slice(0,-tecut_tot))
        if 'datetime64' in str(type(data_in.time.values[0])):
            print('time dimension: '+str(data_in.time.values[0].astype('M8[Y]'))+' - '+
                  str(data_in.time.values[-1].astype('M8[Y]')))
#             print(data_in.time.shape)
        else: 
            print('time dimension: '+str(data_in.time.values[0].year)+' - '+
                  str(data_in.time.values[-1].year))
#             print(data_in.time.shape)
    elif tscut: 
        tscut_tot = (12*tscut)
        data_in = data_in.isel(time=slice(tscut_tot,data_in.time.shape[0]))
        if 'datetime64' in str(type(data_in.time.values[0])):
            print('time dimension: '+str(data_in.time.values[0].astype('M8[Y]'))+' - '+
                  str(data_in.time.values[-1].astype('M8[Y]')))
#             print(data_in.time.shape)
        else: 
            print('time dimension: '+str(data_in.time.values[0].year)+' - '+
                  str(data_in.time.values[-1].year))
#             print(data_in.time.shape)
        
    vardims = list(data_in[var].dims)
    
    if len(vardims)<=2:
        nlat = data_in[vardims[0]].shape[0]
        nlon = data_in[vardims[1]].shape[0]
        ndof = nlat*nlon
    elif len(vardims)>=3:
        ntime = data_in[vardims[0]].shape[0]
        nlat = data_in[vardims[1]].shape[0]
        nlon = data_in[vardims[2]].shape[0]
        ndof = nlat*nlon

    x_var = data_in[var].values
    
    ### Do regridding here ###
    

    if len(vardims)<=2:
        x_var_2d = np.reshape(x_var,[ndof]) 
    else: 
        x_var_2d = np.moveaxis(np.reshape(x_var,[ntime,ndof]),0,-1) 


    if detrend is True: 
        print('detrending...')
        if len(vardims)<=2:
            print('No time dimension, cant detrend...')
        else: 
            x_var_dt = np.zeros((ndof,int(ntime/12),12))
            climo = np.zeros((ndof,12))
            x_var_3d = np.reshape(x_var_2d, (ndof,int(ntime/12),12))
            for i in range(12):
                print('Working on month...'+str(i))
                y = x_var_3d[:,:,i]
                if ('goddard' in fdic[var]) & (np.isnan(y).sum()>0):
                    print('Found some nans in satellite data, going to fill with previous timestep...')
                    inds = np.where(np.isnan(y))
                    ind_int = int(inds[1].min()-1)
                    fill = np.ones(y.shape)*y[:,ind_int][:,np.newaxis]
                    var_nans_mask = np.where(np.isnan(y),np.nan,1)
                    Y = np.where(np.isnan(y),fill,y)
                elif np.isnan(y).sum()>0: 
                    print('Found with nans (not in satellite), going to fill with zeros...')
                    var_nans_mask = np.where(np.isnan(y),np.nan,1)
                    Y = np.where(np.isnan(y),0,y)
                else: 
                    Y = y
    #                print('Y = '+ str(Y.shape))
                X = np.arange(0,int(ntime/12))
                [var_dt,_,intercept] = kbstats.multi_linear_detrend(X,Y,axis=1,atol=False,
                                                                    remove_mn=remove_climo)
                if np.isnan(y).sum()>0:
                    x_var_dt[:,:,i] = var_dt*var_nans_mask
                else: 
                    x_var_dt[:,:,i] = var_dt
                climo[:,i] = intercept

            x_var_anom = np.reshape(x_var_dt,(ndof,ntime))           
    else: 
        x_var_dt = x_var_2d
        if remove_climo is True: 
            print('removing climotology...')
            x_var_3d = np.reshape(x_var_dt,(ndof,int(ntime/12),12))
            climo = np.nanmean(x_var_3d,axis=1)
            x_var_anom = x_var_3d - climo[:,np.newaxis,:]
            x_var_anom = np.reshape(x_var_anom,(ndof,ntime))
        else: 
            x_var_anom = x_var_dt
            climo = 'None'

    X_var = x_var_anom
    
    if lat_cutoff:
        print('latitude cutoff detected: '+str(lat_cutoff))
        if len(data_in.lat.values.shape)>=3:
            lat_og = data_in.lat.values[:,:,0]
            lon_og = data_in.lon.values[:,:,0]
        else: 
            lat_og = np.where(data_in.lat.values>90,np.nan,data_in.lat.values)
            lon_og = np.where(data_in.lon.values>360,np.nan,data_in.lon.values)

        if len(vardims)<=2:
            if len(lat_og.shape)<=1:
                lat_inds = (lat_og>=lat_cutoff)
                nlo = lon_og.shape[0]
                nla = lat_og.shape[0]
                X_var_3d = np.reshape(X_var,(nla,nlo))

                X_var_o = X_var_3d[lat_inds,:]
                X_var_out = np.reshape(X_var_o, (X_var_o.shape[0]*nlo))
                lat = lat_og[lat_inds]
                lon = lon_og
            else:
                lat_max = np.where(lat_og>=lat_cutoff)[0].max()+1
                lat_min = np.where(lat_og>=lat_cutoff)[0].min()
                lon_max = np.where(lat_og>=lat_cutoff)[1].max()+1
                lon_min = np.where(lat_og>=lat_cutoff)[1].min()
                nla = lat_og.shape[0]
                nlo = lat_og.shape[1]

                X_var_3d = np.reshape(X_var,(nla,nlo))
                X_var_o = X_var_3d[lat_min:lat_max,lon_min:lon_max]
                lat = lat_og[lat_min:lat_max,lon_min:lon_max]
                lon = lon_og[lat_min:lat_max,lon_min:lon_max]
                
                nlat_new = lat.shape[0]
                nlon_new = lat.shape[1]
                X_var_out = np.reshape(X_var_o,(nlat_new*nlon_new))

        elif len(vardims)>=3:
            if len(lat_og.shape)<=1:
                lat_inds = np.where(lat_og>=lat_cutoff)[0]
                nlo = lon_og.shape[0]
                nla = lat_og.shape[0]
                X_var_3d = np.reshape(X_var,(nla,nlo,X_var.shape[1]))

                X_var_o = X_var_3d[lat_inds,:,:]
                X_var_out = np.reshape(X_var_o, (X_var_o.shape[0]*nlo,X_var.shape[1]))
                lat = lat_og[lat_inds]
                lon = lon_og
            else:
                lat_max = np.where(lat_og>=lat_cutoff)[0].max()+1
                lat_min = np.where(lat_og>=lat_cutoff)[0].min()
                lon_max = np.where(lat_og>=lat_cutoff)[1].max()+1
                lon_min = np.where(lat_og>=lat_cutoff)[1].min()

                X_var_3d = np.reshape(X_var,(lat_og.shape[0],lat_og.shape[1],X_var.shape[1]))
                X_var_o = X_var_3d[lat_min:lat_max,lon_min:lon_max,:]
                lat = lat_og[lat_min:lat_max,lon_min:lon_max]
                lon = lon_og[lat_min:lat_max,lon_min:lon_max]
                
                nlat_new = lat.shape[0]
                nlon_new = lat.shape[1]
                X_var_out = np.reshape(X_var_o,(nlat_new*nlon_new,X_var.shape[1]))
    else: 
        X_var_out = X_var
        
        print('No latitude cutoff detected.')
        if len(data_in.lat.values.shape)>=3:
            lat = data_in.lat.values[:,:,0]
            lon = data_in.lon.values[:,:,0]
        else: 
            lat = data_in.lat.values
            lon = data_in.lon.values
            
    if len(lat.shape)<2: 
        nlat = lat.shape[0]
        nlon = lon.shape[0]
        ndof = nlat*nlon
    else: 
        nlat = lat.shape[0]
        nlon = lat.shape[1]
        ndof = nlat*nlon

    # save location indices for each variable
    d = {}
#    d['varind'] = k
    
    d['lat'] = lat
    d['lon'] = lon
    if var == 'areacello':
        d['units'] = data_in.areacello.units
    elif var == 'areacella':
        d['units'] = data_in.areacella.units
    
    if len(vardims)>=3:
        d['time'] = data_in.time.values

    d['var_ndof'] = ndof
    d['climo'] = climo
    var_dict[var] = d

    elapsed_time = timestamp.time() - begin_time
    if verbose is True: 
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')
            
    return X_var_out, var_dict

def load_data(var, var_dict, fdic, remove_climo=True, detrend=True, atol=False, 
              verbose=True, cmip6=False, tscut=False, tecut=False, lat_cutoff = False): 
    """
    Does lat cutoff before removing climotology and detrending. 
    
    INPUTS:
    =========
    limvars:   list of strings with variable names 
    fdic:   dictionary of variables names, location and filenames 
            (results from build_training_dic())
    remove_climo: True/False whether anomalies are returned or not
    detrend:  True/False whether detrended data is returned or not 
    verbose:  True/False whether print statements or not
    cmip6: True/False whether to use xr.open_mfdataset() or xr.open_dataset()
    tcut: number of years to exclude from the eof decomposition
    
    OUTPUTS: 
    ==========
    X_all:    concatenated array of all variables in limvars, 
              stacked along 1D spatial dimension
    var_dict: dictionary with variables as keys, contains index location in X_all, 
              lat values, lon values, and 1D spatial dimension (number of DOF) 
              for each variable. 
    """
    
    begin_time = timestamp.time()
    if verbose is True: 
        print('Loading from '+var)
        
    print('Loading from '+ fdic[var][-60:])
    if cmip6 is True: 
        data_in = xr.open_mfdataset(fdic[var])
        data_in = data_in.drop_vars('lon')
        data_in = data_in.drop_vars('lat')
        data_in = data_in.drop_vars('vertices_latitude')
        data_in = data_in.drop_vars('vertices_longitude')
        
        data_one = xr.open_dataset(fdic[var][:-4]+'185001-186912.nc')
        data_in = data_in.assign_coords(lon=data_one['lon'].astype('float32'))
        data_in = data_in.assign_coords(lat=data_one['lat'].astype('float32'))
        data_in.load()
        
    else: 
        data_in = xr.open_dataset(fdic[var])

#     if var is 'zg':
#         data_in = data_in.sel(plev=5e4) 
        
#     if 'LME' in fdic[var]:
#         print('LME detected')
#         data_in = data_in.sel(member=1)
#         if 'tos' in var: 
#             data_in = data_in.isel(time=slice(12,13872))
#         elif 'sic' in var: 
#             data_in = data_in.isel(time=slice(11,11999))
    
    # Fill pole hole for sea ice in satellite data: 
    if (var is 'sic') & ('Observations' in fdic['fpath']):
        data_nans= data_in.where(data_in[var]<1.1)
        data_in = data_nans.where(data_nans.lat<84.2,1.0)
        
    if 'goddard' in fdic[var]: 
        t = data_in.time.values
        t[107] = np.datetime64('1987-12-01')
        t[108] = np.datetime64('1988-01-01')

        #data_in_old['time'] = t
        data_in = data_in.assign_coords(time=t)

        data_in.sic[107,:,:] = np.nan
        data_in.sic[108,:,:] = np.nan
        
    # Select time period of interest: 
    if tscut and tecut:
        tscut_tot = (12*tscut)
        tecut_tot = (12*tecut)
        data_in = data_in.isel(time=slice(tscut_tot,tecut_tot))
        if 'datetime64' in str(type(data_in.time.values[0])):
            print('time dimension: '+str(data_in.time.values[0].astype('M8[Y]'))+' - '+
                  str(data_in.time.values[-1].astype('M8[Y]')))
#            print(data_in.time.shape)
        else: 
            print('time dimension: '+str(data_in.time.values[0].year)+' - '+
                  str(data_in.time.values[-1].year))
#            print(data_in.time.shape)
    elif tecut: 
        tecut_tot = (12*tecut)
        data_in = data_in.isel(time=slice(0,tecut_tot))
        if 'datetime64' in str(type(data_in.time.values[0])):
            print('time dimension: '+str(data_in.time.values[0].astype('M8[Y]'))+' - '+
                  str(data_in.time.values[-1].astype('M8[Y]')))
#             print(data_in.time.shape)
        else: 
            print('time dimension: '+str(data_in.time.values[0].year)+' - '+
                  str(data_in.time.values[-1].year))
#             print(data_in.time.shape)
    elif tscut: 
        tscut_tot = (12*tscut)
        data_in = data_in.isel(time=slice(tscut_tot,data_in.time.shape[0]))
        if 'datetime64' in str(type(data_in.time.values[0])):
            print('time dimension: '+str(data_in.time.values[0].astype('M8[Y]'))+' - '+
                  str(data_in.time.values[-1].astype('M8[Y]')))
#             print(data_in.time.shape)
        else: 
            print('time dimension: '+str(data_in.time.values[0].year)+' - '+
                  str(data_in.time.values[-1].year))
#             print(data_in.time.shape)
        
    vardims = list(data_in[var].dims)
    
    if len(vardims)<=2:
        nlat = data_in[vardims[0]].shape[0]
        nlon = data_in[vardims[1]].shape[0]
        ndof = nlat*nlon
    elif len(vardims)>=3:
        ntime = data_in[vardims[0]].shape[0]
        nlat = data_in[vardims[1]].shape[0]
        nlon = data_in[vardims[2]].shape[0]
        ndof = nlat*nlon

    x_var = data_in[var].values
    
    ### Do regridding here ###
    
    if len(vardims)<=2:
        x_var_2d = np.reshape(x_var,[ndof]) 
    else: 
        x_var_2d = np.moveaxis(np.reshape(x_var,[ntime,ndof]),0,-1) 
    
    if lat_cutoff:
        print('latitude cutoff detected: '+str(lat_cutoff))
        if len(data_in.lat.values.shape)>=3:
            lat_og = data_in.lat.values[:,:,0]
            lon_og = data_in.lon.values[:,:,0]
        else: 
            lat_og = np.where(data_in.lat.values>90,np.nan,data_in.lat.values)
            lon_og = np.where(data_in.lon.values>360,np.nan,data_in.lon.values)

        if len(vardims)<=2:
            if len(lat_og.shape)<=1:
                lat_inds = (lat_og>=lat_cutoff)
                nlo = lon_og.shape[0]
                nla = lat_og.shape[0]
                X_var_3d = np.reshape(x_var_2d,(nla,nlo))

                X_var_o = X_var_3d[lat_inds,:]
                X_var_lc = np.reshape(X_var_o, (X_var_o.shape[0]*nlo))
                lat = lat_og[lat_inds]
                lon = lon_og
            else:
                lat_max = np.where(lat_og>=lat_cutoff)[0].max()+1
                lat_min = np.where(lat_og>=lat_cutoff)[0].min()
                lon_max = np.where(lat_og>=lat_cutoff)[1].max()+1
                lon_min = np.where(lat_og>=lat_cutoff)[1].min()
                nla = lat_og.shape[0]
                nlo = lat_og.shape[1]

                X_var_3d = np.reshape(x_var_2d,(nla,nlo))
                X_var_o = X_var_3d[lat_min:lat_max,lon_min:lon_max]
                lat = lat_og[lat_min:lat_max,lon_min:lon_max]
                lon = lon_og[lat_min:lat_max,lon_min:lon_max]
                
                nlat_new = lat.shape[0]
                nlon_new = lat.shape[1]
                X_var_lc = np.reshape(X_var_o,(nlat_new*nlon_new))

        elif len(vardims)>=3:
            if len(lat_og.shape)<=1:
                lat_inds = np.where(lat_og>=lat_cutoff)[0]
                nlo = lon_og.shape[0]
                nla = lat_og.shape[0]
                X_var_3d = np.reshape(x_var_2d,(nla,nlo,x_var_2d.shape[1]))

                X_var_o = X_var_3d[lat_inds,:,:]
                X_var_lc = np.reshape(X_var_o, (X_var_o.shape[0]*nlo,x_var_2d.shape[1]))
                lat = lat_og[lat_inds]
                lon = lon_og
            else:
                lat_max = np.where(lat_og>=lat_cutoff)[0].max()+1
                lat_min = np.where(lat_og>=lat_cutoff)[0].min()
                lon_max = np.where(lat_og>=lat_cutoff)[1].max()+1
                lon_min = np.where(lat_og>=lat_cutoff)[1].min()

                X_var_3d = np.reshape(x_var_2d,(lat_og.shape[0],lat_og.shape[1],x_var_2d.shape[1]))
                X_var_o = X_var_3d[lat_min:lat_max,lon_min:lon_max,:]
                lat = lat_og[lat_min:lat_max,lon_min:lon_max]
                lon = lon_og[lat_min:lat_max,lon_min:lon_max]
                
                nlat_new = lat.shape[0]
                nlon_new = lat.shape[1]
                X_var_lc = np.reshape(X_var_o,(nlat_new*nlon_new,x_var_2d.shape[1]))
    else: 
        X_var_lc = x_var
        
        print('No latitude cutoff detected.')
        if len(data_in.lat.values.shape)>=3:
            lat = data_in.lat.values[:,:,0]
            lon = data_in.lon.values[:,:,0]
        else: 
            lat = data_in.lat.values
            lon = data_in.lon.values
            
    if len(lat.shape)<2: 
        nlat = lat.shape[0]
        nlon = lon.shape[0]
        ndof = nlat*nlon
    else: 
        nlat = lat.shape[0]
        nlon = lat.shape[1]
        ndof = nlat*nlon
        

    if detrend is True: 
        print('detrending...')
        if len(vardims)<=2:
            print('No time dimension, cant detrend...')
        else: 
            x_var_dt = np.zeros((ndof,int(ntime/12),12))
            climo = np.zeros((ndof,12))
            x_var_3d = np.reshape(X_var_lc, (ndof,int(ntime/12),12))
            for i in range(12):
                print('Working on month...'+str(i))
                y = x_var_3d[:,:,i]
                if ('goddard' in fdic[var]) & (np.isnan(y).sum()>0):
                    print('Found some nans in satellite data, going to fill with previous timestep...')
                    inds = np.where(np.isnan(y))
                    ind_int = int(inds[1].min()-1)
                    fill = np.ones(y.shape)*y[:,ind_int][:,np.newaxis]
                    var_nans_mask = np.where(np.isnan(y),np.nan,1)
                    Y = np.where(np.isnan(y),fill,y)
                elif np.isnan(y).sum()>0: 
                    print('Found with nans (not in satellite), going to fill with zeros...')
                    var_nans_mask = np.where(np.isnan(y),np.nan,1)
                    Y = np.where(np.isnan(y),0,y)
                else: 
                    Y = y
    #                print('Y = '+ str(Y.shape))
                X = np.arange(0,int(ntime/12))
                [var_dt,_,intercept] = kbstats.multi_linear_detrend(X,Y,axis=1,atol=False,
                                                                    remove_mn=remove_climo)
                if np.isnan(y).sum()>0:
                    x_var_dt[:,:,i] = var_dt*var_nans_mask
                else: 
                    x_var_dt[:,:,i] = var_dt
                climo[:,i] = intercept

            x_var_anom = np.reshape(x_var_dt,(ndof,ntime))           
    else: 
        x_var_dt = X_var_lc
        if remove_climo is True: 
            print('removing climotology...')
            x_var_3d = np.reshape(x_var_dt,(ndof,int(ntime/12),12))
            climo = np.nanmean(x_var_3d,axis=1)
            x_var_anom = x_var_3d - climo[:,np.newaxis,:]
            x_var_anom = np.reshape(x_var_anom,(ndof,ntime))
        else: 
            x_var_anom = x_var_dt
            climo = 'None'

    X_var_out = x_var_anom

    # save location indices for each variable
    d = {}
#    d['varind'] = k
    
    d['lat'] = lat
    d['lon'] = lon
    if var == 'areacello':
        d['units'] = data_in.areacello.units
    elif var == 'areacella':
        d['units'] = data_in.areacella.units
    
    if len(vardims)>=3:
        d['time'] = data_in.time.values

    d['var_ndof'] = ndof
    d['climo'] = climo
    var_dict[var] = d

    elapsed_time = timestamp.time() - begin_time
    if verbose is True: 
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')
            
    return X_var_out, var_dict


def load_sat_data(var, var_ln,var_dict, fdic, remove_climo=True, detrend=True, verbose=True): 
    """
    INPUTS:
    =========
    limvars:   list of strings with variable names 
    fdic:   dictionary of variables names, location and filenames 
            (results from build_training_dic())
    remove_climo: True/False whether anomalies are returned or not
    detrend:  True/False whether detrended data is returned or not 
    verbose:  True/False whether print statements or not
    
    OUTPUTS: 
    ==========
    X_all:    concatenated array of all variables in limvars, 
              stacked along 1D spatial dimension
    var_dict: dictionary with variables as keys, contains index location in X_all, 
              lat values, lon values, and 1D spatial dimension (number of DOF) 
              for each variable. 
    """

    begin_time = timestamp.time()
    if verbose is True: 
        print('Loading '+var)

    data_in = xr.open_dataset(fdic[var])

    if var is 'zg':
        data_in = data_in.sel(plev=5e4) 
        
    data_nans= data_in[var_ln].where(data_in[var_ln]<1.1)
    data_ph = data_nans.where(data_nans.latitude<84.7,1.0)
        
    vardims = list(data_ph.dims)
    
    if len(vardims)<=2:
        nlat = data_in[vardims[0]].shape[0]
        nlon = data_in[vardims[1]].shape[0]
        ndof = nlat*nlon
    elif len(vardims)>=3:
        ntime = data_in[vardims[0]].shape[0]
        nlat = data_in[vardims[1]].shape[0]
        nlon = data_in[vardims[2]].shape[0]
        ndof = nlat*nlon
        
    if remove_climo is True: 
        print('removing climotology...')
        climo = data_ph.groupby('time.month').mean(dim='time')
        x_var_anom = data_ph.groupby('time.month')- climo

        x_var = x_var_anom.values
    else: 
        climo = 'None'
        x_var = data_ph.values

#     if detrend is True: 
#         print('detrending...')
#         var_nans_mask = np.where(np.isnan(x_var),np.nan,1)
#         var_dt = spy.signal.detrend(np.where(np.isnan(x_var),0,x_var),axis=0)
#         x_var_dt = var_dt*var_nans_mask
#     else: 
#         x_var_dt = x_var

    if len(vardims)<=2:
        x_var_2d = np.reshape(x_var,[ndof]) 
    else: 
        x_var_2d = np.moveaxis(np.reshape(x_var,[ntime,ndof]),0,-1) 
        
    if detrend is True: 
        print('detrending...')
        var_nans_mask = np.where(np.isnan(x_var_2d),np.nan,1)
        Y = np.where(np.isnan(x_var_2d),0,x_var_2d)
        print(Y.shape)
        X = np.arange(0,data_in.time.shape[0])
        [var_dt,_,_] = kbstats.multi_linear_detrend(X,Y,axis=1,remove_mn=True)
        x_var_dt = var_dt*var_nans_mask
    else: 
        x_var_dt = x_var
        
    X_var = x_var_dt

    # save location indices for each variable
    d = {}
#    d['varind'] = k
    
    if len(vardims)<=2:
        d['lat'] = data_in.latitude.values
        d['lon'] = data_in.longitude.values
#         if var == 'areacello':
#             d['units'] = data_in.areacello.units
#         elif var == 'areacella':
#             d['units'] = data_in.areacella.units
    elif len(vardims)>=3:
        d['lat'] = data_in.latitude.values
        d['lon'] = data_in.longitude.values
        d['time'] = data_in.time.values
        
    d['var_ndof'] = ndof
    d['climo'] = climo
    var_dict[var] = d

    elapsed_time = timestamp.time() - begin_time
    if verbose is True: 
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')
            
    return X_var, var_dict


# def load_data(limvars, fdic, remove_climo=True, detrend=True, verbose=True): 
#     """
#     INPUTS:
#     =========
#     limvars:   list of strings with variable names 
#     fdic:   dictionary of variables names, location and filenames 
#             (results from build_training_dic())
#     remove_climo: True/False whether anomalies are returned or not
#     detrend:  True/False whether detrended data is returned or not 
#     verbose:  True/False whether print statements or not
    
#     OUTPUTS: 
#     ==========
#     X_all:    concatenated array of all variables in limvars, 
#               stacked along 1D spatial dimension
#     var_dict: dictionary with variables as keys, contains index location in X_all, 
#               lat values, lon values, and 1D spatial dimension (number of DOF) 
#               for each variable. 
#     """
#     # load training data...
#     var_dict = {}
#     #
#     for k, var in enumerate(limvars): 
#         begin_time = timestamp.time()
#         if verbose is True: 
#             print('Loading '+var)
            
#         data_in = xr.open_dataset(fdic[var])

#         if var is 'zg':
#             data_in = data_in.sel(plev=5e4)    
#         vardims = list(data_in[var].dims)
#         ntime = data_in[vardims[0]].shape[0]
#         nlat = data_in[vardims[1]].shape[0]
#         nlon = data_in[vardims[2]].shape[0]
#         ndof = nlat*nlon

#         if remove_climo is True: 
#             climo = data_in[var].groupby('time.month').mean(dim='time')
#             x_var_anom = data_in[var].groupby('time.month')- climo

#             x_var = x_var_anom.values
#         else: 
#             x_var = data_in[var].values

#         if detrend is True: 
#             var_nans_mask = np.where(np.isnan(x_var),np.nan,1)
#             var_dt = spy.signal.detrend(np.where(np.isnan(x_var),0,x_var),axis=0)
#             x_var_dt = var_dt*var_nans_mask
#         else: 
#             x_var_dt = x_var

#         X_var = np.moveaxis(np.reshape(x_var_dt,[ntime,ndof]),0,-1) 

#         # save location indices for each variable
#         d = {}
#         d['varind'] = k
#         d['lat'] = data_in.lat.values
#         d['lon'] = data_in.lon.values
#         d['time'] = data_in.time.values
#         d['var_ndof'] = ndof
#         var_dict[var] = d

#         elapsed_time = timestamp.time() - begin_time
#         if verbose is True: 
#             print('-----------------------------------------------------')
#             print('completed in ' + str(elapsed_time) + ' seconds')
#             print('-----------------------------------------------------')

#         if k == 0: 
#             X_all = X_var
#         else: 
#             X_all = np.concatenate((X_all,X_var),axis=0)
            
#     return X_all, var_dict
    
    
def eof_decomp_1var(X,ndof,ntime,ntrunc,areawt=0,Weight=True): 
    """
    INPUTS: 
    ========
    X = (ndof, ntime)
    ndof = integer value (nlat x nlon)
    ntime = integer value (number of timesteps)
    ntrunc = integer value (number of single variable eof modes to retain)
    Weights = True/False, indicates whether to weight by cell area or not
    
    OUTPUTS: 
    =========
    eofs_out
    svals_out
    pcs_out
    total_var
    tot_var_eig
    var_expl_by_retained
    W
    """
    
    if Weight is True: 
        if len(areawt.shape)<=1:
            W = np.nan_to_num(areawt[:,np.newaxis])
        else: 
            W = np.nan_to_num(areawt)
        inp = W*np.nan_to_num(X)
        
    else: 
        inp = np.nan_to_num(X)

    u,s,v = np.linalg.svd(inp,full_matrices=False)

    eofs_out = u[:,:ntrunc]
    svals_out = s[:ntrunc]
    pcs_out = v[:ntrunc]

    eig_vals = (svals_out**2)
    total_var = np.nansum(np.nanvar(X,ddof=1,axis=1))
    tot_var_eig = np.sum(s*s)
    var_expl_by_retained = 100*np.sum(eig_vals)/tot_var_eig
    
    if Weight is True: 
        return eofs_out, svals_out, pcs_out, total_var, tot_var_eig, var_expl_by_retained, W
    else: 
        return eofs_out, svals_out, pcs_out, total_var, tot_var_eig, var_expl_by_retained

def eof_decomp_1var_sqrt(X,ndof,ntime,ntrunc,areawt=0,Weight=True): 
    """
    INPUTS: 
    ========
    X = (ndof, ntime)
    ndof = integer value (nlat x nlon)
    ntime = integer value (number of timesteps)
    ntrunc = integer value (number of single variable eof modes to retain)
    Weights = True/False, indicates whether to weight by cell area or not
    
    OUTPUTS: 
    =========
    eofs_out
    svals_out
    pcs_out
    total_var
    tot_var_eig
    var_expl_by_retained
    W
    """
    
    if Weight is True: 
        if len(areawt.shape)<=1:
            W = np.sqrt(np.nan_to_num(areawt[:,np.newaxis]))
        else: 
            W = np.sqrt(np.nan_to_num(areawt))
        inp = W*np.nan_to_num(X)
        
    else: 
        inp = np.nan_to_num(X)

    u,s,v = np.linalg.svd(inp,full_matrices=False)

    eofs_out = u[:,:ntrunc]
    svals_out = s[:ntrunc]
    pcs_out = v[:ntrunc]

    eig_vals = (svals_out**2)
    total_var = np.nansum(np.nanvar(X,ddof=1,axis=1))
    tot_var_eig = np.sum(s*s)
    var_expl_by_retained = 100*np.sum(eig_vals)/tot_var_eig
    
    if Weight is True: 
        return eofs_out, svals_out, pcs_out, total_var, tot_var_eig, var_expl_by_retained, W
    else: 
        return eofs_out, svals_out, pcs_out, total_var, tot_var_eig, var_expl_by_retained
    
def step1_compress_individual_vars(X_train, limvars, ntrunc, nmodes_sic, var_dict, 
                                   X_allshape, X_sicshape, wt, sic_separate=False): 
    """
    INPUTS: 
    =========
    X_train
    limvars
    ntrunc
    nmodes_sic
    var_dict
    X_allshape
    X_sicshape
    wt
    sic_separate=False
    
    OUTPUTS: 
    =========
    Ptrunc
    E3
    Ptrunc_sic
    E_sic
    W_all
    standard_factor
    tot_var
    tot_var_eig
                                   
    """
    
    nvars = len(limvars)
    W_all = {}
    standard_factor = {}
    tot_var = {}
    tot_var_eig = {}
    
    if sic_separate is True: 
        E3 = np.zeros([X_allshape-X_sicshape,ntrunc*(nvars-1)])
    else:  
        E3 = np.zeros([X_allshape,ntrunc*(nvars)])
    n=0

    for k,var in enumerate(limvars):
        print('decomposing...',var)
        # weight matrix for equal-area covariance normalization
        ndof = var_dict[var]['var_ndof']
        ntime = X_train.shape[1]
        LAT = var_dict[var]['lat']
        if len(LAT.shape)<2:
            nlon = var_dict[var]['lon'].shape[0]
            nlat = var_dict[var]['lat'].shape[0]
            lat = LAT[:,np.newaxis]*np.ones((nlat,nlon))
        else: 
            lat = LAT
            
        if (sic_separate is True) & (var is 'sic'):
            trunc = nmodes_sic
        else: 
            trunc = ntrunc
        
        [eofs_out, svals_out, pcs_out, 
         total_var, total_var_eig, 
         var_expl_by_retained, W] = eof_decomp_1var(X_train[var_dict[var]['var_inds'],:],
                                                    lat,ndof,ntime,trunc,Weight=wt)
        tot_var[var] = total_var
        tot_var_eig[var] = total_var_eig
        W_all[var] = np.squeeze(W)
        
        if (sic_separate is True) & (var is 'sic'):
            print('fraction in first '+str(nmodes_sic)+ ' '+limvars[k]+' EOFs = '+str(var_expl_by_retained))
        else: 
            print('fraction in first '+str(ntrunc)+ ' '+limvars[k]+' EOFs = '+str(var_expl_by_retained))


        if k == 0:
            # projection
            P_var = np.matmul(eofs_out.T,W*np.nan_to_num(X_train[var_dict[var]['var_inds'],:]))
            standard_factor[var] = np.sqrt(np.sum(svals_out*svals_out)/(ntime-1))
            
            Ptrunc = P_var/standard_factor[var]

            # reverse operator from EOFs to grid point space
            E3[var_dict[var]['var_inds'],k*ntrunc:(k+1)*ntrunc] = eofs_out*standard_factor[var]
            n=n+1
        elif (sic_separate is True) & (var is 'sic'):
            print('...separately')
            Pvar_sic = np.matmul(eofs_out.T,W*np.nan_to_num(X_train[var_dict[var]['var_inds'],:]))
            standard_factor[var] = np.sqrt(np.sum(svals_out*svals_out)/(ntime-1))
            
            Ptrunc_sic = Pvar_sic/standard_factor[var]
            E_sic = eofs_out*standard_factor[var]
        else:
            # projection
            P_var = np.matmul(eofs_out.T,W*np.nan_to_num(X_train[var_dict[var]['var_inds'],:]))
            standard_factor[var] = np.sqrt(np.sum(svals_out*svals_out)/(ntime-1))
            
            Ptrunc = np.concatenate((Ptrunc,P_var/standard_factor[var]),axis=0)

            # reverse operator from EOFs to grid point space
            E3[var_dict[var]['var_inds'],n*ntrunc:(n+1)*ntrunc] = eofs_out*standard_factor[var]
            n=n+1
    if sic_separate is True: 
        return Ptrunc, E3, Ptrunc_sic, E_sic, W_all, standard_factor, tot_var, tot_var_eig
    else: 
        return Ptrunc, E3, W_all, standard_factor, tot_var, tot_var_eig
    
    
def step1_compress_individual_var(X_train, var, ntrunc, nmodes_sic, var_dict,areawt=0,
                                  wt=True, sic_separate=False): #, return_eig=False): 
    """
    INPUTS: 
    =======
    X_train = array containing variable to be compressed (nlat*nlon, ntime)
    var = string indicating variable name
    ntrunc = number of modes retained for multivarivariate decomposition (integer)
    nmodes_sic = number of modes retained for sic (integer)
    var_dict = dictionary containing lat, lon, time, number of degrees of freedom
               and indice locations for variables. 
    areawt = cell area values for variable (used only if wt is True), nlat*nlat
    wt = area weight eof decomposition? (Tru or False)
    sic_separate = Process sea ice concentration separate or with other variables (True or False)
    
    OUTPUTS: 
    ========
    Ptrunc = projected, standardized and truncated variable (neofs,ntime)
    E3 = reverse operator from EOF to grid point space (also undoes standardization)
    tot_var = 
    tot_var_eig = 
    W_all = latitude weights 
    standard_factor = standardization factor to make total variance equal 1
    """
    if (sic_separate is True) & (var is 'sic'):
        trunc = nmodes_sic
    else: 
        trunc = ntrunc
    print('truncating to '+str(trunc)) 
        
    if len(var_dict[var]['lat'].shape)<2:
        nlon = var_dict[var]['lon'].shape[0]
        nlat = var_dict[var]['lat'].shape[0]
        lat = var_dict[var]['lat'][:,np.newaxis]*np.ones((nlat,nlon))
    else: 
        lat = var_dict[var]['lat']
    
    if wt is True: 
        [eofs_out, svals_out, pcs_out, 
         tot_var, tot_var_eig, 
         var_expl_by_retained, W] = eof_decomp_1var(X_train,var_dict[var]['var_ndof'],
                                                    X_train.shape[1],trunc,areawt=areawt,Weight=wt)
        W_all = np.squeeze(W)
        
        # Projection
        P_var = np.matmul(eofs_out.T,W*np.nan_to_num(X_train))
    else: 
        [eofs_out, svals_out, pcs_out, 
         tot_var, tot_var_eig, 
         var_expl_by_retained] = eof_decomp_1var(X_train,var_dict[var]['var_ndof'],
                                                   X_train.shape[1],trunc,areawt=areawt,Weight=wt)
        
        W_all = 0
        
        # Projection
        P_var = np.matmul(eofs_out.T,np.nan_to_num(X_train))
    
    ntime = X_train.shape[1]
    print('fraction in first '+str(trunc)+ ' '+var+' EOFs = '+str(var_expl_by_retained))

    standard_factor = np.sqrt(np.sum(np.var(P_var,axis=1)))
#    standard_factor = np.sqrt(np.sum(svals_out*svals_out)/(ntime-1))
    Ptrunc = P_var/standard_factor

    # reverse operator from EOFs to grid point space
    E3 = eofs_out*standard_factor
        
    return Ptrunc, E3, tot_var, tot_var_eig, W_all, standard_factor, var_expl_by_retained

def step1_compress_individual_var_sqrt(X_train, var, ntrunc, nmodes_sic, var_dict,areawt=0,
                                       wt=True, sic_separate=False): #, return_eig=False): 
    """
    INPUTS: 
    =======
    X_train = array containing variable to be compressed (nlat*nlon, ntime)
    var = string indicating variable name
    ntrunc = number of modes retained for multivarivariate decomposition (integer)
    nmodes_sic = number of modes retained for sic (integer)
    var_dict = dictionary containing lat, lon, time, number of degrees of freedom
               and indice locations for variables. 
    areawt = cell area values for variable (used only if wt is True), nlat*nlat
    wt = area weight eof decomposition? (Tru or False)
    sic_separate = Process sea ice concentration separate or with other variables (True or False)
    
    OUTPUTS: 
    ========
    Ptrunc = projected, standardized and truncated variable (neofs,ntime)
    E3 = reverse operator from EOF to grid point space (also undoes standardization)
    tot_var = 
    tot_var_eig = 
    W_all = latitude weights 
    standard_factor = standardization factor to make total variance equal 1
    """
    if (sic_separate is True) & (var is 'sic'):
        trunc = nmodes_sic
    else: 
        trunc = ntrunc
    print('truncating to '+str(trunc)) 
        
    if len(var_dict[var]['lat'].shape)<2:
        nlon = var_dict[var]['lon'].shape[0]
        nlat = var_dict[var]['lat'].shape[0]
        lat = var_dict[var]['lat'][:,np.newaxis]*np.ones((nlat,nlon))
    else: 
        lat = var_dict[var]['lat']
    
    if wt is True: 
        [eofs_out, svals_out, pcs_out, 
         tot_var, tot_var_eig, 
         var_expl_by_retained, W] = eof_decomp_1var_sqrt(X_train,var_dict[var]['var_ndof'],
                                                         X_train.shape[1],trunc,areawt=areawt,Weight=wt)
        W_all = np.squeeze(W)
        
        # Projection
        P_var = np.matmul(eofs_out.T,W*np.nan_to_num(X_train))
    else: 
        [eofs_out, svals_out, pcs_out, 
         tot_var, tot_var_eig, 
         var_expl_by_retained] = eof_decomp_1var_sqrt(X_train,var_dict[var]['var_ndof'],
                                                      X_train.shape[1],trunc,areawt=areawt,Weight=wt)
        
        W_all = 0
        
        # Projection
        P_var = np.matmul(eofs_out.T,np.nan_to_num(X_train))
    
    ntime = X_train.shape[1]
    print('fraction in first '+str(trunc)+ ' '+var+' EOFs = '+str(var_expl_by_retained))

    standard_factor = np.sqrt(np.sum(np.var(P_var,axis=1)))
#    standard_factor = np.sqrt(np.sum(svals_out*svals_out)/(ntime-1))
    Ptrunc = P_var/standard_factor

    # reverse operator from EOFs to grid point space
    E3 = eofs_out*standard_factor
        
    return Ptrunc, E3, tot_var, tot_var_eig, W_all, standard_factor, var_expl_by_retained
    
    
def step2_multivariate_compress(Ptrunc,nmodes, E3, Ptrunc_sic, sic_separate=False, Trunc_truth=False): 
    """
    INPUTS: 
    ========
    Ptrunc
    nmodes
    E3
    Ptrunc_sic
    sic_separate=False
    
    OUTPUTS: 
    ========
    Ptrain
    Fvar
    E
    """
    # truncate the coupled covariance matrix
    U,S,V = np.linalg.svd(Ptrunc,full_matrices=False)
    Etrunc = U[:,0:nmodes]

    Fvar = 100*np.sum(S[:nmodes]*S[:nmodes])/np.sum(S*S)
    print('Fraction in first '+str(nmodes)+ ' multivariate EOFs = '+str(Fvar))

    P = np.matmul(Etrunc.T,Ptrunc)
    # reverse operator from *truncated* EOF space to full grid point space
    E = np.matmul(E3,Etrunc)
    print('Shape of E: '+ str(E.shape))

    if sic_separate is True: 
        P_train = np.concatenate((P,Ptrunc_sic),axis=0)
    #    E_train = np.concatenate((E,E_sic),axis=1)
    else: 
        P_train = P
    
    if Trunc_truth is False: 
        return P_train, Fvar, E
    else: 
        return P_train, Fvar, Etrunc

    
# def unweight_decompressed_vars(x_train_dcomp, limvars, var_dict, W_all):
#     """
#     INPUTS: 
#     ========
#     x_train_dcomp
#     limvars
#     var_dict
#     W_all
    
#     OUTPUTS: 
#     ========
#     X_out 
#     """
#     X_out = np.zeros_like(x_train_dcomp)
    
#     for var in (limvars):
#         inds = var_dict[var]['var_inds']
#         X_out[inds,:] = x_train_dcomp[inds,:]/W_all[var][:,np.newaxis]
        
#     return X_out


def unweight_decompressed_vars(x_train_dcomp, limvars, var_dict, W_all):
    """
    INPUTS: 
    ========
    x_train_dcomp
    limvars
    var_dict
    W_all
    
    OUTPUTS: 
    ========
    X_out 
    """
    X_out = np.zeros_like(x_train_dcomp)
    
    start=0
    if len(x_train_dcomp.shape)<2:
        for var in (limvars):
            inds_end = var_dict[var]['var_ndof']
            X_out[start:start+inds_end] = x_train_dcomp[start:start+inds_end]/W_all[var][:]
            start = start+inds_end
    else:
        for var in (limvars):
            inds_end = var_dict[var]['var_ndof']
            X_out[start:start+inds_end,:] = x_train_dcomp[start:start+inds_end,:]/W_all[var][:,np.newaxis]
            start = start+inds_end
        
    return X_out


def decompress_eof_separate_sic(P_train,nmodes,nmodes_sic,E,E_sic,
                                limvars,var_dict,W_all,Weights=True,
                                sic_separate=False):
    """
    INPUTS: 
    ========
    P_train
    nmodes
    nmodes_sic
    E
    E_sic
    limvars
    var_dict
    W_all
    Weights=True
    sic_separate=False
    
    OUTPUTS: 
    ========
    X_train_dcomp
    """
    if sic_separate is True: 
#        print('sic_separate is True.')
        if len(limvars)<=1:
            print('only one variable detected.')
            x_sic = P_train

            x_train_dcomp  = np.matmul(E_sic,x_sic)
        else: 
            x_multivar = P_train[0:nmodes,:]
            x_sic = P_train[-nmodes_sic:,:]

            x_train_multi_dcomp = np.matmul(E,x_multivar)
            x_train_sic_dcomp = np.matmul(E_sic,x_sic)
            x_train_dcomp = np.concatenate((x_train_multi_dcomp,x_train_sic_dcomp),axis=0)
    else: 
        x_multivar = P_train

        x_train_dcomp = np.matmul(E,x_multivar)

    if Weights is True: 
        X_train_dcomp = unweight_decompressed_vars(x_train_dcomp, limvars, var_dict, W_all)
    else: 
        X_train_dcomp = x_train_dcomp
    
    return X_train_dcomp


def count_ndof_all(limvars, E3, sic_separate=False): 
    #Count total degrees of freedom: 
    if sic_separate is True: 
        if len(limvars) <2: 
            ndof_all = 0
        else: 
            limvars_nosic = [l for l in limvars if l not in 'sic']
            for v,var in enumerate(limvars_nosic):
                if v == 0: 
                    ndof_all = E3[var].shape[0]
                else: 
                    ndof_all = ndof_all+E3[var].shape[0]
    else: 
        for v,var in enumerate(limvars):
            if v == 0: 
                ndof_all = E3[var].shape[0]
            else: 
                ndof_all = ndof_all+E3[var].shape[0]
            print(ndof_all)
            
    return ndof_all


def stack_variable_eofs(limvars, ndof_all, ntrunc, Ptrunc, E3, 
                        var_dict,nmonths=1,sic_separate=False, 
                        verbose=False): 
    start = 0
    if sic_separate is True: 
        limvars_nosic = [l for l in limvars if l not in 'sic']
        nvars = len(limvars_nosic)
        E3_all = np.zeros([ndof_all,int(ntrunc*(nvars)),nmonths])
        if nmonths == 1: 
            E3_all = np.squeeze(E3_all)

        for v,var in enumerate(limvars_nosic):
            print(str(v) + ', '+var)
            if v == 0: 
                Ptrunc_all = Ptrunc[var]
            else: 
                Ptrunc_all = np.concatenate((Ptrunc_all,Ptrunc[var]),axis=0)
                
            E3_all[var_dict[var]['var_inds'],int(v*ntrunc):int((v+1)*ntrunc)] = E3[var]
            
            if verbose is True: 
                print(E3_all[18430:18435,0])
                print('start: '+str(start)+' end: '+str(start+var_dict[var]['var_ndof'])+' '+str(v*ntrunc))
#            start = start+var_dict[var]['var_ndof']

        Ptrunc_sic = Ptrunc['sic']
        E_sic = E3['sic']
    else: 
        nvars = len(limvars)
        E3_all = np.zeros([ndof_all,int(ntrunc*(nvars))])
        if nmonths == 1: 
            E3_all = np.squeeze(E3_all)

        for v,var in enumerate(limvars):
            if v == 0: 
                Ptrunc_all = Ptrunc[var]
            else: 
                Ptrunc_all = np.concatenate((Ptrunc_all,Ptrunc[var]),axis=0)
            E3_all[var_dict[var]['var_inds'],int(v*ntrunc):int((v+1)*ntrunc)] = E3[var]
            
            if verbose is True:
                print(E3_all[18430:18435,0])
                print('start: '+str(start)+' end: '+str(start+var_dict[var]['var_ndof'])+' '+str(v*ntrunc))
#            start = start+var_dict[var]['var_ndof']
            
    if sic_separate == True: 
        return Ptrunc_all, E3_all, Ptrunc_sic, E_sic
    else: 
        return Ptrunc_all, E3_all

    
# def LIM_forecast_Gt(LIMd,x,lags):
#     """
#     deterministic forecasting experiments for states in x and time lags in lags.

#     Inputs:
#     * LIMd: a dictionary with LIM attributes
#     * x: a state-time matrix for initial conditions and verification ~(ndof,ntims)
#     * lags: list of time lags for deterministic forecasts
#     * E: the linear map from the coordinates of the LIM to physical (lat,lon) coordinates ~(nx*ny,ndof)
    
#     Outputs (in a dictionary):
#     *'error' - error variance as a function of space and forecast lead time (ndof,ntims)
#     *'x_forecast' - the forecast states (nlags,ndof,ntims)
#     *'x_truth_phys_space' - true state in physical space (nlat*nlon,*ntims)
#     *'x_forecast_phys_space' - forecast state in physical space (nlat*nlon,*ntims)
#     """
    
#     ndof = x.shape[0]
#     ntims = x.shape[1]
#     nlags = len(lags)
#     LIMfd = {}
    
#     x_predict_save = np.zeros([nlags,ndof,ntims])
    
#     for k,t in enumerate(lags):
#         print('t=',t)
#         # make the propagator for this lead time
#         Gt = np.matmul(np.matmul(LIMd['vec'],np.diag(np.exp(LIMd['lam']*t))),LIMd['veci'])
# #        Gt = np.linalg.matrix_power(LIMd['Gt'],t)
#         # forecast
#         if t == 0:
#             # need to handle this time separately, or the matrix dimension is off
#             x_predict = np.matmul(Gt,x)
#             x_predict_save[k,:,:] = x_predict
#         else:
#             x_predict = np.matmul(Gt,x[:,:-t])
#             x_predict_save[k,:,t:] = x_predict

#         Ld = {}
#         Ld['Gt'] = Gt
#         LIMfd[t] = Ld
    
#     LIMfd['x_forecast'] = np.squeeze(x_predict_save)    
        
#     return LIMfd

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
    LIMd['C_0'] = C_0
    LIMd['C_1'] = C_1
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
    LIMd['C_0'] = C_0
    LIMd['C_1'] = C_1
    LIMd['Gt']= G
    
    return LIMd, G
    
    
def LIM_forecast(LIMd,x,lags,adjust=True):
    """
    # There is a bug with this forecast function function: It uses the eigenvectors and 
    #        values to calculate Gt, but it's giving the same value for all lags in the forecast
    
    deterministic forecasting experiments for states in x and time lags in lags.

    Inputs:
    * LIMd: a dictionary with LIM attributes
    * x: a state-time matrix for initial conditions and verification ~(ndof,ntims)
    * lags: list of time lags for deterministic forecasts
    * adj: True/False, if True removes negative eigenvalues 
    
    Outputs (in a dictionary):
    *'error' - error variance as a function of space and forecast lead time (ndof,ntims)
    *'x_forecast' - the forecast states (nlags,ndof,ntims)
    *'x_truth_phys_space' - true state in physical space (nlat*nlon,*ntims)
    *'x_forecast_phys_space' - forecast state in physical space (nlat*nlon,*ntims)
    """
    
    ndof = x.shape[0]
    ntims = x.shape[1]
    nlags = len(lags)
    LIMfd = {}
    
    x_predict_save = np.zeros([nlags,ndof,ntims])
    
    max_eigenval = np.real(LIMd['lam_L']).max()
    
    if adjust: 
        print('Adjust is True...')
        if max_eigenval >0: 
            print('YES negative eigenvalue found...adjusting')
            LIMd['frac_neg_eigenvals'] = ((LIMd['lam_L']>0).sum())/(LIMd['lam_L'].shape[0])
            LIMd['lam_L_adj'] = LIMd['lam_L'] - (max_eigenval+0.01)
        else: 
            print('NO negative eigenvalue found...')
            LIMd['frac_neg_eigenvals'] = 0
            LIMd['lam_L_adj'] = LIMd['lam_L']
    else: 
        print('Adjust is False...')
        LIMd['frac_neg_eigenvals'] = np.nan
        LIMd['lam_L_adj'] = LIMd['lam_L']
    
    for k,t in enumerate(lags):
        print('lag=',t)
        # make the propagator for this lead time
        Gt = np.matmul(np.matmul(LIMd['vec'],np.diag(np.exp(LIMd['lam_L_adj']*t))),LIMd['veci'])

        # forecast
        if t == 0:
            # need to handle this time separately, or the matrix dimension is off
            x_predict = np.matmul(Gt,x)
            x_predict_save[k,:,:] = x_predict
        else:
            x_predict = np.matmul(Gt,x[:,:-t])
            x_predict_save[k,:,t:] = x_predict

        Ld = {}
        Ld['Gt'] = Gt
        LIMfd[t] = Ld
    
    LIMfd['x_forecast'] = np.squeeze(x_predict_save)    
        
    return LIMfd


def load_truncated_data(var, mod_folder, mod_filename):
    print('Loading truncated '+var+' from: '+str(mod_folder+var+mod_filename))
    mod_data = pickle.load(open(mod_folder+var+mod_filename, "rb" ) )
    
    if 'var_dict_all' in mod_data.keys():
        var_dict = mod_data['var_dict_all']['cmip6_cesm2_hist_regridlme']
    else: 
        var_dict = mod_data['var_dict']
    X_var_trunc = mod_data['Ptrunc']
    X_var_E3 = mod_data['E3']
    X_var_standard_factor = mod_data['standard_factor']
    X_var_W_all= mod_data['W_all']
    
    return X_var_trunc, var_dict, X_var_E3, X_var_standard_factor, X_var_W_all


def step1_projection_validation_var(X_train, E3, standard_factor, W, Weights=False): 
    """
    """
    if Weights is True: 
        if len(W.shape)<2:
            W_new = W[:,np.newaxis]
        else: 
            W_new = W
        eofs_out = E3/standard_factor
        # projection
        P_var = np.matmul(eofs_out.T,W_new*np.nan_to_num(X_train))
    else: 
        eofs_out = E3/standard_factor
        # projection
        P_var = np.matmul(eofs_out.T,np.nan_to_num(X_train))

    Ptrunc = P_var/standard_factor
        
    return Ptrunc