from math import pi
import numpy as np
"""
This module contains functions to calculate BRDF scattering kernels. Equations can be found
in the following papers:
  
Colgan, M. S., Baldeck, C. A., Feret, J. B., & Asner, G. P. (2012). 
Mapping savanna tree species at ecosystem scales using support vector machine classification 
and BRDF correction on airborne hyperspectral and LiDAR data. 
Remote Sensing, 4(11), 3462-3480.            
    
Schlapfer, D., Richter, R., & Feingersh, T. (2015). 
Operational BRDF effects correction for wide-field-of-view optical scanners (BREFCOR). 
IEEE Transactions on Geoscience and Remote Sensing, 53(4), 1855-1864.

Wanner, W., Li, X., & Strahler, A. H. (1995). 
On the derivation of kernels for kernel-driven models of bidirectional reflectance.
Journal of Geophysical Research: Atmospheres, 100(D10), 21077-21089.
"""



def generate_geom_kernel(solar_az,solar_zn,sensor_az,sensor_zn,li):
    '''Calculate the Li geometric scattering kernel. 
    
    Parameters
    ----------
    solar_az:   float or np.array
                Solar zenith angle in radians
    solar_zn:   float or np.array 
                Solar zenith angle in radians
    sensor_az:  np.array
                Sensor view azimuth angle in radians
    sensor_zn:  np.array
                Sensor view zenith angle in radians          
    li :        str 
                Geometric scattering kernel type [dense,sparse]
    
    Returns
    -------
    Volume and geomtric scattering kernels as m x n numpy array
    '''
    
    relative_az = sensor_az - solar_az 
    
    #Li kernels
    ############
    # Constants used from Colgan  et al. RS 2012 
    
    # Eq. 37,52. Wanner et al. JGRA 1995
    solar_zn_ = np.arctan(10* np.tan(solar_zn))
    sensor_zn_ = np.arctan(10* np.tan(sensor_zn))
    # Eq 50. Wanner et al. JGRA 1995
    D = np.sqrt((np.tan(solar_zn_)**2) + (np.tan(sensor_zn_)**2) - 2*np.tan(solar_zn_)*np.tan(sensor_zn_)*np.cos(relative_az))    
    # Eq 49. Wanner et al. JGRA 1995
    t_num = 2. * np.sqrt(D**2 + (np.tan(solar_zn_)*np.tan(sensor_zn_)*np.sin(relative_az))**2) 
    t_denom = (1/np.cos(solar_zn_))  + (1/np.cos(sensor_zn_))
    t = np.arccos(np.clip(t_num/t_denom,-1,1))
    # Eq 33,48. Wanner et al. JGRA 1995
    O = (1/pi) * (t - np.sin(t)*np.cos(t)) * t_denom
    # Eq 51. Wanner et al. JGRA 1995
    cosPhase_ =  np.cos(solar_zn_)*np.cos(sensor_zn_) + np.sin(solar_zn_)*np.sin(sensor_zn_)*np.cos(relative_az)

    if li == 'sparse':
        # Eq 32. Wanner et al. JGRA 1995
        k_geom = O - (1/np.cos(solar_zn_)) - (1/np.cos(sensor_zn_)) + .5*(1+ cosPhase_) * (1/np.cos(sensor_zn_))
    elif li == 'dense':
        # Eq 47. Wanner et al. JGRA 1995
        k_geom = (((1+cosPhase_) * (1/np.cos(sensor_zn_)))/ (t_denom - O)) - 2
    
    return k_geom


def generate_volume_kernel(solar_az,solar_zn,sensor_az,sensor_zn, ross):
    '''Calculate the Ross vlumetric scattering kernel. 
    
    Parameters
    ----------
    solar_az:   float or np.array
                Solar zenith angle in radians
    solar_zn:   float or np.array 
                Solar zenith angle in radians
    sensor_az:  np.array
                Sensor view azimuth angle in radians
    sensor_zn:  np.array
                Sensor view zenith angle in radians          
    ross:       str 
                Volume scattering kernel type [thick,thin]
    
    Returns
    -------
    Volume scattering kernel as m x n numpy array
    '''
      
    relative_az = sensor_az - solar_az 
    
    #Ross kernels 
    ############

    # Eq 2. Schlapfer et al. IEEE-TGARS 2015
    phase = np.arccos(np.cos(solar_zn)*np.cos(sensor_zn) + np.sin(solar_zn)*np.sin(sensor_zn)*  np.cos(relative_az))
    
    if ross == 'thick':
        # Eq 13. Wanner et al. JGRA 1995
        k_vol = ((pi/2 - phase)*np.cos(phase) + np.sin(phase))/ (np.cos(sensor_zn)*np.cos(solar_zn)) - pi/4
    elif ross == 'thin':
        # Eq 13. Wanner et al. JGRA 1995
        k_vol = ((pi/2 - phase)*np.cos(phase) + np.sin(phase))/ (np.cos(sensor_zn)*np.cos(solar_zn)) - pi/2


    return k_vol


