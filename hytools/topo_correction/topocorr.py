from ..file_io import *
import pandas as pd, sys
import numpy as np
from scipy.optimize import nnls  # , least_squares

#import matplotlib.pyplot as plt

"""
This module contains functions to apply a topographic correction (SCS+C)
described in the following papers:

Scott A. Soenen, Derek R. Peddle,  & Craig A. Coburn (2005). 
SCS+C: A Modified Sun-Canopy-Sensor Topographic Correction in Forested Terrain.
IEEE Transactions on Geoscience and Remote Sensing, 43(9), 2148-2159.

TOPO correction consists of the following steps:
    
    1. calcualte incidence angle if it is not provided
    2. estimate C-Correction value
    3. apply C-Correction value to the image
"""


def linearfun(coeffs, x_cos_i ,  y_refl):
  '''
  Linear function of cos_i-refl model
  '''
  return coeffs[1]+coeffs[0]*x_cos_i-y_refl


def calc_cosine_i(solar_zn, solar_az, aspect ,slope):
  """
   All angles are in radians
  """
  relAz = aspect - solar_az
  cosine_i = np.cos(solar_zn)*np.cos(slope) + np.sin(solar_zn)*np.sin(slope)*  np.cos(relAz)

  return cosine_i


def generate_topo_coeff_band(band,mask,cos_i, non_negative=False, robust_lsq=None):
    '''Return the topographic correction coefficients for the input band.
    
    Parameters
    ----------
    band:       m x n np.array
                Image band
    mask:       m x n np.array
                Binary image mask
    non_negative : flag for non-negative least squared regression, force positive slope and intercept
    robust_lsq : choice for robust least squares ('soft_l1' | 'huber' | 'cauchy' | 'arctan')
    Returns
    -------
    C : float
                Topographic correction coefficient
    slope : float 
                slope of the linear model
    intercept : float
                intercept of the linear model
    '''
    
    # Mask cosine i image
    cos_i = cos_i[mask]
    # Reshape for regression
    cos_i = np.expand_dims(cos_i,axis=1)

    X = np.concatenate([cos_i,np.ones(cos_i.shape)], axis=1)
    y = band[mask]
    
    if non_negative:
        slope, intercept = nnls(X, y)[0].flatten()
        

    else:
        if robust_lsq is None:
            # Eq 7. Soenen et al., IEEE TGARS 2005
            slope, intercept = np.linalg.lstsq(X, y)[0].flatten()
        else:
            # initialize
            loss_func = robust_lsq
            reg_coeffs = np.array([0.1,0.1])
            result_optimize = least_squares(linearfun, reg_coeffs, loss=loss_func, f_scale=0.1, args=(cos_i.flatten(), y))
            slope, intercept = result_optimize.x  
            
    # Eq 8. Soenen et al., IEEE TGARS 2005
    C= intercept/slope

    # Set a large number if slope is zero
    if not np.isfinite(C):
      C = 100000.0
      
    return C, slope, intercept

def generate_topo_coeffs_img(hyObj,cos_i = None):
    """
    Generate TOPO Correction coefficients for a single image.
    
    Parameters
    ----------
    hyObj:      hyTools file object
    cos_i:      np.array
                The cosine of the incidence angle (i ), defined as the angle between the normal to the pixel surface and the solar zenith direction

    Returns
    -------
    topoDF:     Wavelengths x 3 pandas dataframe
                TOPO coefficients
    """
    
    # Generate the cosine i
    # the cosine of the incidence angle (i ), 
    # defined as the angle between the normal to the pixel surface and the solar zenith direction;
    if cos_i is None:
      print("Calculating incidence angle...")
      cos_i =  calc_cosine_i(hyObj.solar_zn, hyObj.solar_az, hyObj.aspect , hyObj.slope)

    iterator = hyObj.iterate(by = 'band')
    topo_coeffs = []
    
    while not iterator.complete:        
        band = iterator.read_next()    
        topo_coeffs.append([generate_topo_coeff_band(band,hyObj.mask,cos_i)])
        
    # Store coeffs in a pandas dataframe
    topo_df =  pd.DataFrame(topo_coeffs, index=  hyObj.wavelengths, columns = ['c'])

    del cos_i
    return topo_df



def topo_correct_img(hyObj,output_name,cos_i = None):
    """Topographically correct an image.

    Parameters
    ----------
    hyObj:     hyTools data object
                Data spectrum.
    output_name: str
                Path name for TOPO corrected file.
    cos_i:  np.array
                The cosine of the incidence angle (i ), 
                defined as the angle between the normal to the pixel surface and the solar zenith direction
   
    Returns
    -------
    None
        
    """

    # Generate the cosine i
    # the cosine of the incidence angle (i ), defined as the angle between the normal to the pixel surface and the solar zenith direction;
    if cos_i is None:
      print("Calculating incidence angle...")
      cos_i =  calc_cosine_i(hyObj.solar_zn, hyObj.solar_az, hyObj.aspect , hyObj.slope)
    
    # Eq 11. Soenen et al., IEEE TGARS 2005
    # cos(alpha)* cos(theta)
    # alpha -- slope (slope), theta -- solar zenith angle (solar_zn)
    c1 = np.cos(hyObj.solar_zn) * np.cos(hyObj.slope)
    
    #Calcualate topographic correction coefficients for all bands
    topo_df = generate_topo_coeffs_img(hyObj,cos_i = None)
    
    # Create writer object
    if  hyObj.file_type == "ENVI":
        writer = writeENVI(output_name,hyObj.header_dict)
    elif hyObj.file_type == "HDF":
        writer = None
    else:
        print("ERROR: File format not recognized.")

    iterator = hyObj.iterate(by = 'chunk')

    while not iterator.complete:
        chunk = iterator.read_next()            
        # Chunk Array indices
        line_start =iterator.current_line 
        line_end = iterator.current_line + chunk.shape[0]
        col_start = iterator.current_column
        col_end = iterator.current_column + chunk.shape[1]
        
        # Get C-Correction factor for chunks
        cos_i_chunk = cos_i[line_start:line_end,col_start:col_end]
        c1_chunk = c1[line_start:line_end,col_start:col_end]

        # Apply TOPO correction 
        # Eq 11. Soenen et al., IEEE TGARS 2005
        correctionFactor = (c1_chunk[:,:,np.newaxis]+topo_df.c.values)/(cos_i_chunk[:,:,np.newaxis] + topo_df.c.values)
        topo_chunk = chunk* correctionFactor
        
        # Reassign no_data values            
        topo_chunk[chunk == hyObj.no_data] = hyObj.no_data

        writer.write_chunk(topo_chunk,iterator.current_line,iterator.current_column)

    writer.close()
        
