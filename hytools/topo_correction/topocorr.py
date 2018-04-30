
from ..file_io import *
import pandas as pd, sys

import numpy as np

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

def cal_cosine_i(solar_zn, solar_az, surfacenormal_az , surfacenormal_zn):
  """
   All angles are in radians
  """
  relAz = surfacenormal_az - solar_az
  cosine_i = np.cos(solar_zn)*np.cos(surfacenormal_zn) + np.sin(solar_zn)*np.sin(surfacenormal_zn)*  np.cos(relAz)

  return cosine_i


def generate_topo_coeffs_img(hyObj, solar_az,solar_zn, surfacenormal_az , surfacenormal_zn, cos_i = None):
    """
    Generate TOPO Correction coefficients for a single image.
    
    
    Parameters
    ----------
    hyObj:      hyTools file object
    solar_az:   float or np.array
                Solar zenith angle in radians
    solar_zn:   float or np.array 
                Solar zenith angle in radians
    surfacenormal_az:  np.array
                Terrain surface normal azimuth angle in radians (Aspect of terrain)
    surfacenormal_zn:  np.array
                Terrain surface normal zenith angle in radians (Slope of terrain)   

    cos_i:  np.array
                The cosine of the incidence angle (i ), defined as the angle between the normal to the pixel surface and the solar zenith direction

    Returns
    -------
    topoDF:     Wavelengths x 3 pandas dataframe
                TOPO coefficients
    """
    
    
    # Generate the cosine i
    # the cosine of the incidence angle (i ), defined as the angle between the normal to the pixel surface and the solar zenith direction;
    if cos_i is None:
      print("calculating incidence angle...")
      cos_i = cal_cosine_i(solar_zn, solar_az, surfacenormal_az , surfacenormal_zn)

    # Mask kernels
    cos_i = cos_i[hyObj.mask]


    # Reshape for regression
    cos_i = np.expand_dims(cos_i,axis=1)

    X = np.concatenate([cos_i,np.ones(cos_i.shape)],axis=1)

    iterator = hyObj.iterate(by = 'band')
    topo_c_coeffs = []
    
    while not iterator.complete:        
        band = iterator.read_next()    
        
        y = band[hyObj.mask]
        
        # Eq 7. Soenen et al., IEEE TGARS 2005
        slope, intercept = np.linalg.lstsq(X, y)[0].flatten()
        
        # Eq 8. Soenen et al., IEEE TGARS 2005
        C= intercept/slope

        
        # set a large number if slope is zero
        if not np.isfinite(C):
          C = 100000.0
          
        topo_c_coeffs.append([C])
        
    #print( zip(hyObj.wavelengths,topo_c_coeffs))
    #print(hyObj.wavelengths)

    # Store coeffs in a pandas dataframe
    topoDF =  pd.DataFrame.from_dict(dict(zip(hyObj.wavelengths,topo_c_coeffs))).T
    topoDF.columns = ['c']

    del cos_i

    return topoDF


def apply_topo_coeffs(hyObj,output_name, topo_coeffs,solar_az,solar_zn,surfacenormal_az , surfacenormal_zn, cos_i = None):
    """Apply TOPO Correction coeffients to an image.

    Parameters
    ----------
    hyObj:     hyTools data object
                Data spectrum.
    output_name: str
                Path name for TOPO corrected file.
    topo_coeffs: 
                Path name for TOPO corrected file.
    solar_az:   float or np.array
                Solar zenith angle in radians
    solar_zn:   float or np.array 
                Solar zenith angle in radians
    surfacenormal_az:  np.array
                Terrain surface normal azimuth angle in radians (Aspect of terrain)
    surfacenormal_zn:  np.array
                Terrain surface normal zenith angle in radians (Slope of terrain)     

    cos_i:  np.array
                The cosine of the incidence angle (i ), defined as the angle between the normal to the pixel surface and the solar zenith direction
   
    Returns
    -------
    None
        
    """
    # Load TOPO coefficents to pandas dataframe
    topo_df =pd.read_csv(topo_coeffs,index_col = 0)
    
    # Generate the cosine i
    # the cosine of the incidence angle (i ), defined as the angle between the normal to the pixel surface and the solar zenith direction;
    if cos_i is None:
      print("calculating incidence angle...")
      cos_i = cal_cosine_i(solar_zn, solar_az, surfacenormal_az , surfacenormal_zn)
    
    # Eq 11. Soenen et al., IEEE TGARS 2005
    # cos(alpha)* cos(theta)
    # alpha -- slope (surfacenormal_zn), theta -- solar zenith angle (solar_zn)
    c1 = np.cos(solar_zn) * np.cos(surfacenormal_zn)
    
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
        #print(chunk.shape)
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
        
        
    
       