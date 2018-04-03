import numpy as np
from ..file_io import *


def apply_plsr(hyObj,plsr_coeffs):
    """Apply Partial Least Squares Regression (PLSR) Coefficients to image.

    This function applies a set of PLSR coefficients to an image and return
    the mean and standard deviation of the estimates.


    Parameters
    ----------
    hyObj : hyTools data object
        Data spectrum.
    plsr_coeffs :  pathname of csv containing coefficients in the following format:
        
        +---------+-----------+--------------+----------+--------------+
        |         | intercept | wavelength_1 | ........ | wavelength_n |
        +---------+-----------+--------------+----------+--------------+
        | 1       |           |              |          |              |
        | ....... |           |              |          |              |
        | n       |           |              |          |              |
        +---------+-----------+--------------+----------+--------------+
                
   Where  wavelength_1 is the wavelength of the band in the same units as the image.
        
        
    Returns
    -------
    trait_pred : numpy array (lines x columns x 2)
        Array same XY shape as input image, 
            first band: mean trait prediction
            second band: standar deviation of the trait prediction
        
    """
    
    
    coeffs = np.loadtxt(plsr_coeffs,dtype= str, delimiter=',')
    
    #TODO: Check that coeffiecient wavelengths and image wavelengths match
    
    iterator = hyObj.iterate(by = 'chunk')

    # Empty array to hold trait estimates
    trait_pred = np.zeros((hyObj.lines,hyObj.columns,2)

    while not iterator.complete:
        chunk = iterator.read_next()            
        
        
   
    
    return trait_pred

