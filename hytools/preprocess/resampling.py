import numpy as np
from ..file_io import *

def __gaussian(x,mu,fwhm):
    """Return a gaussian distribution.
    
    Parameters
    ----------
    x : Numpy array of values along which to generate gaussian.    
    mu : Mean of the gaussian function.
    fwhm : Full width half maximum.

    Returns
    -------
    Numpy array of gaussian along input range.
    """
    
    c = fwhm/(2* np.sqrt(2*np.log(2)))
    return np.exp(-1*((x-mu)**2/(2*c**2)))

def __resample_coeff_single(srcWaves, dstWaves,dstFWHMs):
    """ Return a set of coeffiencients for spectrum resampling
    
    Given a set of source wavelengths, destination wavelengths and FWHMs this
    function caculates the relative contribution or each input wavelength
    to the output wavelength. It assumes that output 
    response functions follow a gaussian distribution.
    
    Parameters
    ----------
    srcWaves : List of source wavelength centers.    
    dstWaves : List of destination wavelength centers.    
    dstFWHMs : List of destination full width half maxes.

    Returns
    -------
    m x n matrix of coeffiecients, where m is the number of source wavelengths
    and n is the number of destination wavelengths.    
    """

    # For each destination band calculate the relative contribution
    # of each wavelength to the band response at source resolution
    dstMatrix = []
    #oneNM = np.arange(280,2600)
    for dstWave,dstFWHM in zip(dstWaves,dstFWHMs):
        a = __gaussian(srcWaves -.5,dstWave,dstFWHM)
        b = __gaussian(srcWaves +.5,dstWave,dstFWHM) 
        areas = (a +b)/2
        dstMatrix.append(np.divide(areas,np.sum(areas)))
    dstMatrix = np.array(dstMatrix) 
    
    return dstMatrix.T

def __matrix_inverse(srcWaves,srcFWHMs,dstWaves,dstFWHMs):
    """ Return a set of coeffiencients for spectrum resampling
    
    Given a set of source and destination wavelengths and FWHMs this
    function downscales each input wavelength
    to 1nm, then upscales the 1nm wavelength to the output wavelength. 
    It assumes that both input and output 
    response functions follow a gaussian distribution.
    
    While downscaling, it is an underdetermined problem with multiple possible solutions, 
    the minimum L2 Norm of the spectral is used to regularize the problem solving. 
    A Pseudoinverse (Moore–Penrose inverse) is used to estimate the best-fit downscaling result.
    
    Reference for numpy.linalg.pinv :
    G. Strang, Linear Algebra and Its Applications, 2nd Ed., Orlando, FL, Academic Press, Inc., 1980, pp. 139-142.

    
    Parameters
    ----------
    srcWaves : List of source wavelength centers.    
    srcFWHMs : List of source full width half maxes.
    dstWaves : List of destination wavelength centers.    
    dstFWHMs : List of destination full width half maxes.
    spacing : resolution at which to model the spectral resposnse functions

    Returns
    -------
    m x n matrix of coeffiecients, where m is the number of source wavelengths
    and n is the number of destination wavelengths.    
    """
    
    dstMatrix = []
    oneNM = np.arange(280,2600)

    for dstWave,dstFWHM in zip(dstWaves,dstFWHMs):
        #print dstWave,dstFWHM
        a = __gaussian(oneNM -.5,dstWave,dstFWHM)
        b = __gaussian(oneNM +.5,dstWave,dstFWHM) 
        areas = (a +b)/2
        dstMatrix.append(np.divide(areas,np.sum(areas)))

    dstMatrix = np.array(dstMatrix)
    
    srcMatrix = []

    for srcWave,srcFWHM in zip(srcWaves,srcFWHMs):

        a = __gaussian(oneNM -.5,srcWave,srcFWHM)
        b = __gaussian(oneNM +.5,srcWave,srcFWHM) 
        areas = (a +b)/2
        srcMatrix.append(np.divide(areas,np.sum(areas)))        

    srcMatrix = np.array(srcMatrix)    

    pseudo = np.linalg.pinv(srcMatrix)
    #pseudo = np.dot(srcMatrix.T,np.linalg.inv(np.dot(srcMatrix,srcMatrix.T)))

    coef = np.dot(dstMatrix,pseudo)
    
    return coef.T

def __resample_coeff(srcWaves,srcFWHMs,dstWaves,dstFWHMs, spacing = 1):
    """ Return a set of coeffiencients for spectrum resampling
    
    Given a set of source and destination wavelengths and FWHMs this
    function caculates the relative contribution or each input wavelength
    to the output wavelength. It assumes that both input and output 
    response functions follow a gaussian distribution.
    
    Parameters
    ----------
    srcWaves : List of source wavelength centers.    
    srcFWHMs : List of source full width half maxes.
    dstWaves : List of destination wavelength centers.    
    dstFWHMs : List of destination full width half maxes.
    spacing : resolution at which to model the spectral resposnse functions

    Returns
    -------
    m x n matrix of coeffiecients, where m is the number of source wavelengths
    and n is the number of destination wavelengths.    
    """

    # For each destination band calculate the relative contribution
    # of each wavelength to the band response at 1nm resolution
    dstMatrix = []
    oneNM = np.arange(280,2600)
    for dstWave,dstFWHM in zip(dstWaves,dstFWHMs):
        a = __gaussian(oneNM -.5,dstWave,dstFWHM)
        b = __gaussian(oneNM +.5,dstWave,dstFWHM) 
        areas = (a +b)/2
        dstMatrix.append(np.divide(areas,np.sum(areas)))
    dstMatrix = np.array(dstMatrix)

    # For each source wavelength generate the gaussion response
    # function at 1nm resolution
    srcMatrix = []
    for srcWave,srcFWHM in zip(srcWaves,srcFWHMs):
        srcMatrix.append(__gaussian(oneNM ,srcWave,srcFWHM))
    srcMatrix = np.array(srcMatrix)
   
    # Calculate the relative contribution of each source response function
    ratio =  srcMatrix/srcMatrix.sum(axis=0)
    ratio[np.isnan(ratio)] = 0
    ratio2 = np.einsum('ab,cb->acb',ratio,dstMatrix)
    
    # Calculate the relative contribution of each input wavelength
    # to each destination wavelength
    coeffs = np.trapz(ratio2)

    return coeffs

    

def  __est_fwhm(hyObj, dstWaves, dstFWHMs):
    """
    Acquire source wavelength information from source dataset, and estimate the list of destination full width half maxes if they are not specify by the user.
    
    Parameters
    ----------
    hyObj : hyTools data object
    
    dstWaves : List of destination wavelength centers.
    
    dstFWHMs : List of destination full width half maxes.


    Returns
    -------
    srcWaves : List of source wavelength centers.    
    srcFWHMs : List of source full width half maxes.
    dstFWHMs : List of destination full width half maxes.    
    """
    
    srcWaves = hyObj.wavelengths    
    srcFWHMs = hyObj.fwhm
    
    if dstFWHMs is None:
      gap = 0.5 * (dstWaves[1:] - dstWaves[:-2])
      #print(gap,gap[1:], gap[:-1],gap[-1])
      dstFWHMs_middle = gap[1:] + gap[:-1]
      #print(dstFWHMs_middle)
      dstFWHMs = np.append(np.append(gap[0]*2, dstFWHMs_middle), gap[-1]*2)
      
    return (srcWaves, srcFWHMs, dstFWHMs)
    
    
def __est_transform_matrix(srcWaves, dstWaves, srcFWHMs, dstFWHMs, method_code):

    if method_code==0:
        coeffs = __resample_coeff_single(srcWaves,dstWaves,dstFWHMs)
    elif method_code==1:
        coeffs = __resample_coeff(srcWaves,srcFWHMs,dstWaves,dstFWHMs, spacing = 1)
    else:
        coeffs = __matrix_inverse(srcWaves,srcFWHMs,dstWaves,dstFWHMs)
           
    return coeffs
    

def resample(hyObj, output_name, dstWaves, method="single_FWHM",  dstFWHMs = None):
    """.

    Parameters
    ----------
    hyObj : hyTools data object

    output_name: str
        Path name for resampled file.

    dstWaves
        List of destination wavelength centers.
        
    dstFWHMs
        List of destination full width half maxes. Default is None. 
        
    method
        single_FWHM (default)
            interpolation without srcFWHMs (List of source full width half maxes)
        two_FWHM
            interpolation with both srcFWHMs and dstFWHMs
        two_FWHM_minnorm
            Minimum-norm least squares problem (underdetermined case while resampling to 1nm) solved by pseudoinverse, with both srcFWHMs and dstFWHMs
            
        All methods require the list of source wavelength centers (srcWaves), which should be stored in hyObj.
        
    Returns
    -------
    None
            
    """

    # Dictionary of all methods
    methodDict = {"single_FWHM": 0,
                            "two_FWHM":1, 
                            "two_FWHM_minnorm": 2
                           }
                 
    srcWaves, srcFWHMs, dstFWHMs = __est_fwhm(hyObj, dstWaves, dstFWHMs)
    
    band_mask = [x==1 for x in hyObj.bad_bands]
    
    coeffs = __est_transform_matrix(srcWaves, dstWaves, srcFWHMs, dstFWHMs, methodDict[method])

    # update destination bad band list
    new_badband = np.dot(band_mask,coeffs)
    new_badband = (new_badband > 0.9).astype(np.uint8)

    # update header dictionary of the destination image
    new_headerDict = hyObj.header_dict
    new_headerDict["bands"] = len(dstWaves)
    new_headerDict["wavelength"] = '{'+','.join(['%g' % x for x in dstWaves])+'}'
    new_headerDict["fwhm"]  = '{'+','.join(['%g' % x for x in dstFWHMs])+'}'
    new_headerDict["bbl"] = '{'+','.join([str(x) for x in new_badband])+'}'
    new_bandnames = ['Band_'+str(x) for x in dstWaves]
    new_headerDict["band names"] = '{'+ ','.join(new_bandnames) +'}'
    
    
    if  hyObj.file_type == "ENVI":
        writer = writeENVI(output_name,new_headerDict)
    elif hyObj.file_type == "HDF":
        writer = None
    else:
        print("ERROR: File format not recognized.")       
        
    iterator = hyObj.iterate(by = 'chunk')

    while not iterator.complete:
        chunk = iterator.read_next()            
        #resampled_chunk = np.dot(coeffs,chunk[:,:,band_mask])
        resampled_chunk = np.dot(chunk[:,:,:], coeffs)         
        #resampled_chunk[ resampled_chunk <1000 ] = hyObj.no_data 
        writer.write_chunk(resampled_chunk,iterator.current_line,iterator.current_column)


    writer.close()    
    
    return 1
        
        
        
        
        
        