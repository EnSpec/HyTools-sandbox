from .kernels import *
from ..file_io import *
import pandas as pd

"""
This module contains functions to apply a modified version of the the BRDF correction
described in the following papers:

Colgan, M. S., Baldeck, C. A., Feret, J. B., & Asner, G. P. (2012). 
Mapping savanna tree species at ecosystem scales using support vector machine classification 
and BRDF correction on airborne hyperspectral and LiDAR data. 
Remote Sensing, 4(11), 3462-3480.        

Collings, S., Caccetta, P., Campbell, N., & Wu, X. (2010). 
Techniques for BRDF correction of hyperspectral mosaics. 
IEEE Transactions on Geoscience and Remote Sensing, 48(10), 3733-3746.
    
Schlapfer, D., Richter, R., & Feingersh, T. (2015). 
Operational BRDF effects correction for wide-field-of-view optical scanners (BREFCOR). 
IEEE Transactions on Geoscience and Remote Sensing, 53(4), 1855-1864.

Wanner, W., Li, X., & Strahler, A. H. (1995). 
On the derivation of kernels for kernel-driven models of bidirectional reflectance. 
Journal of Geophysical Research: Atmospheres, 100(D10), 21077-21089.

Weyermann, J., Kneubuhler, M., Schlapfer, D., & Schaepman, M. E. (2015). 
Minimizing Reflectance Anisotropy Effects in Airborne Spectroscopy Data Using Ross-Li Model Inversion 
With Continuous Field Land Cover Stratification. 
IEEE Transactions on Geoscience and Remote Sensing, 53(11), 5814-5823.

BRDF correction consists of the following steps:
    
    1. OPTIONAL: Stratified random sampling of the image(s) based on predefined scattering classes
    2. Regression modeling per scattering class, per wavelength
            reflectance = fIso + fVol*kVol +  fGeo*kGeo
            (eq 2. Weyermann et al. IEEE-TGARS 2015)
    3. Adjust reflectance using a multiplicative correction per class, per wavelength. 
            (eq 4. Weyermann et al. IEEE-TGARS 2015)
"""



def generate_brdf_coeff_band(band,mask,k_vol,k_geom):
    '''Return the BRDF coefficients for the input band.
    
    Parameters
    ----------
    band:       m x n np.array
                Image band
    mask:       m x n np.array
                Binary image mask
    k_vol:      m x n np.array
                Volume scattering kernel image
    k_geom:     m x n np.array
                Geometric scattering kernel image
    Returns
    -------
    brdf_coeff: list
                BRDF coefficients
    '''
    
    # Mask kernels
    k_vol = k_vol[mask]
    k_geom = k_geom[mask]
    # Reshape for regression
    k_vol = np.expand_dims(k_vol,axis=1)
    k_geom = np.expand_dims(k_geom,axis=1)
    #X = np.concatenate([k_vol,k_geom,np.ones(k_geom.shape)],axis=1)
    X = np.concatenate([k_vol,k_geom,np.ones(k_geom.shape)],axis=1)
    # Mask input band
    y = band[mask]
    # Calculate BRDF coefficients
    brdf_coeff = np.linalg.lstsq(X, y)[0].flatten()
    
    return brdf_coeff
    

def generate_brdf_coeffs_img(hyObj,ross,li):
    '''Generate BRDF coefficients for a single image.
    
    Parameters
    ----------
    hyObj:      hyTools file object   
    ross:       str 
                Volume scattering kernel type [dense,sparse]
    li :        str 
                Geometric scattering kernel type [dense,sparse]    
    
    Returns
    -------
    brdfDF:     Wavelengths x 3 pandas dataframe
                BRDF coefficients
    '''
    
    # Generate scattering kernels
    k_vol = generate_volume_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn, ross = ross)
    k_geom = generate_geom_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn,li = li)
    
    brdf_coeffs = [] 
    iterator = hyObj.iterate(by = 'band')

    while not iterator.complete:        
        band = iterator.read_next()    
        brdf_coeffs.append(generate_brdf_coeffs_band(band,hyObj.mask,k_vol,k_geom))
    # Store coeffs in a pandas dataframe
    brdf_df =  pd.DataFrame(brdf_coeffs,index = hyObj.wavelengths,columns=['k_vol','k_geom','k_iso'])

    del k_vol, k_geom
    return brdf_df
    

    
def brdf_correct_img(hyObj,output_name,ross,li):
    """Apply BRDF coeffients to an image.

    Parameters
    ----------
    hyObj:     hyTools data object
                Data spectrum.
    output_name: str
                Path name for BRDF corrected file.
    brdf_coeffs: 
                Path name for BRDF corrected file.      
    li :        str 
                Geometric scattering kernel type [dense,sparse]
    ross:       str 
                Volume scattering kernel type [thick,thin]
   
    Returns
    -------
    None
        
    """
    # Calculate BRDF coefficents
    brdf_df = generate_brdf_coeffs_img(hyObj,ross,li)
    
    # Generate scattering kernels
    k_vol = generate_volume_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn, ross = ross)
    k_geom = generate_geom_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn,li = li)
    
    # Generate scattering kernels at Nadir
    k_vol_nadir = generate_volume_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,0, ross = ross)
    k_geom_nadir = generate_geom_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,0,li = li)
    
    
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
        
        # Get scattering kernel for chunks
        k_vol_nadir_chunk = k_vol_nadir[line_start:line_end,col_start:col_end]
        k_geom_nadir_chunk = k_geom_nadir[line_start:line_end,col_start:col_end]
        k_vol_chunk = k_vol[line_start:line_end,col_start:col_end]
        k_geom_chunk = k_geom[line_start:line_end,col_start:col_end]

        # Apply brdf correction 
        # eq 5. Weyermann et al. IEEE-TGARS 2015)
        brdf = np.einsum('i,jk-> jki', brdf_df.k_vol,k_vol_chunk) + np.einsum('i,jk-> jki', brdf_df.k_geom,k_geom_chunk)  + brdf_df.k_iso.values
        brdf_nadir = np.einsum('i,jk-> jki', brdf_df.k_vol,k_vol_nadir_chunk) + np.einsum('i,jk-> jki', brdf_df.k_geom,k_geom_nadir_chunk)  + brdf_df.k_iso.values
        correctionFactor = brdf_nadir/brdf
        brdf_chunk = chunk* correctionFactor
        
        # Reassign no_data values            
        brdf_chunk[chunk == hyObj.no_data] = hyObj.no_data

        writer.write_chunk(brdf_chunk,iterator.current_line,iterator.current_column)

    writer.close()
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    