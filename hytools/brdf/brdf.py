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
            (eq 5. Weyermann et al. IEEE-TGARS 2015)
"""

def generate_brdf_coeffs_img(hyObj,solar_az,solar_zn,sensor_az,sensor_zn,ross,li):
    '''Generate BRDF coefficients for a single image.
    
    Parameters
    ----------
    hyObj:      hyTools file object
    solar_az:   float or np.array
                Solar zenith angle in radians
    solar_zn:   float or np.array 
                Solar zenith angle in radians
    sensor_az:  np.array
                Sensor view azimuth angle in radians
    sensor_zn:  np.array
                Sensor view zenith angle in radians          
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
    k_vol = generate_volume_kernel(solar_az,solar_zn,sensor_az,sensor_zn, ross = ross)
    k_geom = generate_geom_kernel(solar_az,solar_zn,sensor_az,sensor_zn,li = li)
    
    # Mask kernels
    k_vol = k_vol[hyObj.mask]
    k_geom = k_geom[hyObj.mask]

    # Reshape for regression
    k_vol = np.expand_dims(k_vol,axis=1)
    k_geom = np.expand_dims(k_geom,axis=1)
    #X = np.concatenate([k_vol,k_geom,np.ones(k_geom.shape)],axis=1)
    X = np.concatenate([k_vol,np.ones(k_geom.shape)],axis=1)

    iterator = hyObj.iterate(by = 'band')
    brdf_coeffs = []
    
    while not iterator.complete:        
        band = iterator.read_next()    
        
        y = band[hyObj.mask]
        brdf_coeffs.append(np.linalg.lstsq(X, y)[0].flatten())
    
    # Store coeffs in a pandas dataframe
    brdfDF =  pd.DataFrame.from_dict(dict(zip(hyObj.wavelengths,brdf_coeffs))).T
    #brdfDF.columns = ['k_vol','k_geom','k_iso']
    brdfDF.columns = ['k_vol','k_iso']

    del k_vol, k_geom

    return brdfDF
    
def generate_brdf_coeffs_mosaic(pathname):

    return None


    
def apply_brdf_coeffs(hyObj,output_name,brdf_coeffs,solar_az,solar_zn,sensor_az,sensor_zn,ross,li):
    """Apply BRDF coeffients to an image.

    Parameters
    ----------
    hyObj:     hyTools data object
                Data spectrum.
    output_name: str
                Path name for BRDF corrected file.
    brdf_coeffs: 
                Path name for BRDF corrected file.
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
    ross:       str 
                Volume scattering kernel type [dense,sparse]
   
    Returns
    -------
    None
        
    """
    # Load BRDF coefficents to pandas dataframe
    brdf_df =pd.read_csv(brdf_coeffs,index_col = 0)
    
    # Generate scattering kernels
    k_vol = generate_volume_kernel(solar_az,solar_zn,sensor_az,sensor_zn, ross = ross)
    k_geom = generate_geom_kernel(solar_az,solar_zn,sensor_az,sensor_zn,li = li)
    
    # Generate scattering kernels at Nadir
    k_vol_nadir = generate_volume_kernel(solar_az,solar_zn,sensor_az,0, ross = ross)
    k_geom_nadir = generate_geom_kernel(solar_az,solar_zn,sensor_az,0,li = li)
    
    
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
        #brdf = np.einsum('i,jk-> jki', brdf_df.k_vol,k_vol_chunk) + np.einsum('i,jk-> jki', brdf_df.k_geom,k_geom_chunk)  + brdf_df.k_iso.values
        #brdf_nadir = np.einsum('i,jk-> jki', brdf_df.k_vol,k_vol_nadir_chunk) + np.einsum('i,jk-> jki', brdf_df.k_geom,k_geom_nadir_chunk)  + brdf_df.k_iso.values
        brdf = np.einsum('i,jk-> jki', brdf_df.k_vol,k_vol_chunk)  + brdf_df.k_iso.values
        brdf_nadir = np.einsum('i,jk-> jki', brdf_df.k_vol,k_vol_nadir_chunk)  + brdf_df.k_iso.values
        correctionFactor = brdf_nadir/brdf
        brdf_chunk = chunk* correctionFactor
        
        # Reassign no_data values            
        brdf_chunk[chunk == hyObj.no_data] = hyObj.no_data

        writer.write_chunk(brdf_chunk,iterator.current_line,iterator.current_column)

    writer.close()
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    