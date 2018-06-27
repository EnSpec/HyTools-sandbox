import argparse,gdal,copy
import numpy as np, os,pandas as pd
import glob
import hytools as ht
from hytools.brdf import *
from hytools.topo_correction import *
from hytools.helpers import *
from hytools.preprocess.resampling import *
from hytools.preprocess.vector_norm import *
from hytools.file_io import array_to_geotiff
home = os.path.expanduser("~")


def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

def image_to_traits(image,observables,mask_type,mask_threshold,topo_correct,brdf_correct,vnorm,rgbim,traits):
    '''
    Perform in memory trait estimation.
    '''
    
    if image.endswith(".h5"):
        hyObj = ht.openHDF(image,load_obs = True)
        hyObj.create_bad_bands([[300,400],[1330,1430],[1800,1960],[2450,2600]])
    else:
        hyObj = ht.openENVI(image)
        hyObj.load_obs(observables)

    hyObj.load_data()
    
    # Generate mask
    if mask_type == "ndvi":
        ir = hyObj.get_wave(850)
        red = hyObj.get_wave(665)
        ndvi = (ir-red)/(ir+red)
        mask = (ndvi > mask_threshold) & (ir != hyObj.no_data)
        hyObj.mask = mask 
        del ir,red,ndvi
    elif mask_type == "none":
        print("Warning no mask specified, results may be unreliable!")

    # Generate cosine i and c1 image for topographic correction
    if topo_correct == True:
        cos_i =  calc_cosine_i(hyObj.solar_zn, hyObj.solar_az, hyObj.azimuth , hyObj.slope)
        c1 = np.cos(hyObj.solar_zn) * np.cos( hyObj.slope)
        topo_coeffs= []
           
    # Gernerate scattering kernel images for brdf correction
    if brdf_correct == True:
        k_vol = generate_volume_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn, ross = ross)
        k_geom = generate_geom_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn,li = li)
        k_vol_nadir = generate_volume_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,0, ross = ross)
        k_geom_nadir = generate_geom_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,0,li = li)
        brdf_coeffs= []

    # Cycle through the bands and calculate the topographic and BRDF correction coefficients
    print("Calculating image correction coefficients.....")
    iterator = hyObj.iterate(by = 'band')
    while not iterator.complete:   
        band = iterator.read_next() 
        progbar(iterator.current_band+1, len(hyObj.wavelengths), 100)
        #Skip bad bands
        if hyObj.bad_bands[iterator.current_band]:
            # Generate topo correction coefficients
            if topo_correct:
                topo_coeff= generate_topo_coeff_band(band,hyObj.mask,cos_i)
                topo_coeffs.append(topo_coeff)
                # Apply topo correction to current band
                correctionFactor = (c1 * topo_coeff)/(cos_i * topo_coeff)
                band = band* correctionFactor
            # Gernerate BRDF correction coefficients
            if brdf_correct:
                brdf_coeffs.append(generate_brdf_coeff_band(band,hyObj.mask,k_vol,k_geom))
    print()
    
    topo_df =  pd.DataFrame(topo_coeffs, index=  hyObj.wavelengths[hyObj.bad_bands], columns = ['c'])
    brdf_df =  pd.DataFrame(brdf_coeffs,index = hyObj.wavelengths[hyObj.bad_bands],columns=['k_vol','k_geom','k_iso'])
    
    # Load first trait csv and compare wavelengths
    traitModel  = pd.read_csv(traits[0],index_col=0)
    traitModel.columns = traitModel.columns.map(column_retype)
    image_wavelengths = np.array([x for x in traitModel.columns if type(x) == float])
    model_wavelengths = np.array([x for x in  traitModel.dropna().columns if type(x) == float])

    # Check if all bands match in image and coeffs
    if np.sum([x in image_wavelengths for x in hyObj.wavelengths]) != len(image_wavelengths):
        print("Coefficient wavelengths and image wavelengths do not match, calculating resampling coefficients...")
        resample = True
        
        #Update bad band list for resampling data
        if "bad_bands" in traitModel.index:
            band_mask = [x==0 for x in traitModel.loc["bad_bands",image_wavelengths]]
        else: 
            band_mask = [True for x in image_wavelengths]

        # Check for FWHM in coeffiecient dataframe
        if "fwhm" in traitModel.index:
            fwhm = traitModel.loc['fwhm',image_wavelengths].values
            # Check for FWHM in object
            if type(hyObj.fwhm) == np.ndarray:
                # Calculate resampling coefficients while excluding bad bands
                resampling_coeffs = est_transform_matrix(hyObj.wavelengths[hyObj.bad_bands],image_wavelengths[band_mask],hyObj.fwhm[hyObj.bad_bands],fwhm[band_mask],1)
                # Create a new Hytools file object for the resampled data    
                hyObj_resamp = copy.copy(hyObj)
                hyObj_resamp.wavelengths = image_wavelengths[band_mask]
                hyObj_resamp.bad_bands = band_mask
    
    else:
        resample = False

    #Cycle through the chunks and apply topo, brdf, vnorm,resampling and trait estimation steps
    print("Calculating values for %s traits....." % len(traits))
    pixels_processed = 0
    iterator = hyObj.iterate(by = 'chunk',chunk_size = (100,100))
    while not iterator.complete:  
        chunk = iterator.read_next()  
        chunk_nodata_mask = chunk[:,:,1] == hyObj.no_data
        
        pixels_processed += chunk.shape[0]*chunk.shape[1]
        progbar(pixels_processed, hyObj.columns*hyObj.lines, 100)

        # Chunk Array indices
        line_start =iterator.current_line 
        line_end = iterator.current_line + chunk.shape[0]
        col_start = iterator.current_column
        col_end = iterator.current_column + chunk.shape[1]
        
        # Apply TOPO correction 
        if topo_correct:
            cos_i_chunk = cos_i[line_start:line_end,col_start:col_end]
            c1_chunk = c1[line_start:line_end,col_start:col_end]
            correctionFactor = (c1_chunk[:,:,np.newaxis]+topo_df.c.values)/(cos_i_chunk[:,:,np.newaxis] + topo_df.c.values)
            chunk = chunk[:,:,hyObj.bad_bands]* correctionFactor
        
        # Apply BRDF correction 
        if brdf_correct:
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
            chunk= chunk* correctionFactor
        
        #Reassign no data values
        chunk[chunk_nodata_mask,:] = 0
        
        
        # Resample data
        if resample == True:            
            chunk = np.dot(chunk[:,:,:], resampling_coeffs) 
        
        # Export RGBIM image
        if rgbim == True:
            dstFile = image + '_rgbim.tif'
            if line_start + col_start == 0:
                driver = gdal.GetDriverByName("GTIFF")
                tiff = driver.Create(dstFile,hyObj.columns,hyObj.lines,5,gdal.GDT_Float32)
                tiff.SetGeoTransform(hyObj.transform)
                tiff.SetProjection(hyObj.projection)
                for band in range(1,6):
                    tiff.GetRasterBand(band).SetNoDataValue(0)
                tiff.GetRasterBand(5).WriteArray(mask)

                del tiff,driver
            # Write rgbi chunk
            rgbi_geotiff = gdal.Open(dstFile, gdal.GA_Update)
            for i,wave in enumerate([480,560,660,850],start=1):
                if resample == True:   
                    band = hyObj_resamp.wave_to_band(wave)
                else:
                    band = hyObj.wave_to_band(wave)
                rgbi_geotiff.GetRasterBand(i).WriteArray(chunk[:,:,band], col_start, line_start)
            rgbi_geotiff = None
        
        # Vector normalize data
        if vnorm == True:            
            chunk = vector_normalize_chunk(chunk,[True for x in range(sum(hyObj_resamp.bad_bands))],vnorm_scaler)
        
        # Cycle through trait models 
        for i,trait in enumerate(traits):
            dstFile =image +"_" +os.path.splitext(os.path.basename(trait))[0] +".tif"
            
            # Trait estimation preparation
            if line_start + col_start == 0:
                # Load coefficients to memory
                traitModel  = pd.read_csv(trait,index_col=0)
                traitModel.columns = traitModel.columns.map(column_retype)
                traitModel.index = traitModel.index.map(column_retype)
                model_wavelengths = [x for x in  traitModel.dropna(axis=1).columns if type(x) == float]
                model_iterations = [x for x in  traitModel.index if type(x) == float]
                intercept = traitModel.loc[model_iterations,"intercept"].values
                coeffs = traitModel.loc[model_iterations,model_wavelengths].values
                trait_band_mask = [x in model_wavelengths for x in hyObj_resamp.wavelengths]

                #Initialize trait dictionary
                if i == 0:
                    trait_dict = {}
                trait_dict[i] = [coeffs,intercept,trait_band_mask]
        
                # Create geotiff 
                driver = gdal.GetDriverByName("GTIFF")
                tiff = driver.Create(dstFile,hyObj.columns,hyObj.lines,2,gdal.GDT_Float32)
                tiff.SetGeoTransform(hyObj.transform)
                tiff.SetProjection(hyObj.projection)
                tiff.GetRasterBand(1).SetNoDataValue(0)
                tiff.GetRasterBand(2).SetNoDataValue(0)
                del tiff,driver
            
            # Apply trait model
            coeffs,intercept,trait_band_mask = trait_dict[i]
            trait_mean,trait_std = apply_plsr_chunk(chunk[:,:,trait_band_mask],coeffs,intercept)
            
            # Change no data pixel values
            trait_mean[chunk_nodata_mask] = 0
            trait_std[chunk_nodata_mask] = 0

            # Write trait estimate to file
            trait_geotiff = gdal.Open(dstFile, gdal.GA_Update)
            trait_geotiff.GetRasterBand(1).WriteArray(trait_mean, col_start, line_start)
            trait_geotiff.GetRasterBand(2).WriteArray(trait_std, col_start, line_start)
            trait_geotiff = None



image = "/data/tmp/NEON_D05_CHEQ_DP1_20170911_183324_reflectance.h5"
observables = "/data/aviris/ang20150901t155015_obs_ort"
mask_type = "ndvi"
mask_threshold = .5
topo_correct = True
brdf_correct = True
traits = glob.glob("%s/Dropbox/projects/hyTools/PLSR_Hyspiri_test/*.csv" % home)
vnorm= True
vnorm_scaler = 1
rgbim = True
ross = 'thick'
li = 'dense'



image_to_traits(image,observables,mask_type,mask_threshold,topo_correct,brdf_correct,vnorm,rgbim,traits):
#
    
    
parser = argparse.ArgumentParser()

#parser.add_argument("--in", help="Input image pathname", action='store_true')
#parser.add_argument("--obs", help="Input observables pathname", action='store_true')
#parser.add_argument("--brdf", help="Perform BRDF correction",action='store_true')
#parser.add_argument("--topo", help="Perform topographic correction", action='store_true')
#parser.add_argument("--mask", help="Image mask type to use", action='store_true')
#parser.add_argument("--mask_threshold", help="Mask threshold value", action='store_true')
#parser.add_argument("--vnorm", help="Vector normalize image", action='store_true')
#parser.add_argument("--vnorm_scaler", help="Scaling value for vecotr normalization", action='store_true')
#parser.add_argument("--coeffs", help="Trait coefficients directory", action='store_true')
#
#






