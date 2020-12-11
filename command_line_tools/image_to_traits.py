import argparse,gdal,copy,sys,warnings
import numpy as np, os, glob,json, sys
import scipy.signal as sig

# for NDVI-BRDF-coefficient interpolation
from scipy.interpolate import interp1d

# for NDVI-BRDF linear/polynomial fitting
from numpy.polynomial import polynomial as P


import matplotlib.pyplot as plt
import hytools as ht
from hytools.brdf import *
from hytools.topo_correction import *
from hytools.helpers import *
from hytools.preprocess.resampling import *
from hytools.preprocess.vector_norm import *
from hytools.file_io import *
home = os.path.expanduser("~")

warnings.filterwarnings("ignore")


###########################################
############Constant Section####################
###########################################


# No data value of input images
NO_DATA_VALUE = -9999 #-0.9999  # -9999

# NDVI range for data considered for BRDF correction. Pixels outside BIN range will not be corrected for BRDF effect 
NDVI_APPLIED_BIN_MIN_THRESHOLD = 0.05
NDVI_APPLIED_BIN_MAX_THRESHOLD = 1.0

# Thresholds for Topographic correction. Pixels beyond those ranges will not be corrected for topographic effect. 
# Minimum illumination value (cosine of the incident angle): 0.12 ( < 83.1 degrees)
# Minimum slope: 5 degrees
COSINE_I_MIN_THRESHOLD = 0.12
SLOPE_MIN_THRESHOLD = 0.087

# Chunk size of data reading in row direction (1st dimension)
CHUNK_EDGE_SIZE = 32

# Band number used for making no data mask, 0-based
BAND_NO_DATA = 10

# NDVI range for NDVI-based BRDF coefficients interpolation / regression. NDVI bins outside the range will not be used for BRDF coefficients interpolation or regression. 
BRDF_VEG_upper_bound = 0.85
BRDF_VEG_lower_bound = 0.25

# Wavelengths for NDVI calculation, unit: nanometers
BAND_IR_NM = 850
BAND_RED_NM = 665

# Bad band range. Bands within these ranges will be treated as bad bands, unit: nanometers
BAD_RANGE =   [[300,400],[1337,1430],[1800,1960],[2450,2600]]

# Bands in output GeoTIFF file, unit: nanometers
RGBIM_BAND = [480,560,660,850,976,1650,2217]

# Name field of correction /  smoothing factors. It apprears in the header file of CORR product.
NAME_FIELD_SMOOTH = 'correction factors'


###########################################

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

def check_wave_match(hyObj, dst_wave_list):
    
    wavelengths = hyObj.wavelengths[hyObj.bad_bands]
    fwhm_list = hyObj.fwhm[hyObj.bad_bands]
    match_index_list = []
    for single_wave, single_fwhm in dst_wave_list:

        if hyObj.wavelength_units == "micrometers" and single_wave > 3:
            single_wave /= 1000
        if hyObj.wavelength_units == "nanometers" and single_wave < 3:
            single_wave *= 1000    
    
        if single_wave in wavelengths:
            wave_band = np.argwhere(single_wave == wavelengths)[0][0]
            match_index_list = match_index_list + [wave_band]
        elif (single_wave  > wavelengths.max()) | (single_wave  < wavelengths.min()):
            return {'flag':False}
        else: 
            wave_band = np.argmin(np.abs(wavelengths - single_wave))    
            if abs(wavelengths[wave_band]-single_wave) > 0.5:
            # there is offset between source and destintion wavelength
              #print(wave_band, abs(wavelengths[wave_band]-single_wave),single_wave,wavelengths[wave_band])
              if abs(fwhm_list[wave_band]-single_fwhm) < 1.2:
              # center has offset, but fwhm matches
                return {'flag':False, 'center_interpolate': True}            
              else:
              # neither center nor fwhm matches
                return {'flag':False,  'center_interpolate': False}
            match_index_list = match_index_list + [wave_band]
            
    # both center and fwhm match
    return {'flag':True, 'index':match_index_list}        

# from r2 csv file    
def get_sample_size(csv_file, total_bin):
  if os.path.exists(csv_file):
    mid_pnt, n = np.loadtxt(csv_file, delimiter=',', usecols=(0,2), unpack=True,skiprows=1)
    return n[:total_bin]
  else:
    print("Cannot open '{}' for sample size, use equal weight in each bin. ".format(csv_file))
    return None
  

# fill the coefficient values based on NDVI, in linear regression or weighted linear regression method; update the input array new_k
def  interpol_kernel_coeff(ndvi_sub, veg_mask,new_k, coeff_list)  :
    ndvi_flat = ndvi_sub.flatten()

    stack = np.vstack((np.ones(ndvi_flat.shape[0]), ndvi_flat))  #P.polyfit , weighted fitting, 1st order polynomial
    new_kernel = np.matmul(coeff_list, stack).T.reshape((ndvi_sub.shape[0], ndvi_sub.shape[1], coeff_list.shape[0]))

    new_k += new_kernel*veg_mask
    return new_k

# fill the coefficient values based on NDVI, in piecewise linear interpolation method; update the input array new_k
def interpol_1d_kernel_coeff(ndvi_sub, veg_mask,new_k, ndvi_mid_pnt, k_list):

    # k_list shape: m_ndvi_point by n_wavelength
    interp_func = interp1d(ndvi_mid_pnt, k_list, axis=0,fill_value="extrapolate") # fill_value="extrapolate" for border of 0.25~0.275 0.825~0.85

    new_kernel = interp_func(ndvi_sub)
    
    new_k += new_kernel*veg_mask
    return new_k


# prepare header dictionary for H5 in order to output ENVI format file
def h5_make_header_dict(hyObj):

    newheader_dict = {}
    '''
    dtypeDict = {1:np.uint8,
             2:np.int16,
             3:np.int32,
             4:np.float32,
             5:np.float64,
             12:np.uint16,
             13:np.uint32,
             14:np.int64,
             15:np.uint64}
    '''    
    
    newheader_dict["lines"] = hyObj.lines 
    newheader_dict["samples"] = hyObj.columns 
    newheader_dict["bands"] = hyObj.bands 
    newheader_dict["interleave"] = "bil"
    newheader_dict["fwhm"] = hyObj.fwhm 
    newheader_dict["wavelength"] = hyObj.wavelengths
    newheader_dict["wavelength units"] = hyObj.wavelength_units
    
    newheader_dict['data ignore value']  = hyObj.no_data
    newheader_dict['map info'] = hyObj.map_info    
    newheader_dict['projection info'] = hyObj.projection

    newheader_dict["data type"]= 4
    return newheader_dict
    
# make a circular / disc image for image convolusion in order to make mask buffer 
def make_disc_for_buffer(window_radius_size):
  y_grid,x_grid = np.ogrid[-window_radius_size: window_radius_size+1, -window_radius_size: window_radius_size+1]
  return  (x_grid**2+y_grid**2 <= window_radius_size**2).astype(np.float)


    
def main():
    '''
    Perform in-memory trait estimation.
    '''
    
    parser = argparse.ArgumentParser(description = "In memory trait mapping tool.")
    parser.add_argument("-img", help="Input image pathname",required=True, type = str)
    parser.add_argument("--obs", help="Input observables pathname", required=False, type = str)
    parser.add_argument("--out", help="Output full corrected image", required=False, type = str)
    parser.add_argument("-od", help="Output directory for all resulting products", required=True, type = str)
    parser.add_argument("--brdf", help="Perform BRDF correction",type = str, default = '')
    parser.add_argument("--topo", help="Perform topographic correction", type = str, default = '')
    parser.add_argument("--mask", help="Image mask type to use", action='store_true')
    parser.add_argument("--mask_threshold", help="Mask threshold value", nargs = '*', type = float)
    parser.add_argument("--rgbim", help="Export RGBI +Mask image.", action='store_true')
    parser.add_argument("-coeffs", help="Trait coefficients directory", required=False, type = str)
    parser.add_argument("-nodata", help="New value to assign for no_data values", required=False, type = float, default =-9999)
    parser.add_argument("-smooth", help="BRDF smooth methods L: Linear regression; W: Weighted linear regression; I: Linear interpolation", required=False  , choices=['L', 'W', 'I'])
    parser.add_argument("-sszn", help="standard solar zenith angle (degree)", required=False, type = float)
    
    parser.add_argument("--buffer_neon", help="neon buffer", action='store_true')
    
    parser.add_argument("-boxsszn", help="Use box average standard solar zenith angle (degree)", action='store_true')
        
    args = parser.parse_args()

    traits = glob.glob("%s/*.json" % args.coeffs)

    std_solar_zn = None  #float(args.sszn)/180*np.pi
    
    #Load data objects memory
    if args.img.endswith(".h5"):
        hyObj = ht.openHDF(args.img,load_obs = True)
        smoothing_factor = 1
    else:
        hyObj = ht.openENVI(args.img)
        
        smoothing_factor = hyObj.header_dict[NAME_FIELD_SMOOTH]
        if isinstance(smoothing_factor, (list, tuple, np.ndarray)):
        # CORR product has smoothing factor, and all bands are converted back to uncorrected / unsmoothed version by dividing the corr/smooth factors
            smoothing_factor = np.array(smoothing_factor)
        else:
        # REFL version
            smoothing_factor = 1
        
    if (len(args.topo) != 0) | (len(args.brdf) != 0):
        hyObj.load_obs(args.obs)
    if not args.od.endswith("/"):
        args.od+="/"
    hyObj.create_bad_bands(BAD_RANGE)
    
    # no data  / ignored values varies by product
    hyObj.no_data = NO_DATA_VALUE
    
    hyObj.load_data()  


    # Generate mask
    extra_mask=True
    
    if args.mask:
        ir = hyObj.get_wave(BAND_IR_NM)
        red = hyObj.get_wave(BAND_RED_NM)
        ndvi = (ir-red)/(ir+red)
        
        
        if args.buffer_neon: 
          print("Creating buffer of image edge...")
          buffer_edge=sig.convolve2d(ir <= 0.5*hyObj.no_data,make_disc_for_buffer(30), mode = 'same', fillvalue=1)              
          #ag_mask = ag_mask or (buffer_edge>0)
          extra_mask = (buffer_edge == 0)
        
        hyObj.mask = ((ndvi > NDVI_APPLIED_BIN_MIN_THRESHOLD) & (ir != hyObj.no_data)) 

        del ir, red #,ndvi
    else:
        hyObj.mask = np.ones((hyObj.lines,hyObj.columns)).astype(bool)
        print("Warning no mask specified, results may be unreliable!")

    # Generate cosine i and c1 image for topographic correction
    if len(args.topo) != 0:
        with open( args.topo) as json_file:  
            topo_coeffs = json.load(json_file)
            
        topo_coeffs['c'] = np.array(topo_coeffs['c'])   
        cos_i =  calc_cosine_i(hyObj.solar_zn, hyObj.solar_az, hyObj.aspect , hyObj.slope)
        c1 = np.cos(hyObj.solar_zn)
        c2 = np.cos(hyObj.slope)
        
        topomask = hyObj.mask & (cos_i > COSINE_I_MIN_THRESHOLD) & (hyObj.slope > SLOPE_MIN_THRESHOLD)

    
    # Gernerate scattering kernel images for brdf correction
    if len(args.brdf) != 0:
        brdf_coeffs_List = []

        ndvi_thres_complete = False
        if (args.mask_threshold):
          total_bin = len(args.mask_threshold) + 1
          ndvi_thres = [NDVI_APPLIED_BIN_MIN_THRESHOLD] + args.mask_threshold + [NDVI_APPLIED_BIN_MAX_THRESHOLD]
          ndvi_thres_complete = True
        else:
        # read NDVI binning info from existing json files
          #total_bin=1
          #ndvi_thres = [NDVI_APPLIED_BIN_MIN_THRESHOLD, NDVI_APPLIED_BIN_MAX_THRESHOLD]
          total_bin = len(glob.glob(args.brdf + '_brdf_coeffs_*.json'))
          ndvi_thres = [None] * total_bin + [NDVI_APPLIED_BIN_MAX_THRESHOLD] 
          
    
        if args.smooth is None:
          brdfmask = np.ones((total_bin, hyObj.lines,hyObj.columns )).astype(bool)
            
        first_effective_ibin = 0  # in case some bins are missing due to small sample size
        
        for ibin in range(total_bin):
        
          if not os.path.exists(args.brdf + '_brdf_coeffs_' + str(ibin + 1) + '.json'):
            brdf_coeffs_List.append(None)
            print('No ' + args.brdf + '_brdf_coeffs_' + str(ibin + 1) + '.json')
            if args.smooth is None:
              brdfmask[ibin, :, :] = False
            continue
            
          first_effective_ibin = ibin
          
          with open(args.brdf + '_brdf_coeffs_' + str(ibin + 1) + '.json') as json_file:  
            brdf_coeffs = json.load(json_file)
            brdf_coeffs['fVol'] = np.array(brdf_coeffs['fVol'])
            brdf_coeffs['fGeo'] = np.array(brdf_coeffs['fGeo'])
            brdf_coeffs['fIso'] = np.array(brdf_coeffs['fIso'])
            brdf_coeffs_List.append(brdf_coeffs)
            if not ndvi_thres_complete:
              ndvi_thres[ibin] = max(float(brdf_coeffs['ndvi_lower_bound']), NDVI_APPLIED_BIN_MIN_THRESHOLD)
              ndvi_thres[ibin + 1] = min(float(brdf_coeffs['ndvi_upper_bound']), NDVI_APPLIED_BIN_MAX_THRESHOLD)
            
            if std_solar_zn is None:
              print(std_solar_zn)
              if args.sszn:
                std_solar_zn = float(args.sszn) / 180 * np.pi
              elif args.boxsszn:
                std_solar_zn = float(brdf_coeffs['flight_box_avg_sza']) / 180 * np.pi
              else:
                std_solar_zn = -9999
 

          if  args.smooth is None:          
            brdfmask[ibin, :, :] = hyObj.mask & (ndvi > ndvi_thres[ibin]) & (ndvi <= ndvi_thres[ibin + 1])

        if args.smooth is not None:
            n_wave = len(brdf_coeffs_List[first_effective_ibin]['fVol']) 
            fvol_y_n = np.zeros((total_bin, n_wave), dtype=np.float)
            fiso_y_n = np.zeros((total_bin, n_wave), dtype=np.float)
            fgeo_y_n = np.zeros((total_bin, n_wave), dtype=np.float)
            upper_bound_y_n = np.zeros(total_bin)
            lower_bound_y_n = np.zeros(total_bin)
            
            min_lower_bound = 1000
            max_upper_bound = BRDF_VEG_upper_bound 
            
            for ibin in range(total_bin):
              if brdf_coeffs_List[ibin] is None:
                continue
              else:
                n_wave =len(brdf_coeffs_List[ibin]['fVol']) 
                fvol_y_n[ibin, :] = brdf_coeffs_List[ibin]["fVol"]
                fgeo_y_n[ibin, :] = brdf_coeffs_List[ibin]["fGeo"]
                fiso_y_n[ibin, :] = brdf_coeffs_List[ibin]["fIso"]
                upper_bound_y_n[ibin] = float(brdf_coeffs_List[ibin]["ndvi_upper_bound"])
                lower_bound_y_n[ibin] = float(brdf_coeffs_List[ibin]["ndvi_lower_bound"]) 
                min_lower_bound = min(min_lower_bound, lower_bound_y_n[ibin])
                if lower_bound_y_n[ibin] > BRDF_VEG_upper_bound:
                  max_upper_bound = upper_bound_y_n[ibin] + 0.0001
             
            mid_pnt = (upper_bound_y_n + lower_bound_y_n) / 2.0
            
            if args.smooth == 'I':            
                #poly_bin = (upper_bound_y_n<=BRDF_VEG_upper_bound) & (lower_bound_y_n>=BRDF_VEG_lower_bound)
                poly_bin = (upper_bound_y_n <= max_upper_bound) & (lower_bound_y_n >= BRDF_VEG_lower_bound)
                old_bin_low = (lower_bound_y_n < BRDF_VEG_lower_bound)
                old_bin_hi = (upper_bound_y_n > max_upper_bound)
                n_old_bin = np.count_nonzero(old_bin_low) + np.count_nonzero(old_bin_hi)
                outside_list = np.where(old_bin_low | old_bin_hi)[0].tolist() 

                mid_x = mid_pnt[poly_bin]
                yy_vol = fvol_y_n[poly_bin, :]
                yy_geo = fgeo_y_n[poly_bin, :]
                yy_iso =  fiso_y_n[poly_bin, :]

                brdfmask = np.zeros((n_old_bin + 1, hyObj.lines,hyObj.columns)).astype(bool)  # ones?
                for order_bin, ibin in enumerate(outside_list):
                #for ibin in range(n_old_bin):
                  if (upper_bound_y_n[ibin] == lower_bound_y_n[ibin]) and (upper_bound_y_n[ibin] == 0.0):
                  # missing bins will be used for interpolation / extrapolation, add mask to the last bin-mask
                    brdfmask[n_old_bin, :, :] = (brdfmask[n_old_bin, :, :]) | (hyObj.mask & (ndvi > ndvi_thres[ibin]) & (ndvi <= ndvi_thres[ibin + 1]))
                  else:
                  # bins outsize bound will be kept without any extrapolation
                    
                    brdfmask[order_bin, :, :] = hyObj.mask & (ndvi > ndvi_thres[ibin]) & (ndvi <= ndvi_thres[ibin + 1])

                brdfmask[n_old_bin, :, :] = (brdfmask[n_old_bin, :, :]) | (hyObj.mask & (ndvi >= BRDF_VEG_lower_bound) & (ndvi <= max_upper_bound))              
            
            elif args.smooth == 'L' or args.smooth == 'W':
                
                #poly_bin = (upper_bound_y_n<=BRDF_VEG_upper_bound) & (lower_bound_y_n>BRDF_VEG_lower_bound)
                poly_bin = (upper_bound_y_n <= max_upper_bound) & (lower_bound_y_n > BRDF_VEG_lower_bound)
                old_bin = (upper_bound_y_n <= max(BRDF_VEG_lower_bound,min_lower_bound))
                n_old_bin = np.count_nonzero(old_bin)
                
                if args.smooth == 'W':
                  sample_size = get_sample_size(args.brdf + '_brdf_coeffs_r2.csv', total_bin)
                  if sample_size is not None:
                    #wight_thres = np.percentile(sample_size, 50)
                    #sample_size = np.clip(sample_size, 0 , wight_thres)
                    weight_n =  (1.0 * sample_size) / np.sum(sample_size)
                  else:
                    weight_n = np.ones(total_bin)
                else:
                  weight_n = np.ones(total_bin)
                  
                weight_n = weight_n[poly_bin]
                
                mid_x = mid_pnt[poly_bin]
                yy_vol = fvol_y_n[poly_bin, :]
                yy_geo = fgeo_y_n[poly_bin, :]
                yy_iso =  fiso_y_n[poly_bin, :]
                
                coeff_list_vol = np.zeros((n_wave, 2))
                coeff_list_geo = np.zeros((n_wave, 2))
                coeff_list_iso = np.zeros((n_wave, 2))
                
                for j in range(n_wave):
                    yy_vol_ = yy_vol[:, j]
                    yy_geo_ = yy_geo[:, j]
                    yy_iso_ = yy_iso[:, j]
                    
                    coeff_vol = P.polyfit(mid_x, yy_vol_, 1, full=False,w=weight_n)  # order: x^0, x^1, x^2 ... 
                    coeff_geo = P.polyfit(mid_x, yy_geo_, 1, full=False,w=weight_n)
                    coeff_iso = P.polyfit(mid_x, yy_iso_, 1, full=False,w=weight_n)

                    coeff_list_vol[j, :] = coeff_vol
                    coeff_list_geo[j, :] = coeff_geo
                    coeff_list_iso[j, :] = coeff_iso
                    
                brdfmask = np.ones((n_old_bin + 1, hyObj.lines, hyObj.columns)).astype(bool)
                for ibin in range(n_old_bin):
                  brdfmask[ibin, :, :] = hyObj.mask & (ndvi > ndvi_thres[ibin]) & (ndvi <= ndvi_thres[ibin + 1])   
     
                brdfmask[n_old_bin, :, :] = hyObj.mask & (ndvi > NDVI_APPLIED_BIN_MIN_THRESHOLD) & (ndvi < 1.0)                


        
        k_vol = generate_volume_kernel(hyObj.solar_az,hyObj.solar_zn, hyObj.sensor_az, hyObj.sensor_zn, ross = brdf_coeffs_List[first_effective_ibin]['ross'])
        k_geom = generate_geom_kernel(hyObj.solar_az,hyObj.solar_zn, hyObj.sensor_az, hyObj.sensor_zn, li = brdf_coeffs_List[first_effective_ibin]['li'])
        
        print('std_solar_zn', std_solar_zn)
        if std_solar_zn == -9999:
        # NADIR without solor zenith angle normalization
          k_vol_nadir = generate_volume_kernel(hyObj.solar_az, hyObj.solar_zn, hyObj.sensor_az, 0, ross = brdf_coeffs_List[first_effective_ibin]['ross'])
          k_geom_nadir = generate_geom_kernel(hyObj.solar_az, hyObj.solar_zn, hyObj.sensor_az, 0, li = brdf_coeffs_List[first_effective_ibin]['li']) 
        else:       
        # use solor zenith angle normalization, either from flight box average (json), or from user specification
          k_vol_nadir = generate_volume_kernel(np.pi, std_solar_zn, hyObj.sensor_az, 0, ross = brdf_coeffs_List[first_effective_ibin]['ross'])
          k_geom_nadir = generate_geom_kernel(np.pi, std_solar_zn, hyObj.sensor_az, 0, li = brdf_coeffs_List[first_effective_ibin]['li'])

    if len(traits) != 0:
      
        #Cycle through the chunks and apply topo, brdf, vnorm,resampling and trait estimation steps
        print("Calculating values for %s traits....." % len(traits))
        
        
        # Cycle through trait models and gernerate resampler
        trait_waves_all = []
        trait_fwhm_all = []
        
        for i,trait in enumerate(traits):
            with open(trait) as json_file:  
                trait_model = json.load(json_file)
             
            # Check if wavelength units match
            if trait_model['wavelength_units'] == 'micrometers':
                trait_wave_scaler = 10**3
            else:
                trait_wave_scaler = 1    
            
            # Get list of wavelengths to compare against image wavelengths
            if len(trait_model['vector_norm_wavelengths']) == 0:
                trait_waves_all += list(np.array(trait_model['model_wavelengths'])*trait_wave_scaler)
            else:
                trait_waves_all += list(np.array(trait_model['vector_norm_wavelengths'])*trait_wave_scaler)
            
            trait_fwhm_all += list(np.array(trait_model['fwhm'])* trait_wave_scaler)        
               
        # List of all unique pairs of wavelengths and fwhm    
        trait_waves_fwhm = list(set([x for x in zip(trait_waves_all,trait_fwhm_all)]))
        trait_waves_fwhm.sort(key = lambda x: x[0])

        # Create a single set of resampling coefficients for all wavelength and fwhm combos
        #resampling_coeffs = est_transform_matrix(hyObj.wavelengths[hyObj.bad_bands],[x for (x,y) in trait_waves_fwhm] ,hyObj.fwhm[hyObj.bad_bands],[y for (x,y) in trait_waves_fwhm],1)

        # if wavelengths match, no need to resample
        # check_wave_match_result = check_wave_match(hyObj, [x for (x, y) in trait_waves_fwhm])
        check_wave_match_result = check_wave_match(hyObj, trait_waves_fwhm)
        print("match_flag", check_wave_match_result['flag']) 
        if (check_wave_match_result['flag']):
            match_flag = True
        else:
            match_flag = False
            center_interpolate = check_wave_match_result['center_interpolate']
            if not center_interpolate:
                resampling_coeffs = est_transform_matrix(hyObj.wavelengths[hyObj.bad_bands], [x for (x, y) in trait_waves_fwhm], hyObj.fwhm[:hyObj.bands][hyObj.bad_bands], [y for (x, y) in trait_waves_fwhm], 2)  # 2       


    hyObj.wavelengths = hyObj.wavelengths[hyObj.bad_bands]
    
    pixels_processed = 0
    iterator = hyObj.iterate(by = 'chunk',chunk_size = (CHUNK_EDGE_SIZE,hyObj.columns))

    while not iterator.complete:  
        chunk = iterator.read_next()  
        #chunk_nodata_mask = chunk[:,:, BAND_NO_DATA] == hyObj.no_data  
        chunk_nodata_mask = chunk[:,:, BAND_NO_DATA] <= 0.5 * hyObj.no_data
        pixels_processed += chunk.shape[0] * chunk.shape[1]
        progbar(pixels_processed, hyObj.columns * hyObj.lines, 100)
        
        chunk = chunk/smoothing_factor

        # Chunk Array indices
        line_start =iterator.current_line 
        line_end = iterator.current_line + chunk.shape[0]
        col_start = iterator.current_column
        col_end = iterator.current_column + chunk.shape[1]
        
        # Apply TOPO correction 
        if len(args.topo) != 0:
            cos_i_chunk = cos_i[line_start:line_end,col_start:col_end]
            c1_chunk = c1[line_start:line_end,col_start:col_end]
            c2_chunk = c2[line_start:line_end,col_start:col_end]
            topomask_chunk = topomask[line_start:line_end,col_start:col_end,np.newaxis]
            correctionFactor = (c2_chunk[:,:,np.newaxis]*c1_chunk[:,:,np.newaxis]+topo_coeffs['c']  )/(cos_i_chunk[:,:,np.newaxis] + topo_coeffs['c'])
            correctionFactor = correctionFactor*topomask_chunk + 1.0*(1-topomask_chunk)
            chunk = chunk[:,:,hyObj.bad_bands]* correctionFactor
        else:
            chunk = chunk[:,:,hyObj.bad_bands] 
        
        # Apply BRDF correction 
        if len(args.brdf) != 0:
            # Get scattering kernel for chunks
            k_vol_nadir_chunk = k_vol_nadir[line_start:line_end,col_start:col_end]
            k_geom_nadir_chunk = k_geom_nadir[line_start:line_end,col_start:col_end]
            k_vol_chunk = k_vol[line_start:line_end,col_start:col_end]
            k_geom_chunk = k_geom[line_start:line_end,col_start:col_end]
        
            n_wavelength = brdf_coeffs_List[first_effective_ibin]['fVol'].shape[0]
            new_k_vol = np.zeros((chunk.shape[0],chunk.shape[1],n_wavelength),dtype=np.float32)
            new_k_geom = np.zeros((chunk.shape[0],chunk.shape[1],n_wavelength),dtype=np.float32)
            new_k_iso = np.zeros((chunk.shape[0],chunk.shape[1],n_wavelength),dtype=np.float32)

            
            if args.smooth is None:
                for ibin in range(total_bin):

                  if brdf_coeffs_List[ibin] is None:
                    continue

                  veg_mask = brdfmask[ibin,line_start:line_end,col_start:col_end][:,:,np.newaxis]
                  
                  new_k_vol +=  brdf_coeffs_List[ibin]['fVol'] * veg_mask
                  new_k_geom += brdf_coeffs_List[ibin]['fGeo'] * veg_mask
                  new_k_iso += brdf_coeffs_List[ibin]['fIso'] * veg_mask
                  
            else:
              if args.smooth=='I':  

                for ibin in range(n_old_bin):
                  if brdf_coeffs_List[outside_list[ibin]] is None:
                    continue
                  else:
                    veg_mask = brdfmask[ibin,line_start:line_end,col_start:col_end][:,:,np.newaxis]
                    
                    #new_k_vol +=  brdf_coeffs_List[ibin]['fVol'] * veg_mask
                    #new_k_geom += brdf_coeffs_List[ibin]['fGeo'] * veg_mask
                    #new_k_iso += brdf_coeffs_List[ibin]['fIso'] * veg_mask
                    new_k_vol +=  brdf_coeffs_List[outside_list[ibin]]['fVol'] * veg_mask
                    new_k_geom += brdf_coeffs_List[outside_list[ibin]]['fGeo'] * veg_mask
                    new_k_iso += brdf_coeffs_List[outside_list[ibin]]['fIso'] * veg_mask
                    
                veg_mask = brdfmask[n_old_bin,line_start:line_end,col_start:col_end][:,:,np.newaxis]
                ndvi_sub = ndvi[line_start:line_end,col_start:col_end]
                
                new_k_vol = interpol_1d_kernel_coeff(ndvi_sub, veg_mask,new_k_vol, mid_x, yy_vol)
                new_k_geom = interpol_1d_kernel_coeff(ndvi_sub, veg_mask,new_k_geom, mid_x, yy_geo) 
                new_k_iso = interpol_1d_kernel_coeff(ndvi_sub, veg_mask,new_k_iso, mid_x, yy_iso)   
                
              elif args.smooth=='L' or args.smooth=='W':                  
                veg_mask = brdfmask[n_old_bin,line_start:line_end,col_start:col_end][:,:,np.newaxis]
                ndvi_sub = ndvi[line_start:line_end,col_start:col_end]

                new_k_vol = interpol_kernel_coeff(ndvi_sub, veg_mask,new_k_vol, coeff_list_vol)
                new_k_geom = interpol_kernel_coeff(ndvi_sub, veg_mask,new_k_geom,coeff_list_geo)
                new_k_iso = interpol_kernel_coeff(ndvi_sub, veg_mask,new_k_iso,coeff_list_iso) 

                
            # Apply brdf correction 
            # eq 5. Weyermann et al. IEEE-TGARS 2015)
                       
            brdf = np.einsum('ijk,ij-> ijk', new_k_vol,k_vol_chunk) + np.einsum('ijk,ij-> ijk', new_k_geom,k_geom_chunk)  + new_k_iso
            brdf_nadir = np.einsum('ijk,ij-> ijk', new_k_vol,k_vol_nadir_chunk) + np.einsum('ijk,ij-> ijk', new_k_geom,k_geom_nadir_chunk)  + new_k_iso
            correctionFactor = brdf_nadir/brdf  #*veg_total+(1.0-veg_total)
            correctionFactor[brdf == 0.0] = 1.0
            chunk = chunk   * correctionFactor
            
        
        #Reassign no data values
        chunk[chunk_nodata_mask,:] = args.nodata
        

        if len(traits)>0:

            if match_flag==False:  
            # Resample chunk or interpolate chunk
                if center_interpolate:
                # interpolate chunk, only offset appears
                    interp_func =  interp1d(hyObj.wavelengths, chunk, kind='cubic', axis=2, fill_value="extrapolate")
                    chunk_r = interp_func(np.array([x for (x, y) in trait_waves_fwhm]))
                else:
                # fhhm and center of band do not match
                    chunk_r = np.dot(chunk, resampling_coeffs) 
            # subset of chunk
            else:
                chunk_r = chunk[:,:,check_wave_match_result['index']]

        interp_func =  interp1d(hyObj.wavelengths, chunk, kind='cubic', axis=2, fill_value="extrapolate")

        
        # Export RGBIM image
        if args.rgbim:
            dstFile = args.od + os.path.splitext(os.path.basename(args.img))[0] + '_rgbim.tif'
            nband_rgbim = len(RGBIM_BAND)
            
            if line_start + col_start == 0:
                driver = gdal.GetDriverByName("GTIFF")
                tiff = driver.Create(dstFile,hyObj.columns,hyObj.lines,nband_rgbim+1,gdal.GDT_Float32)
                tiff.SetGeoTransform(hyObj.transform)
                tiff.SetProjection(hyObj.projection)
                for band in range(1,nband_rgbim+2):
                    tiff.GetRasterBand(band).SetNoDataValue(args.nodata)
                tiff.GetRasterBand(nband_rgbim+1).WriteArray(hyObj.mask  & extra_mask )

                del tiff,driver
                
            # Write rgbi chunk
            rgbi_geotiff = gdal.Open(dstFile, gdal.GA_Update)
            
            for i,wave in enumerate(RGBIM_BAND,start=1):
                    band = hyObj.wave_to_band(wave)                    
                    rgbi_geotiff.GetRasterBand(i).WriteArray(chunk[:,:,band], col_start, line_start)
                    
            rgbi_geotiff = None
        
        # Export BRDF and topo corrected image
        if args.out:
            if line_start + col_start == 0:
                output_name = args.od + os.path.splitext(os.path.basename(args.img))[0] + args.out 

                if isinstance( hyObj.header_dict, dict): 
                    # ENVI
                    header_dict =hyObj.header_dict
                    header_dict['wavelength']= header_dict['wavelength'][hyObj.bad_bands]
                else:    
                    #HDF5
                    header_dict = h5_make_header_dict(hyObj) # bad bands removed


                # Update header
                
                header_dict['fwhm'] = header_dict['fwhm'][hyObj.bad_bands]
                #header_dict['bbl'] = header_dict['bbl'][hyObj.bad_bands]
                if 'band names' in header_dict:  
                  del header_dict['band names']
                header_dict['bands'] = int(hyObj.bad_bands.sum())
                
                # clean ENVI header
                header_dict.pop('band names', None)
                header_dict.pop(NAME_FIELD_SMOOTH, None)
                header_dict.pop('bbl', None)                
                header_dict.pop('smoothing factors', None)               
                
                writer = writeENVI(output_name,header_dict)
            writer.write_chunk(chunk,iterator.current_line,iterator.current_column)
            if iterator.complete:
                writer.close()
                
        for i,trait in enumerate(traits):
            dstFile = args.od + os.path.splitext(os.path.basename(args.img))[0] +"_" +os.path.splitext(os.path.basename(trait))[0] +".tif"
            
            # Trait estimation preparation
            if line_start + col_start == 0:
                
                with open(trait) as json_file:  
                    trait_model = json.load(json_file)
       
                intercept = np.array(trait_model['intercept'])
                coefficients = np.array(trait_model['coefficients'])
                transform = trait_model['transform']
                
                # Get list of wavelengths to compare against image wavelengths
                if len(trait_model['vector_norm_wavelengths']) == 0:
                    dst_waves = np.array(trait_model['model_wavelengths'])*trait_wave_scaler
                else:
                    dst_waves = np.array(trait_model['vector_norm_wavelengths'])*trait_wave_scaler
                
                dst_fwhm = np.array(trait_model['fwhm'])* trait_wave_scaler
                model_waves = np.array(trait_model['model_wavelengths'])* trait_wave_scaler
                model_fwhm = [dict(zip(dst_waves, dst_fwhm))[x] for x in model_waves]
                
                vnorm_band_mask = [x in zip(dst_waves,dst_fwhm) for x in trait_waves_fwhm]
                model_band_mask = [x in zip(model_waves,model_fwhm) for x in trait_waves_fwhm]
                
                vnorm_band_mask = np.array(vnorm_band_mask)  # convert list to numpy array, otherwise True/False will be treated as 1/0, which is the 2nd/1st band 
                model_band_mask = np.array(model_band_mask)   # convert list to numpy array, otherwise True/False will be treated as 1/0, which is the 2nd/1st band

                if trait_model['vector_norm']:
                    vnorm_scaler = trait_model["vector_scaler"]
                else:
                    vnorm_scaler = None

                # Initialize trait dictionary
                if i == 0:
                    trait_dict = {}
                trait_dict[i] = [coefficients,intercept,trait_model['vector_norm'],vnorm_scaler,vnorm_band_mask,model_band_mask,transform]
        
                # Create geotiff driver
                driver = gdal.GetDriverByName("GTIFF")
                
                if args.buffer_neon:
                  tiff = driver.Create(dstFile,hyObj.columns,hyObj.lines,3,gdal.GDT_Float32, options=["INTERLEAVE=BAND"])
                  tiff.GetRasterBand(3).WriteArray(hyObj.mask  & extra_mask )
                  tiff.GetRasterBand(3).SetDescription("Buffered Mask")
                else:
                  tiff = driver.Create(dstFile,hyObj.columns,hyObj.lines,2,gdal.GDT_Float32, options=["INTERLEAVE=BAND"])  # , "TILED=YES" ,"COMPRESS=LZW"
                
                tiff.SetGeoTransform(hyObj.transform)
                tiff.SetProjection(hyObj.projection)
                tiff.GetRasterBand(1).SetNoDataValue(args.nodata)
                tiff.GetRasterBand(2).SetNoDataValue(args.nodata)
                tiff.GetRasterBand(1).SetDescription("Model Mean")
                tiff.GetRasterBand(2).SetDescription("Model Standard Deviation")

                del tiff,driver
            
            coefficients,intercept,vnorm,vnorm_scaler,vnorm_band_mask,model_band_mask,transform = trait_dict[i]

            chunk_t =np.copy(chunk_r)

            if vnorm:                    
                chunk_t[:,:,vnorm_band_mask] = vector_normalize_chunk(chunk_t[:,:,vnorm_band_mask],vnorm_scaler)

            
            if transform == "log(1/R)":
                chunk_t[:,:,model_band_mask] = np.log(1/chunk_t[:,:,model_band_mask] )
                

            trait_mean,trait_std = apply_plsr_chunk(chunk_t[:,:,model_band_mask],coefficients,intercept)
            
            
            # Change no data pixel values
            trait_mean[chunk_nodata_mask] = args.nodata
            trait_std[chunk_nodata_mask] = args.nodata

            # Write trait estimate to file
            trait_geotiff = gdal.Open(dstFile, gdal.GA_Update)
            trait_geotiff.GetRasterBand(1).WriteArray(trait_mean, col_start, line_start)
            trait_geotiff.GetRasterBand(2).WriteArray(trait_std, col_start, line_start)
            trait_geotiff = None

if __name__== "__main__":
    main()
