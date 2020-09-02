
import argparse,warnings,copy
import numpy as np, os,json, sys
from scipy import stats

#sys.path.append('./HyTools-sandbox')   # need to modify the path

import hytools as ht
from hytools.brdf import *
from hytools.topo_correction import *
from hytools.helpers import *
home = os.path.expanduser("~")

warnings.filterwarnings("ignore")

###########################################
############Constant Section####################
###########################################

# No data value of input images
NO_DATA_VALUE = -9999  #-0.9999  # -9999 

# The value that replace NaN in diagnostic tables
DIAGNO_NAN_OUTPUT = -9999 

# The C value for image with relatively flat terrain, it will make the TOPO correction factor to be close to 1
FLAT_COEFF_C = -9999

# Data range considered from image data in BRDF coefficient estimation
REFL_MIN_THRESHOLD = 0.001 # 10
REFL_MAX_THRESHOLD = 0.9  # 9000

# NDVI range for image mask
NDVI_MIN_THRESHOLD = 0.01
NDVI_MAX_THRESHOLD = 1.0

# NDVI range for data considered in bins in BRDF coefficient estimation. Pixels outside BIN range will not be corrected for BRDF effect 
NDVI_BIN_MIN_THRESHOLD = 0.05 #0.005
NDVI_BIN_MAX_THRESHOLD = 1.0

# Thresholds for Topographic correction. Pixels beyond those ranges will not be corrected for topographic effect. 
# Minimum illumination value (cosine of the incident angle): 0.12 ( < 83.1 degrees)
# Minimum slope: 5 degrees
COSINE_I_MIN_THRESHOLD = 0.12
SLOPE_MIN_THRESHOLD = 0.087
SAMPLE_SLOPE_MIN_THRESHOLD = 0.03

# BRDF coefficients in an NDVI bin with sample size less than this threshold will not be estimated in BRDF correction. Its coefficients might be estimated by its neighboring NDVI bins in later steps.
MIN_SAMPLE_COUNT = 100

# if there are too few pixels with terrain (above certian threshold), no TOPO correction factor will be close to 1
MIN_SAMPLE_COUNT_TOPO = 100

# Thresholds for BRDF correction. Pixels beyond this range will not be used for BRDF coefficients estimation.
SENSOR_ZENITH_MIN_DEG = 2

# Wavelengths for NDVI calculation, unit: nanometers
BAND_IR_NM = 850
BAND_RED_NM = 665

# Bad band range. Bands within these ranges will be treated as bad bands, unit: nanometers
BAD_RANGE =  [[300,400],[1320,1430],[1800,1960],[2450,2600]]  #[[300,400],[1330,1430],[1800,1960],[2450,2600]]

# Name field of correction /  smoothing factors. It apprears in the header file of CORR product.
NAME_FIELD_SMOOTH = 'correction factors'

# NDVI range for NDVI-based BRDF coefficients interpolation / regression. NDVI bins outside the range will not be used for BRDF coefficients interpolation or regression. 
BRDF_VEG_upper_bound = 0.85
BRDF_VEG_lower_bound = 0.25

###########################################

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
  

# Calculate R Squared and rmse for a multiple regression
def cal_r2(y,x1,x2,a):

  nn = y.shape[0]
  if nn<10:
    return DIAGNO_NAN_OUTPUT, nn, DIAGNO_NAN_OUTPUT

  est_y =  a[0]*x1+a[1]*x2+a[2]
  
  avg_y = np.mean(y)
  avg_est_y = np.mean(est_y)
  
  ss_total = np.sum((y-avg_y)**2)
  ss_res =  np.sum((y-est_y)**2)
  r_2 = 1.0-ss_res/ss_total
  
  rmse = np.sqrt(ss_res/nn)
  
  return r_2, nn, rmse

# Calculate R Squared and rmse for a single regression
def cal_r2_single(x, y):

  nn = y.shape[0]

  slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
  est_y =  intercept + x*slope
  
  avg_y = np.mean(y)
  avg_est_y = np.mean(est_y)
  
  ss_total = np.sum((y-avg_y)**2)
  ss_res =  np.sum((y-est_y)**2)
  r_2 = 1.0-ss_res/ss_total
  
  rmse = np.sqrt(ss_res/nn)
  
  return slope, intercept, r_value, p_value, std_err,  rmse   #r_2,
    
def main():
    '''
    Generate topographic and BRDF correction coefficients. Corrections can be calculated on individual images
    or groups of images.
    '''
    parser = argparse.ArgumentParser(description = "In memory trait mapping tool.")
    parser.add_argument("--img", help="Input image/directory pathname",required=True,nargs = '*', type = str)
    parser.add_argument("--obs", help="Input observables pathname", required=False,nargs = '*', type = str)
    parser.add_argument("--od", help="Ouput directory", required=True, type = str)
    parser.add_argument("--pref", help="Coefficient filename prefix", required=True, type = str)
    parser.add_argument("--brdf", help="Perform BRDF correction",action='store_true')
    parser.add_argument("--kernels", help="Li and Ross kernel types",nargs = 2, type =str)
    parser.add_argument("--topo", help="Perform topographic correction", action='store_true')
    parser.add_argument("--mask", help="Image mask type to use", action='store_true')
    parser.add_argument("--mask_threshold", help="Mask threshold value", nargs = '*', type = float)
    parser.add_argument("--samp_perc", help="Percent of unmasked pixels to sample", type = float,default=1.0)
    parser.add_argument("--agmask", help="ag / urban mask file", required=False, type = str)
    parser.add_argument("--topo_sep", help="In multiple image mode, perform topographic correction in a image-based fasion", action='store_true')
    
    args = parser.parse_args()   


    if not args.od.endswith("/"):
        args.od+="/"

    if len(args.img) == 1:
        image = args.img[0]
    
        #Load data objects memory
        if image.endswith(".h5"):
            hyObj = ht.openHDF(image,load_obs = True)
            smoothing_factor = np.ones(hyObj.bands)
        else:
            hyObj = ht.openENVI(image)
            hyObj.load_obs(args.obs[0])
            
            smoothing_factor = hyObj.header_dict[NAME_FIELD_SMOOTH]
            if isinstance(smoothing_factor, (list, tuple, np.ndarray)):
            # CORR product has smoothing factor, and all bands are converted back to uncorrected / unsmoothed version by dividing the corr/smooth factors
                smoothing_factor = np.array(smoothing_factor)
            else:
            # REFL version
                smoothing_factor = np.ones(hyObj.bands)
            
            
        hyObj.create_bad_bands(BAD_RANGE)
        
        # no data  / ignored values varies by product
        hyObj.no_data = NO_DATA_VALUE
        
        hyObj.load_data()


        
        # Generate mask
        if args.mask:
            ir = hyObj.get_wave(BAND_IR_NM)
            red = hyObj.get_wave(BAND_RED_NM)
            ndvi = (1.0*ir-red)/(1.0*ir+red)

            ag_mask=0
            if args.agmask:
              ag_mask =  np.fromfile(args.agmask, dtype=np.uint8).reshape((hyObj.lines,hyObj.columns))

            hyObj.mask = (ndvi > NDVI_MIN_THRESHOLD) & (ndvi < NDVI_MAX_THRESHOLD) & (ir != hyObj.no_data) & (ag_mask ==0)
            del ir,red #,ndvi
        else:
            hyObj.mask = np.ones((hyObj.lines,hyObj.columns)).astype(bool)
            print("Warning no mask specified, results may be unreliable!")

        # Generate cosine i and c1 image for topographic correction
        
        if args.topo:    
            
            topo_coeffs = {}
            topo_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist() 
            topo_coeffs['c'] = []
            cos_i = calc_cosine_i(hyObj.solar_zn, hyObj.solar_az, hyObj.azimuth , hyObj.slope)
            c1 = np.cos(hyObj.solar_zn) 
            c2 = np.cos(hyObj.slope)
            
            terrain_msk = (cos_i > COSINE_I_MIN_THRESHOLD)  & (hyObj.slope > SLOPE_MIN_THRESHOLD)
            topomask = hyObj.mask #& (cos_i > 0.12)  & (hyObj.slope > 0.087) 
              
        # Generate scattering kernel images for brdf correction
        if args.brdf:
        
            if args.mask_threshold:
              ndvi_thres = [NDVI_BIN_MIN_THRESHOLD]+ args.mask_threshold +[NDVI_BIN_MAX_THRESHOLD]
              total_bin = len(args.mask_threshold)+1
            else:
              ndvi_thres = [NDVI_BIN_MIN_THRESHOLD , NDVI_BIN_MAX_THRESHOLD]
              total_bin = 1  

            brdfmask = np.ones(( total_bin, hyObj.lines, hyObj.columns )).astype(bool)
            
            for ibin in range(total_bin):
              brdfmask[ibin,:,:] = hyObj.mask & (ndvi > ndvi_thres[ibin]) & (ndvi <= ndvi_thres[ibin+1]) &  (hyObj.sensor_zn > np.radians(SENSOR_ZENITH_MIN_DEG))

        
            li,ross =  args.kernels
            # Initialize BRDF dictionary
            
            brdf_coeffs_List = [] #initialize
            brdf_mask_stat = np.zeros(total_bin)
            
            for ibin in range(total_bin):
                brdf_mask_stat[ibin] = np.count_nonzero(brdfmask[ibin,:,:])
                  
                brdf_coeffs = {}
                brdf_coeffs['li'] = li
                brdf_coeffs['ross'] = ross
                brdf_coeffs['ndvi_lower_bound'] = ndvi_thres[ibin]
                brdf_coeffs['ndvi_upper_bound'] = ndvi_thres[ibin+1]
                brdf_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist() 
                brdf_coeffs['fVol'] = []
                brdf_coeffs['fGeo'] = []
                brdf_coeffs['fIso'] = []
                brdf_coeffs_List.append(brdf_coeffs)
                   
            k_vol = generate_volume_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn, ross = ross)
            k_geom = generate_geom_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn,li = li)
            k_finite = np.isfinite(k_vol) & np.isfinite(k_geom)
    
        # Cycle through the bands and calculate the topographic and BRDF correction coefficients
        print("Calculating image correction coefficients.....")
        iterator = hyObj.iterate(by = 'band')
        
        if args.topo or args.brdf:
            while not iterator.complete:   

                if hyObj.bad_bands[iterator.current_band+1]: # load data to RAM, if it is a goog band
                  band = iterator.read_next() 
                  band = band / smoothing_factor[iterator.current_band]
                  band_msk = (band> REFL_MIN_THRESHOLD) & (band<REFL_MAX_THRESHOLD)
                  
                else: #  similar to .read_next(), but do not load data to RAM, if it is a bad band
                  iterator.current_band +=1
                  if iterator.current_band == hyObj.bands-1:
                    iterator.complete = True                
                
                progbar(iterator.current_band+1, len(hyObj.wavelengths), 100)
                #Skip bad bands
                if hyObj.bad_bands[iterator.current_band]:
                    # Generate topo correction coefficients
                    if args.topo:
 
                        topomask_b = topomask & band_msk
                        
                        if np.count_nonzero(topomask_b & terrain_msk) > MIN_SAMPLE_COUNT_TOPO: 
                          topo_coeff= generate_topo_coeff_band(band,topomask_b & terrain_msk,cos_i)
                        else:
                          topo_coeff= FLAT_COEFF_C
                          
                        topo_coeffs['c'].append(topo_coeff)

                    # Gernerate BRDF correction coefficients
                    if args.brdf:
                        if args.topo:
                            # Apply topo correction to current band                            
                            correctionFactor = (c2*c1 + topo_coeff  )/(cos_i + topo_coeff)
                            correctionFactor = correctionFactor*topomask_b + 1.0*(1-topomask_b) # only apply to orographic area
                            band = band* correctionFactor
                            
                        for ibin in range(total_bin):

                            if brdf_mask_stat[ibin]<MIN_SAMPLE_COUNT:
                              continue
                              
                            band_msk_new = (band> REFL_MIN_THRESHOLD) & (band< REFL_MAX_THRESHOLD)  
                            
                            if np.count_nonzero(brdfmask[ibin,:,:] & band_msk & k_finite & band_msk_new) < MIN_SAMPLE_COUNT:
                              brdf_mask_stat[ibin] = DIAGNO_NAN_OUTPUT
                              continue
                              
                            fVol,fGeo,fIso =  generate_brdf_coeff_band(band,brdfmask[ibin,:,:] & band_msk & k_finite & band_msk_new ,k_vol,k_geom)
                            brdf_coeffs_List[ibin]['fVol'].append(fVol)
                            brdf_coeffs_List[ibin]['fGeo'].append(fGeo)
                            brdf_coeffs_List[ibin]['fIso'].append(fIso)
            #print()

           
    # Compute topographic and BRDF coefficients using data from multiple scenes
    elif len(args.img) > 1:

        if args.brdf:
        
            li,ross =  args.kernels
            if args.mask_threshold:
              ndvi_thres = [NDVI_BIN_MIN_THRESHOLD]+ args.mask_threshold +[ NDVI_BIN_MAX_THRESHOLD ]
              total_bin = len(args.mask_threshold)+1
            else:
              ndvi_thres = [NDVI_BIN_MIN_THRESHOLD, NDVI_BIN_MAX_THRESHOLD]
              total_bin = 1   

            brdf_coeffs_List = [] #initialize
            brdf_mask_stat = np.zeros(total_bin)

            for ibin in range(total_bin):
                brdf_coeffs = {}
                brdf_coeffs['li'] = li
                brdf_coeffs['ross'] = ross
                brdf_coeffs['ndvi_lower_bound'] = ndvi_thres[ibin]
                brdf_coeffs['ndvi_upper_bound'] = ndvi_thres[ibin+1]
                #brdf_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist() 
                brdf_coeffs['fVol'] = []
                brdf_coeffs['fGeo'] = []
                brdf_coeffs['fIso'] = []
                brdf_coeffs_List.append(brdf_coeffs)

            
        hyObj_dict = {}
        sample_dict = {}
        sample_k_vol = []
        sample_k_geom = []
        sample_cos_i = []
        sample_c1 = []
        sample_slope = []
        sample_ndvi = []
        sample_index = [0]
        sample_img_tag = [] # record which image that sample is drawn from
        sub_total_sample_size = 0
        ndvi_mask_dict  = {}
        image_smooth = []
        
        for i,image in enumerate(args.img):
            #Load data objects memory
            if image.endswith(".h5"):
                hyObj = ht.openHDF(image,load_obs = True)
                smoothing_factor = np.ones(hyObj.bands)
                image_smooth += [smoothing_factor]  
            else:
                hyObj = ht.openENVI(image)
                hyObj.load_obs(args.obs[i])
                
                smoothing_factor = hyObj.header_dict[NAME_FIELD_SMOOTH]
                if isinstance(smoothing_factor, (list, tuple, np.ndarray)):
                # CORR product has smoothing factor, and all bands are converted back to uncorrected / unsmoothed version by dividing the corr/smooth factors
                    smoothing_factor = np.array(smoothing_factor)
                else:
                # REFL version
                    smoothing_factor = np.ones(hyObj.bands)
                image_smooth += [smoothing_factor]                
                
            hyObj.create_bad_bands(BAD_RANGE)
            hyObj.no_data = NO_DATA_VALUE
            hyObj.load_data()
            

            
            # Generate mask
            if args.mask:
                ir = hyObj.get_wave(BAND_IR_NM)
                red = hyObj.get_wave(BAND_RED_NM)
                ndvi = (1.0*ir-red)/(1.0*ir+red)
                hyObj.mask  =  (ndvi > NDVI_MIN_THRESHOLD) & (ndvi <= NDVI_MAX_THRESHOLD) & (ir != hyObj.no_data)#(ndvi > .5) & (ir != hyObj.no_data) #(ndvi > .01) & (ir != hyObj.no_data)

                del ir,red  #,ndvi
            else:
                hyObj.mask = np.ones((hyObj.lines,hyObj.columns)).astype(bool)
                print("Warning no mask specified, results may be unreliable!")
    
            # Generate sampling mask
            sampleArray = np.zeros(hyObj.mask.shape).astype(bool)
            idx = np.array(np.where(hyObj.mask == True)).T
            
            #np.random.seed(0)  # just for test
            
            idxRand= idx[np.random.choice(range(len(idx)),int(len(idx)*args.samp_perc), replace = False)].T   # actually used
            sampleArray[idxRand[0],idxRand[1]] = True
            sample_dict[i] = sampleArray
            
            print(idxRand.shape)
            sub_total_sample_size+=idxRand.shape[1]
            sample_index = sample_index+ [sub_total_sample_size]
            
            # Initialize and store band iterator
            hyObj_dict[i] = copy.copy(hyObj).iterate(by = 'band')
            
            # Generate cosine i and slope samples
            sample_cos_i += calc_cosine_i(hyObj.solar_zn, hyObj.solar_az, hyObj.azimuth , hyObj.slope)[sampleArray].tolist()
            sample_slope += (hyObj.slope)[sampleArray].tolist()
            
            # Generate c1 samples for topographic correction
            if args.topo:   
                # Initialize topographic correction dictionary
                topo_coeffs = {}
                topo_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist() 
                topo_coeffs['c'] = []
                sample_c1 += (np.cos(hyObj.solar_zn) * np.cos( hyObj.slope))[sampleArray].tolist()
            
            # Gernerate scattering kernel samples for brdf correction
            if args.brdf:
            
                sample_ndvi += (ndvi)[sampleArray].tolist()
                sample_img_tag += [i+1]*idxRand.shape[1] # start from 1
                               
                for ibin in range(total_bin):
                    brdf_coeffs_List[ibin]['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist()     
                
                sample_k_vol += generate_volume_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn, ross = ross)[sampleArray].tolist()
                sample_k_geom += generate_geom_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn,li = li)[sampleArray].tolist()
        
            #del ndvi, topomask, brdfmask
            if args.mask:
                del ndvi
            
        sample_k_vol = np.array(sample_k_vol)
        sample_k_geom = np.array(sample_k_geom)
        sample_cos_i = np.array(sample_cos_i)
        sample_c1= np.array(sample_c1)
        sample_slope = np.array(sample_slope)
        sample_ndvi = np.array(sample_ndvi)
        sample_img_tag = np.array(sample_img_tag)
        
        if args.topo_sep:
            topo_coeff_list=[]
            for _ in range(len(args.img)):
                topo_coeff_list+= [{'c':[], 'wavelengths':hyObj.wavelengths[hyObj.bad_bands].tolist()}] 

        if args.brdf:
            for ibin in range(total_bin):
                ndvi_mask = (sample_ndvi>brdf_coeffs_List[ibin]['ndvi_lower_bound']) & (sample_ndvi <=brdf_coeffs_List[ibin]['ndvi_upper_bound'])
                ndvi_mask_dict[ibin] = ndvi_mask
                count_ndvi= np.count_nonzero(ndvi_mask)
                brdf_mask_stat[ibin] = count_ndvi

            # initialize arrays for BRDF coefficient estimation diagnostic files
            total_image = len(args.img)

            # r-squared array
            r_squared_array = np.ndarray((total_bin*(total_image+1),3+len(hyObj.wavelengths)),dtype=object) # total + flightline by flightline
            r_squared_array[:]=0
            
            r_squared_array[:,0] =  (0.5*(np.array(ndvi_thres[:-1])+np.array(ndvi_thres[1:]))).tolist()*( total_image+1) 
            r_squared_array[:total_bin,1] = 'group'
            r_squared_array[total_bin:,1] =   np.repeat(np.array([os.path.basename(x) for x in args.img]) , total_bin, axis=0) 

            r2_header = 'NDVI_Bin_Center,Flightline,Sample_Size,'+ ','.join('B'+str(wavename) for wavename in hyObj.wavelengths)
          
            # RMSE array
            rmse_array = np.copy(r_squared_array)
            
            # BRDF coefficient array, volumetric+geometric+isotopic
            brdf_coeff_array = np.ndarray((total_bin+8,3+3*len(hyObj.wavelengths)),dtype=object)
            brdf_coeff_array[:]=0
            brdf_coeff_array[:total_bin,0] = 0.5*(np.array(ndvi_thres[:-1])+np.array(ndvi_thres[1:]))
            
            brdf_coeffs_header = 'NDVI_Bin_Center,Sample_Size,'+ ','.join('B'+str(wavename)+'_vol' for wavename in hyObj.wavelengths) +',' + ','.join('B'+str(wavename)+'_geo' for wavename in hyObj.wavelengths)+',' + ','.join('B'+str(wavename)+'_iso' for wavename in hyObj.wavelengths)
            brdf_coeff_array[total_bin+1:total_bin+7,0] = ['slope', 'intercept', 'r_value', 'p_value', 'std_err',  'rmse']


                
        # Calculate bandwise correction coefficients
        print("Calculating image correction coefficients.....")
        current_progress = 0
        
        for w,wave in enumerate(hyObj.wavelengths):
            progbar(current_progress, len(hyObj.wavelengths) * len(args.img), 100)
            wave_samples = []
            for i,image in enumerate(args.img):
                
                if hyObj.bad_bands[hyObj_dict[i].current_band+1]: # load data to RAM, if it is a goog band
                  wave_samples +=  hyObj_dict[i].read_next()[sample_dict[i]].tolist()
                  
                else: #  similar to .read_next(), but do not load data to RAM, if it is a bad band
                  hyObj_dict[i].current_band +=1
                  if hyObj_dict[i].current_band == hyObj_dict[i].bands-1:
                    hyObj_dict[i].complete = True
                  
                current_progress+=1
            
            if hyObj.bad_bands[hyObj_dict[i].current_band]:
            
                wave_samples = np.array(wave_samples)
                
                for i_img_tag in range(len(args.img)):
                  img_tag_true = sample_img_tag==i_img_tag+1
                  wave_samples[img_tag_true] = wave_samples[img_tag_true] / image_smooth[i_img_tag][w]

                # Generate cosine i and c1 image for topographic correction
                if args.topo:    
                  if args.topo_sep==False:
                    topo_coeff  = generate_topo_coeff_band(wave_samples,(wave_samples> REFL_MIN_THRESHOLD) & (wave_samples<REFL_MAX_THRESHOLD ) & (sample_cos_i> COSINE_I_MIN_THRESHOLD) &  (sample_slope> SLOPE_MIN_THRESHOLD) ,sample_cos_i)
                    topo_coeffs['c'].append(topo_coeff)
                    correctionFactor = (sample_c1 + topo_coeff)/(sample_cos_i + topo_coeff)
                    wave_samples = wave_samples* correctionFactor
                  else:                   

                    for i in range(len(args.img)):
                        
                        wave_samples_sub = wave_samples[sample_index[i]:sample_index[i+1]]
                        sample_cos_i_sub = sample_cos_i[sample_index[i]:sample_index[i+1]]
                        sample_slope_sub = sample_slope[sample_index[i]:sample_index[i+1]]
                        sample_c1_sub = sample_c1[sample_index[i]:sample_index[i+1]]

                        if np.count_nonzero((sample_cos_i_sub> COSINE_I_MIN_THRESHOLD) &  (sample_slope_sub> SLOPE_MIN_THRESHOLD))>MIN_SAMPLE_COUNT_TOPO:
                          topo_coeff  = generate_topo_coeff_band(wave_samples_sub,(wave_samples_sub> REFL_MIN_THRESHOLD) & (wave_samples_sub< REFL_MAX_THRESHOLD) & (sample_cos_i_sub> COSINE_I_MIN_THRESHOLD) &  (sample_slope_sub> SLOPE_MIN_THRESHOLD) ,sample_cos_i_sub)
                        else:
                          topo_coeff=FLAT_COEFF_C                       
                          
                        topo_coeff_list[i]['c'].append(topo_coeff)

                        correctionFactor = (sample_c1_sub + topo_coeff)/(sample_cos_i_sub + topo_coeff)
                        wave_samples[sample_index[i]:sample_index[i+1]] = wave_samples_sub* correctionFactor
                        
                # Gernerate scattering kernel images for brdf correction
                if args.brdf:
                
                    temp_mask = (wave_samples> REFL_MIN_THRESHOLD) & (wave_samples< REFL_MAX_THRESHOLD) & np.isfinite(sample_k_vol) & np.isfinite(sample_k_geom)
                    temp_mask = temp_mask & (sample_cos_i> COSINE_I_MIN_THRESHOLD) &  (sample_slope> SAMPLE_SLOPE_MIN_THRESHOLD) 
                
                    for ibin in range(total_bin):
                        
                        # skip BINs that has not enough samples in diagnostic output
                        if brdf_mask_stat[ibin]<MIN_SAMPLE_COUNT or np.count_nonzero(temp_mask)< MIN_SAMPLE_COUNT:
                          r_squared_array[range(ibin,total_bin*(total_image+1),total_bin),w+3]= DIAGNO_NAN_OUTPUT  #0.0
                          rmse_array[range(ibin,total_bin*(total_image+1),total_bin),w+3]= DIAGNO_NAN_OUTPUT
                          brdf_mask_stat[ibin] = brdf_mask_stat[ibin] + DIAGNO_NAN_OUTPUT
                          continue
                          
                        fVol,fGeo,fIso = generate_brdf_coeff_band(wave_samples, temp_mask & ndvi_mask_dict[ibin]  ,sample_k_vol,sample_k_geom)
                        
                        mask_sub = temp_mask & ndvi_mask_dict[ibin]
                        r_squared_array[ibin,2] = wave_samples[mask_sub].shape[0]
                        est_r2, sample_nn, rmse_total = cal_r2(wave_samples[mask_sub],sample_k_vol[mask_sub],sample_k_geom[mask_sub],[fVol,fGeo,fIso])
                        r_squared_array[ibin,w+3]=est_r2
                        rmse_array[ibin,w+3]=rmse_total
                        rmse_array[ibin,2] = r_squared_array[ibin,2]
                        
                        brdf_coeff_array[ibin,1] = wave_samples[mask_sub].shape[0]
                        
                        # update diagnostic information scene by scene
                        for img_order in range(total_image):

                          img_mask_sub = (sample_img_tag==(img_order+1)) & mask_sub

                          est_r2, sample_nn, rmse_bin = cal_r2(wave_samples[img_mask_sub],sample_k_vol[img_mask_sub],sample_k_geom[img_mask_sub],[fVol,fGeo,fIso])
                          r_squared_array[ibin+(img_order+1)*total_bin,w+3]=est_r2
                          r_squared_array[ibin+(img_order+1)*total_bin,2]=max(sample_nn,int(r_squared_array[ibin+(img_order+1)*total_bin,2])) # update many times

                          rmse_array[ibin+(img_order+1)*total_bin,w+3]=rmse_bin
                          rmse_array[ibin+(img_order+1)*total_bin,2]=r_squared_array[ibin+(img_order+1)*total_bin,2]
                        
                        brdf_coeffs_List[ibin]['fVol'].append(fVol)
                        brdf_coeffs_List[ibin]['fGeo'].append(fGeo)
                        brdf_coeffs_List[ibin]['fIso'].append(fIso)
                
                        # save the same coefficient information in diagnostic arrays
                        brdf_coeff_array[ibin,2+w] = fVol
                        brdf_coeff_array[ibin,2+w+len(hyObj.wavelengths)] = fGeo
                        brdf_coeff_array[ibin,2+w+2*len(hyObj.wavelengths)] = fIso
    
                    # update array for BRDF output diagnostic files 
                    mid_ndvi_list = brdf_coeff_array[:total_bin,0].astype(np.float)
                    if np.count_nonzero(brdf_coeff_array[:, 2+w]) > 3:
                    # check linearity( NDVI as X v.s. kernel coefficients as Y ), save to diagnostic file, BIN by BIn, and wavelength by wavelength

                      # volumetric coefficients 
                      temp_y = brdf_coeff_array[:total_bin, 2+w].astype(np.float)
                      slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin,  rmse_ndvi_bin = cal_r2_single(mid_ndvi_list[ (mid_ndvi_list>BRDF_VEG_lower_bound) & (temp_y != 0)], temp_y[(mid_ndvi_list>BRDF_VEG_lower_bound) & (temp_y != 0)] )
                      brdf_coeff_array[total_bin+1:total_bin+7,2+w] = slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin,  rmse_ndvi_bin
                
                      # geometric coefficients
                      temp_y = brdf_coeff_array[:total_bin, 2+w+1*len(hyObj.wavelengths)].astype(np.float)
                      slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin,  rmse_ndvi_bin = cal_r2_single(mid_ndvi_list[(mid_ndvi_list>BRDF_VEG_lower_bound) & (temp_y != 0)], temp_y[(mid_ndvi_list>BRDF_VEG_lower_bound) & (temp_y != 0)])
                      brdf_coeff_array[total_bin+1:total_bin+7,2+w+len(hyObj.wavelengths)] = slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin, rmse_ndvi_bin

                      # isotropic coefficients
                      temp_y = brdf_coeff_array[:total_bin, 2+w+2*len(hyObj.wavelengths)].astype(np.float)
                      slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin,  rmse_ndvi_bin = cal_r2_single(mid_ndvi_list[(mid_ndvi_list>BRDF_VEG_lower_bound) & (temp_y != 0)], temp_y[(mid_ndvi_list>BRDF_VEG_lower_bound) & (temp_y != 0)])
                      brdf_coeff_array[total_bin+1:total_bin+7,2+w+2*len(hyObj.wavelengths)] = slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin, rmse_ndvi_bin               
                
    # Export coefficients to JSON           
    if args.topo:
      if (args.topo_sep==False) or (len(args.img) ==1):
        topo_json = "%s%s_topo_coeffs.json" % (args.od,args.pref)
        with open(topo_json, 'w') as outfile:
            json.dump(topo_coeffs,outfile) 
      else:
        for i_img in range(len(args.img)):
            filename_pref = (os.path.basename(args.img[i_img])).split('_')[0]
            topo_json = "%s%s_topo_coeffs.json" % (args.od,filename_pref)
            with open(topo_json, 'w') as outfile:
                json.dump(topo_coeff_list[i_img],outfile)            


    if args.brdf:
    
      if len(args.img) > 1:     
        # In grouping mode, save arrays for BRDF diagnostic information to ascii files
        np.savetxt("%s%s_brdf_coeffs_r2.csv" % (args.od,args.pref), r_squared_array, header = r2_header, delimiter=',' ,fmt='%s')
        np.savetxt("%s%s_brdf_coeffs_rmse.csv" % (args.od,args.pref), rmse_array, header = r2_header, delimiter=',' ,fmt='%s')
        np.savetxt("%s%s_brdf_coeffs_fit.csv" % (args.od,args.pref), brdf_coeff_array, header = brdf_coeffs_header, delimiter=',' ,fmt='%s')
      
      if total_bin>0:
        for ibin in range(total_bin):            
            if brdf_mask_stat[ibin] < MIN_SAMPLE_COUNT:
              continue
            brdf_json = "%s%s_brdf_coeffs_%s.json" % (args.od,args.pref,str(ibin+1))
            with open(brdf_json, 'w') as outfile:
                json.dump(brdf_coeffs_List[ibin],outfile)
      else:
            brdf_json = "%s%s_brdf_coeffs_1.json" % (args.od,args.pref)
            with open(brdf_json, 'w') as outfile:
                json.dump(brdf_coeffs,outfile)         

                
if __name__== "__main__":
    main()
