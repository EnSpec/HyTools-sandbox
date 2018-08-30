import argparse,warnings,copy
import numpy as np, os,json
import hytools as ht
from hytools.brdf import *
from hytools.topo_correction import *
from hytools.helpers import *
home = os.path.expanduser("~")

warnings.filterwarnings("ignore")

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

def main():
    '''
    Perform in-memory trait estimation.
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
    parser.add_argument("--mask_threshold", help="Mask threshold value", type = float)
    parser.add_argument("--samp_perc", help="Percent of unmasked pixels to sample", type = float,default=1.0)

    args = parser.parse_args()

    if not args.od.endswith("/"):
        args.od+="/"

    if len(args.img) == 1:
        image = args.img[0]
    
        #Load data objects memory
        if image.endswith(".h5"):
            hyObj = ht.openHDF(image,load_obs = True)
        else:
            hyObj = ht.openENVI(image)
            hyObj.load_obs(args.obs)
        hyObj.create_bad_bands([[300,400],[1330,1430],[1800,1960],[2450,2600]])
        hyObj.load_data()
        
        # Generate mask
        if args.mask:
            ir = hyObj.get_wave(850)
            red = hyObj.get_wave(665)
            ndvi = (ir-red)/(ir+red)
            mask = (ndvi > args.mask_threshold) & (ir != hyObj.no_data)
            hyObj.mask = mask 
            del ir,red,ndvi
        else:
            hyObj.mask = np.ones((hyObj.lines,hyObj.columns)).astype(bool)
            print("Warning no mask specified, results may be unreliable!")

        # Generate cosine i and c1 image for topographic correction
        if args.topo:    
            topo_coeffs = {}
            topo_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist() 
            topo_coeffs['c'] = []
            cos_i = calc_cosine_i(hyObj.solar_zn, hyObj.solar_az, hyObj.azimuth , hyObj.slope)
            c1 = np.cos(hyObj.solar_zn) * np.cos( hyObj.slope)
               
        # Gernerate scattering kernel images for brdf correction
        if args.brdf:
            li,ross =  args.kernels
            # Initialize BRDF dictionary
            brdf_coeffs = {}
            brdf_coeffs['li'] = li
            brdf_coeffs['ross'] = ross
            brdf_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist() 
            brdf_coeffs['fVol'] = []
            brdf_coeffs['fGeo'] = []
            brdf_coeffs['fIso'] = []
            k_vol = generate_volume_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn, ross = ross)
            k_geom = generate_geom_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn,li = li)
    
        # Cycle through the bands and calculate the topographic and BRDF correction coefficients
        print("Calculating image correction coefficients.....")
        iterator = hyObj.iterate(by = 'band')
        
        if args.topo or args.brdf:
            while not iterator.complete:   
                band = iterator.read_next() 
                progbar(iterator.current_band+1, len(hyObj.wavelengths), 100)
                #Skip bad bands
                if hyObj.bad_bands[iterator.current_band]:
                    # Generate topo correction coefficients
                    if args.topo:
                        topo_coeff= generate_topo_coeff_band(band,hyObj.mask,cos_i)
                        topo_coeffs['c'].append(topo_coeff)

                    # Gernerate BRDF correction coefficients
                    if args.brdf:
                        if args.topo:
                            # Apply topo correction to current band
                            correctionFactor = (c1 * topo_coeff)/(cos_i * topo_coeff)
                            band = band* correctionFactor
                            
                        fVol,fGeo,fIso =  generate_brdf_coeff_band(band,hyObj.mask,k_vol,k_geom)
                        brdf_coeffs['fVol'].append(fVol)
                        brdf_coeffs['fGeo'].append(fGeo)
                        brdf_coeffs['fIso'].append(fIso)
            print()

    # Compute topographic and BRDF coefficients using data from multiple scenes
    elif len(args.img) > 1:
        
        hyObj_dict = {}
        sample_dict = {}
        sample_k_vol = []
        sample_k_geom = []
        sample_cos_i = []
        sample_c1 = []
        
        for i,image in enumerate(args.img):
            #Load data objects memory
            if image.endswith(".h5"):
                hyObj = ht.openHDF(image,load_obs = True)
            else:
                hyObj = ht.openENVI(image)
                hyObj.load_obs(args.obs[i])
            hyObj.create_bad_bands([[300,400],[1330,1430],[1800,1960],[2450,2600]])
            hyObj.load_data()
            
            # Generate mask
            if args.mask:
                ir = hyObj.get_wave(850)
                red = hyObj.get_wave(665)
                ndvi = (ir-red)/(ir+red)
                hyObj.mask  = (ndvi > .7) & (ir != hyObj.no_data)
                del ir,red,ndvi
            else:
                hyObj.mask = np.ones((hyObj.lines,hyObj.columns)).astype(bool)
                print("Warning no mask specified, results may be unreliable!")
    
            # Generate sampling mask
            sampleArray = np.zeros(hyObj.mask.shape).astype(bool)
            idx = np.array(np.where(hyObj.mask == True)).T
            idxRand= idx[np.random.choice(range(len(idx)),int(len(idx)*args.samp_perc), replace = False)].T
            sampleArray[idxRand[0],idxRand[1]] = True
            sample_dict[i] = sampleArray    
            
            # Initialize and store band iterator
            hyObj_dict[i] = copy.copy(hyObj).iterate(by = 'band')
            
            # Generate cosine i and c1 image for topographic correction
            if args.topo:   
                # Initialize topographic correction dictionary
                topo_coeffs = {}
                topo_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist() 
                topo_coeffs['c'] = []
                sample_cos_i += calc_cosine_i(hyObj.solar_zn, hyObj.solar_az, hyObj.azimuth , hyObj.slope)[sampleArray].tolist()
                sample_c1 += (np.cos(hyObj.solar_zn) * np.cos( hyObj.slope))[sampleArray].tolist()
            # Gernerate scattering kernel images for brdf correction
            if args.brdf:
                li,ross =  args.kernels
                # Initialize BRDF dictionary
                brdf_coeffs = {}
                brdf_coeffs['li'] = li
                brdf_coeffs['ross'] = ross
                brdf_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist() 
                brdf_coeffs['fVol'] = []
                brdf_coeffs['fGeo'] = []
                brdf_coeffs['fIso'] = []
                sample_k_vol += generate_volume_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn, ross = ross)[sampleArray].tolist()
                sample_k_geom += generate_geom_kernel(hyObj.solar_az,hyObj.solar_zn,hyObj.sensor_az,hyObj.sensor_zn,li = li)[sampleArray].tolist()
        
        sample_k_vol = np.array(sample_k_vol)
        sample_k_geom = np.array(sample_k_geom)
        sample_cos_i = np.array(sample_cos_i)
        sample_c1= np.array(sample_c1)
        
        # Calculate bandwise correction coefficients
        print("Calculating image correction coefficients.....")
        current_progress = 0
        for w,wave in enumerate(hyObj.wavelengths):
            progbar(current_progress, len(hyObj.wavelengths) * len(args.img), 100)
            wave_samples = []
            for i,image in enumerate(args.img):
                wave_samples +=  hyObj_dict[i].read_next()[sample_dict[i]].tolist()
                current_progress+=1
            
            if hyObj.bad_bands[hyObj_dict[i].current_band]:
                wave_samples = np.array(wave_samples)
                # Generate cosine i and c1 image for topographic correction
                if args.topo:    
                    topo_coeff  = generate_topo_coeff_band(wave_samples,[True for x in wave_samples],sample_cos_i)
                    topo_coeffs['c'].append(topo_coeff)
                    correctionFactor = (sample_c1 + topo_coeff)/(sample_cos_i + topo_coeff)
                    wave_samples = wave_samples* correctionFactor
                # Gernerate scattering kernel images for brdf correction
                if args.brdf:
                    fVol,fGeo,fIso = generate_brdf_coeff_band(wave_samples,[True for x in wave_samples],sample_k_vol,sample_k_geom)                  
                    brdf_coeffs['fVol'].append(fVol)
                    brdf_coeffs['fGeo'].append(fGeo)
                    brdf_coeffs['fIso'].append(fIso)
        
    # Export coefficients to JSON
    if args.topo:
        topo_json = "%s%s_topo_coeffs.json" % (args.od,args.pref)
        with open(topo_json, 'w') as outfile:
            json.dump(topo_coeffs,outfile)      
    if args.brdf:
        brdf_json = "%s%s_brdf_coeffs.json" % (args.od,args.pref)
        with open(brdf_json, 'w') as outfile:
            json.dump(brdf_coeffs,outfile)
        
              
if __name__== "__main__":
    main()

#python topo_brdf_coeffs.py --img --pref --brdf --kernels dense thick --mask --mask_threshold .7 --samp_perc 0.1


