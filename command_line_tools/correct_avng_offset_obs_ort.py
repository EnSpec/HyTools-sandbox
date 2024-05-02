
# This script is used to corrected the offsets in SLOPE and ASPECT in the AVIRIS-NG L1 OBS_ORT dataset.
# New_SLOPE = 90 - Old_SLOPE. New result has the range from 0 to 90 degrees
# New_ASPECT = (Old_ASPECT - 90) % 360. New result has the range from 0 to 360 degrees
# Illumination factor COSINE_i will be recalulated using this equation:  cos_i = np.cos(np.radians(slope))*np.cos(np.radians(to_sun_zenith))+np.sin(np.radians(slope))*np.sin(np.radians(to_sun_zenith))*np.cos(np.radians(aspect - to_sun_azimuth))

# Usage: python correct_avng_offset_obs_ort.py -i unccorected_obs_ort_image -o corrected_obs_ort_image
# python correct_avng_offset_obs_ort.py -i  E:/obs_ort_compare/ang20171109t201821_rdn_v2p11_obs_ort   -o   E:/obs_ort_compare/ang20171109t201821_rdn_v2p11_obs_ort_newoffset
# python correct_avng_offset_obs_ort.py -i  E:/obs_ort_compare/new/ang20170803t225710_rdn_obs_ort  -o  E:/obs_ort_compare/new/ang20170803t225710_rdn_obs_ort_newoffset 

import os, sys, argparse
import numpy as np
from shutil import copyfile

# copy from HyTools
dtypeDict = {1:np.uint8,
                   2:np.int16,
                   3:np.int32,
                   4:np.float32,
                   5:np.float64,
                   12:np.uint16,
                   13:np.uint32,
                   14:np.int64,
                   15:np.uint64}
             
# copy from HyTools
def parse_ENVI_header(hdrFile):
    """Parse ENVI header into dictionary
    """

    # Dictionary of all types
    fieldDict = {"acquisition time": "str",
                 "band names":"list_str", 
                 "bands": "int", 
                 "bbl": "list_float",
                 "byte order": "int",
                 "class lookup": "str",
                 "class names": "str",
                 "classes": "int",
                 "cloud cover": "float",
                 "complex function": "str",
                 "coordinate system string": "str",
                 "correction factors": "list_float",
                 "data gain values": "list_float",
                 "data ignore value": "float",
                 "data offset values": "list_float",
                 "data reflectance gain values": "list_float",
                 "data reflectance offset values": "list_float",
                 "data type": "int",
                 "default bands": "list_float",
                 "default stretch": "str",
                 "dem band": "int",
                 "dem file": "str",
                 "description": "str",
                 "envi description":"str",
                 "file type": "str",
                 "fwhm": "list_float",
                 "geo points": "list_float",
                 "header offset": "int",
                 "interleave": "str",
                 "lines": "int",
                 "map info": "list_str",
                 "pixel size": "list_str",
                 "projection info": "str",
                 "read procedures": "str",
                 "reflectance scale factor": "float",
                 "rpc info": "str",
                 "samples":"int",
                 "security tag": "str",
                 "sensor type": "str",
                 "smoothing factors": "list_float",
                 "solar irradiance": "float",
                 "spectra names": "list_str",
                 "sun azimuth": "float",
                 "sun elevation": "float",
                 "wavelength": "list_float",
                 "wavelength units": "str",
                 "x start": "float",
                 "y start": "float",
                 "z plot average": "str",
                 "z plot range": "str",
                 "z plot titles": "str"}

    headerDict = {}

    headerFile = open(hdrFile,'r')
    line = headerFile.readline()
      
    while line :
        if "=" in line:
            key,value = line.rstrip().split("=",1)
            # Add field not in ENVI default list
            if key.strip() not in fieldDict.keys():
                fieldDict[key.strip()] = "str"
            
            valType = fieldDict[key.strip()]
            
            if "{" in value and not "}" in value: 
                while "}" not in line:
                    line = headerFile.readline()
                    value+=line

            if '{}' in value: 
                value = np.nan
            elif valType == "list_float":
                value= np.array([float(x) for x in value.translate(str.maketrans("\n{}","   ")).split(",")])
            elif valType == "list_int":
                value= np.array([int(x) for x in value.translate(str.maketrans("\n{}","   ")).split(",")])
            elif valType == "list_str":
                value= [x.strip() for x in value.translate(str.maketrans("\n{}","   ")).split(",")]
            elif valType == "int":
                value = int(value.translate(str.maketrans("\n{}","   ")))
            elif valType == "float":
                value = float(value.translate(str.maketrans("\n{}","   ")))
            elif valType == "str":
                value = value.translate(str.maketrans("\n{}","   ")).strip().lower()

            headerDict[key.strip()] = value
                            
        line = headerFile.readline()
    
    # Fill unused fields with nans
    for key in fieldDict.keys():
        if key not in headerDict.keys():
            headerDict[key] = np.nan
    
    headerFile.close()
    return headerDict


def copy_hdr(inimg, outimg):
    copyfile(inimg+'.hdr', outimg+'.hdr')

def copy_binary(inimg, outimg):
    copyfile(inimg, outimg)
    
def update_slope_aspect(slope, aspect, mask, flag_both):
        #mask = slope>no_data_value
        if flag_both:
          slope[mask] = 90 -  slope[mask]

        aspect[mask] = ((aspect[mask] - 90) % 360)
        #aspect[aspect > 180] = aspect[aspect > 180] - 360
    
# input angles are in degrees    
def cal_cosine_i(slope, aspect, to_sun_zenith, to_sun_azimuth, mask, no_data_value):
    out_cos_i =  np.cos(np.radians(slope))*np.cos(np.radians(to_sun_zenith))+np.sin(np.radians(slope))*np.sin(np.radians(to_sun_zenith))*np.cos(np.radians(aspect - to_sun_azimuth))
    out_cos_i[~mask] = no_data_value
    return out_cos_i

    
def main(argv):

    parser = argparse.ArgumentParser(description='This code is to correct offsets for slope, aspect, cos_i in older version of AVIRIS-NG obs_ort image')
    parser.add_argument('-i','--infile',type=str, help='Input obs_ort image',required=True)
    parser.add_argument('-o','--outfile',  type=str, help='Output obs_ort image', required=True)
    parser.add_argument('--aspect_only',   help='Do not correct slope, only correct aspect if set true', required=False, action="store_true")

    args = parser.parse_args()


    inimg=args.infile 
    outimg=args.outfile
    if args.aspect_only:
      flag_both=False
    else:
      flag_both=True

    #print(flag_both)
    #return
    copy_hdr(inimg, outimg)
    copy_binary(inimg, outimg)
  
    headerDict = parse_ENVI_header(outimg+'.hdr')
    
    no_data_value = headerDict["data ignore value"]

    if headerDict["interleave"]=='bip' or headerDict["interleave"]=='BIP':
        image_mammap = np.memmap(outimg, dtype = dtypeDict[headerDict["data type"]], mode='r+', shape = (headerDict['lines'],headerDict['samples'], int(headerDict['bands'])), offset=0)        
        to_sun_azimuth = image_mammap[:,:,3]
        to_sun_zenith = image_mammap[:,:,4]
        Slope = np.copy(image_mammap[:,:,6])
        Aspect = np.copy(image_mammap[:,:,7])
        
        mask = Slope>no_data_value
        update_slope_aspect(Slope, Aspect, mask, flag_both)
        cos_i = cal_cosine_i(Slope, Aspect,  to_sun_zenith, to_sun_azimuth, mask, no_data_value)        
        
        if flag_both:
          image_mammap[:,:,6] = Slope

        image_mammap[:,:,7] = Aspect
        image_mammap[:,:,8] = cos_i        

    elif headerDict["interleave"]=='bil' or headerDict["interleave"]=='BIL':
        image_mammap = np.memmap(outimg, dtype = dtypeDict[headerDict["data type"]], mode='r+', shape = (headerDict['lines'], int(headerDict['bands']),headerDict['samples']), offset=0)
        to_sun_azimuth = image_mammap[:,3,:]
        to_sun_zenith = image_mammap[:,4,:]
        Slope = np.copy(image_mammap[:,6,:])
        Aspect = np.copy(image_mammap[:,7,:])
        
        mask = Slope>no_data_value
        update_slope_aspect(Slope, Aspect, mask, flag_both)
        cos_i = cal_cosine_i(Slope, Aspect,  to_sun_zenith, to_sun_azimuth, mask, no_data_value)

        if flag_both:
          image_mammap[:,6,:] = Slope

        image_mammap[:,7,:] = Aspect
        image_mammap[:,8,:] = cos_i
        
    else:
    # 'BSQ'
        image_mammap = np.memmap(outimg, dtype = dtypeDict[headerDict["data type"]], mode='r+', shape = (int(headerDict['bands']),headerDict['lines'], headerDict['samples']), offset=0)    
        to_sun_azimuth = image_mammap[3,:,:]
        to_sun_zenith = image_mammap[4,:,:]
        Slope = np.copy(image_mammap[6,:,:])
        Aspect = np.copy(image_mammap[7,:,:])
        
        mask = Slope>no_data_value
        update_slope_aspect(Slope, Aspect, mask, flag_both)
        cos_i = cal_cosine_i(Slope, Aspect,  to_sun_zenith, to_sun_azimuth, mask, no_data_value)

        if flag_both:        
          image_mammap[6,:,:] = Slope

        image_mammap[7,:,:] = Aspect
        image_mammap[8,:,:] = cos_i
        
    del image_mammap
    
    

if __name__ == "__main__":
    main(sys.argv[1:])