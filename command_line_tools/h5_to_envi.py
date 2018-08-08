import argparse,warnings
import numpy as np,os
import hytools as ht
from hytools.file_io import writeENVI,ENVI_header_from_hdf
home = os.path.expanduser("~")
import h5py

warnings.filterwarnings("ignore")

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

def main():

    parser = argparse.ArgumentParser(description = "Convert NEON AOP H5 to ENVI format")
    parser.add_argument("--img", help="Input image pathname",required=True, type = str)
    parser.add_argument("--out", help="Output directory",required=True, type = str)
    parser.add_argument("--obs", help="Ouput observables", required=False, action='store_true')

    args = parser.parse_args()


    if not args.out.endswith("/"):
        args.out+="/"

    #Load data objects memory
    hyObj = ht.openHDF(args.img)
    hyObj.load_data()
    
    iterator = hyObj.iterate(by = 'chunk')
    pixels_processed = 0

    output_name = args.out+ os.path.basename(os.path.splitext(args.img)[0]) 
    writer = writeENVI(output_name, ENVI_header_from_hdf(hyObj))

    while not iterator.complete:   
        chunk = iterator.read_next()  
        pixels_processed += chunk.shape[0]*chunk.shape[1]
        progbar(pixels_processed, hyObj.columns*hyObj.lines, 100)
    
        writer.write_chunk(chunk,iterator.current_line,iterator.current_column)
        if iterator.complete:
            writer.close()

    if args.obs:
        print("Exporting observables....")
        obs_header = ENVI_header_from_hdf(hyObj)
        obs_header['bands'] = 11
        obs_header['band_names'] = ['path length', 'to-sensor azimuth', 'to-sensor zenith','to-sun azimuth',
                                      'to-sun zenith','phase', 'slope', 'aspect', 'cosine i','UTC time']
        obs_header['wavelength units'] = np.nan
        obs_header['data type'] = 4
        writer = writeENVI(output_name + "_obs_ort", obs_header)
       
        hdfObj = h5py.File(args.img,'r')
        base_key = list(hdfObj.keys())[0]
        metadata = hdfObj[base_key]["Reflectance"]["Metadata"]

        writer.write_band(metadata['Ancillary_Imagery']['Path_Length'][:,:],0)
        writer.write_band(metadata['to-sensor_Azimuth_Angle'][:,:],1)
        writer.write_band(metadata['to-sensor_Zenith_Angle'][:,:],2)
        writer.write_band(np.ones((hyObj.lines, hyObj.columns)) * metadata['Logs']['Solar_Azimuth_Angle'].value,3)
        writer.write_band(np.ones((hyObj.lines, hyObj.columns)) * metadata['Logs']['Solar_Zenith_Angle'].value,4)
        writer.write_band(metadata['Ancillary_Imagery']['Slope'][:,:],6)
        writer.write_band(metadata['Ancillary_Imagery']['Aspect'][:,:],7)
        
        writer.close()
        hdfObj.close()

if __name__== "__main__":
    main()



