import numpy as np

def vector_normalize(hyObj,output_name):
    """
    Vector normalize spectra.
    
    Calculates pixelwise vector norm excluding bad bands. Output file format
    is the same as the input format.


    Parameters
    ----------
    hyObj : hyTools data object
    output_name: Path name for vector normalized file.

    Returns
    -------
    None
    
    Vector normalized image saved to disk.
    """
    
    if  hyObj.file_type == "ENVI":
        writer = writeENVI(output_name,hyObj.header_dict)
    elif  hyObj.file_type == "HDF":
        writer = None
    else:
        print("ERROR: File format not recognized.")

    bandMask = [x==1 for x in hyObj.bad_bands]

    iterator = hyObj.iterate(by = 'chunk')
    while not iterator.complete:
        chunk = iterator.read_next()            
        norm = np.expand_dims(np.linalg.norm(chunk[:,:,bandMask],axis=2),axis=2)            
        vnorm = 100000*chunk/norm
        # Reassign no_data values            
        vnorm[chunk ==  hyObj.no_data] = hyObj.no_data
        writer.write_chunk(vnorm,iterator.current_line,iterator.current_column)

    writer.close()
        
        
        
        
        
        
        