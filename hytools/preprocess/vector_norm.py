import numpy as np
from ..file_io import *


def vector_normalize_chunk(chunk,scaler):
    """Compute the pixel-wise vector norm of an image chunk.
    
    
    Parameters
    ----------
    chunk : m x n x b np.array
             Image chunck
    scaler: int
            Vector normalization scaling value

    Returns
    -------
    vnorm_chunk : m x n x b np.array
            Vector normalized chunk
    
    """
    
    norm = np.expand_dims(np.linalg.norm(chunk,axis=2),axis=2)            
    vnorm = scaler*chunk/norm
    
    return vnorm

def vector_normalize_img(hyObj,output_name,scaler = 100000):
    """Compute the pixel-wise vector norm of an image.

    Parameters
    ----------
    hyObj : hyTools data object
        Data spectrum.
    output_name: str
        Path name for vector normalized file.

    Returns
    -------
    None
        
    """
    
    if  hyObj.file_type == "ENVI":
        writer = writeENVI(output_name,hyObj.header_dict)
    elif hyObj.file_type == "HDF":
        writer = None
    else:
        print("ERROR: File format not recognized.")

    band_mask = [x==1 for x in hyObj.bad_bands]

    iterator = hyObj.iterate(by = 'chunk')

    while not iterator.complete:
        chunk = iterator.read_next()            
        vnorm = vector_normalize_chunk(chunk[:,:,band_mask],scaler)
        
        # Reassign no_data values            
        vnorm[chunk == hyObj.no_data] = hyObj.no_data 
        writer.write_chunk(vnorm,iterator.current_line,iterator.current_column)

    writer.close()
        
        
        
        
        
        
        