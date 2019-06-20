import numpy as np
import pandas as pd
from ..file_io import *

from ..sampling import extract_point_spec
#import ..sampling.extract_point_spec as eps

#J. W. BoardmanF. A. Kruse (2011). 
#Analysis of imaging spectrometer data using N-dimensional geometry and a mixture-tuned matched filtering approach.
#IEEE Transactions on Geoscience and Remote Sensing, 49(11), 4138–4152.

def create_subset(hyObj, random_subset,  subset_control):
    
    value=np.ones((hyObj.lines, hyObj.columns),dtype=np.byte)
    
    if isinstance(hyObj.mask,np.ndarray):
      value=np.multiply(value,hyObj.mask)
    
    bder_buffer=5
    value_sub = value[bder_buffer:hyObj.lines-bder_buffer,bder_buffer:hyObj.columns-bder_buffer]
    
    if random_subset:
        rBand=np.random.randn(hyObj.lines -2*bder_buffer, hyObj.columns-2*bder_buffer)
        y_ind, x_ind = np.where((rBand > subset_control) & (value_sub > 0))
    else:
        y_ind, x_ind = np.where( value_sub > 0)
    
    y_ind=y_ind+bder_buffer
    x_ind=x_ind+bder_buffer
    
    order_ind = np.where(value[(y_ind+1, x_ind)]*value[(y_ind, x_ind+1)]*value[(y_ind, x_ind)]>0)
    
    return y_ind[order_ind], x_ind[order_ind]


def create_dataframe(subset_y_idx, subset_x_idx):

    return  pd.DataFrame(
        {'img_row': subset_y_idx,
            'img_col': subset_x_idx
        })
    
def pca2d_matrix(indata, scale):
# indata: n samples by m features

    feature_mean=np.mean(indata,axis=0)
    total_pixel_per_band = indata.shape[0]
    data_mean_corrected = indata - feature_mean
    cov_mat =np.einsum('ik,il->kl', data_mean_corrected, data_mean_corrected) /  (total_pixel_per_band-1)

    #cov_mat=np.cov(indata.T)/scale
    
    eigen_val, eigen_vec = np.linalg.eig(cov_mat)
    
    eigen_val[np.where(eigen_val<0)]=1e-20
    
    idx = eigen_val.argsort()[::-1]   
    eigen_val = eigen_val[idx]
    eigen_vec = eigen_vec[:,idx]
    
    
    
    return {"eg_value":eigen_val,
                 "eg_vector":eigen_vec}

def mnft_img(hyObj, n_mnf_components = 40, random_subset=True, subset_control=2.5, eigen_report = True):

    if len(hyObj.bad_bands) > 1:
      ngoodband=np.sum(hyObj.bad_bands)
    else:
      ngoodband=hyObj.bands

    if (n_mnf_components>=ngoodband or n_mnf_components<=0):
        n_mnf_components=ngoodband
            
    if (hyObj.bands>50): # mask : all pixels > 0, buffer from the borders
      if (hyObj.bad_bands[50] == True):
        ibmask=49
      else:
        ibmask=0        
    else:
      if (hyObj.bands>4):
        if ( hyObj.bad_bands[4]==True):
          ibmask=3
        else:
          ibmask=0
      else:
        ibmask=0    
          
    
    subset_y_idx, subset_x_idx = create_subset(hyObj, random_subset,  subset_control)
    
    # EXTRACT point from image using subset_y_idx, subset_x_idx
    point_colrow_df = create_dataframe(subset_y_idx, subset_x_idx)
    spec_data0 = extract_point_spec.extract_from_point(hyObj, point_colrow_df)
    spec_data0 = spec_data0[:,hyObj.bad_bands]
    # M points * nbands
    
    point_colrow_df = create_dataframe(subset_y_idx+1, subset_x_idx)
    spec_data_nb1 = extract_point_spec.extract_from_point(hyObj, point_colrow_df)    
    spec_data_nb1 = spec_data_nb1[:,hyObj.bad_bands]

    point_colrow_df = create_dataframe(subset_y_idx, subset_x_idx+1)
    spec_data_nb2 = extract_point_spec.extract_from_point(hyObj, point_colrow_df)
    spec_data_nb2 = spec_data_nb2[:,hyObj.bad_bands]

    nb_noise=1.0*spec_data0-0.5*spec_data_nb1-0.5*spec_data_nb2
    
    eigen_noise = pca2d_matrix(nb_noise, 1.5)
    
    #print(eigen_noise["eg_value"])
    nw= np.zeros((ngoodband, ngoodband), np.float32)

    np.fill_diagonal(nw, 1.0/np.sqrt(eigen_noise["eg_value"]).astype(np.float32))
    
    matrix_mcnw=np.dot(spec_data0, np.dot(eigen_noise["eg_vector"], nw))
    
    eigen_mcnw = pca2d_matrix(matrix_mcnw, 1.0)
    #print(eigen_mcnw["eg_vector"].shape)
    
    rotmatrix=np.dot(eigen_mcnw["eg_vector"].T,np.dot(nw,eigen_noise["eg_vector"].T))
    
    rotmatrix = rotmatrix[:n_mnf_components, :]
    
    if eigen_report:
        total_ev=sum(eigen_mcnw["eg_value"])
        gain= eigen_mcnw["eg_value"][:n_mnf_components]/total_ev*100
        return {
          "mnf_rotation":rotmatrix,
          "eigen _values": eigen_mcnw["eg_value"],
          "gain": gain
        }
    else:
        return {
          "mnf_rotation":rotmatrix
        }        

def apply_mnft(hyObj, output_name, mnf_coef):

    mnf_arr = np.zeros((hyObj.lines,hyObj.columns,mnf_coef["mnf_rotation"].shape[0]))
    
    out_header_dict = {
    
    "interleave":'bip',
    "coordinate system string":hyObj.header_dict["coordinate system string"],
    "map info":hyObj.header_dict["map info"],
    "data type": 4,
    "bands":mnf_coef["mnf_rotation"].shape[0],
    "lines":hyObj.header_dict["lines"],
    "samples":hyObj.header_dict["samples"]
    
    }

    if  hyObj.file_type == "ENVI":
        writer = writeENVI(output_name,out_header_dict)
    elif hyObj.file_type == "HDF":
        writer = None
    else:
        print("ERROR: File format not recognized.")

    #band_mask = [x==1 for x in hyObj.bad_bands]

    iterator = hyObj.iterate(by = 'chunk')

    while not iterator.complete:
        chunk = iterator.read_next()           
        
        #line_start =iterator.current_line 
        #line_end = iterator.current_line + chunk.shape[0]
        #col_start = iterator.current_column
        #col_end = iterator.current_column + chunk.shape[1]
        
        
        #print(chunk[:,:,hyObj.bad_bands].shape, mnf_coef["mnf_rotation"].shape)   
        mnfimg = np.einsum('jkl,ml->jkm',chunk[:,:,hyObj.bad_bands],mnf_coef["mnf_rotation"])
        #print(mnfimg.shape)
        #mnf_arr[line_start:line_end,col_start:col_end,:] = mnfimg
        # Reassign no_data values            
        #mnfimg[chunk[:,:,hyObj.bad_bands] == hyObj.no_data] = hyObj.no_data 
        writer.write_chunk(mnfimg,iterator.current_line,iterator.current_column)

    writer.close()
    
    return True
    #return mnf_arr