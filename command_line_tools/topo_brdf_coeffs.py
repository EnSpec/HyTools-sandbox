
import argparse, warnings, copy
import numpy as np, os, json, sys
import pandas as pd
# from pathlib import Path
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist  # ,squareform
import scipy.signal as sig
import osr


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
NO_DATA_VALUE = -9999  # -0.9999  # -9999

# The value that replace NaN in diagnostic tables
DIAGNO_NAN_OUTPUT = -9999

# The C value for image with relatively flat terrain, it will make the TOPO correction factor to be close to 1
FLAT_COEFF_C = -9999

# Data range considered from image data in BRDF coefficient estimation
REFL_MIN_THRESHOLD = 10  # 0.001 # 10
REFL_MAX_THRESHOLD = 9000  # 0.9  # 9000

# NDVI range for image mask
NDVI_MIN_THRESHOLD = 0.01
NDVI_MAX_THRESHOLD = 1.0

# NDVI range for data considered in bins in BRDF coefficient estimation. Pixels outside BIN range will not be corrected for BRDF effect
NDVI_BIN_MIN_THRESHOLD = 0.05
NDVI_BIN_MAX_THRESHOLD = 1.0

# Thresholds for Topographic correction. Pixels beyond those ranges will not be corrected for topographic effect.
# Minimum illumination value (cosine of the incident angle): 0.12 ( < 83.1 degrees)
# Minimum slope: 5 degrees
COSINE_I_MIN_THRESHOLD = 0.12
SLOPE_MIN_THRESHOLD = 0.087
SAMPLE_SLOPE_MIN_THRESHOLD = 0.03

# BRDF coefficients in an NDVI bin with sample size less than this threshold will not be estimated in BRDF correction.
# Its coefficients might be estimated by its neighboring NDVI bins in later steps.
MIN_SAMPLE_COUNT = 100

# if there are too few pixels with terrain (above certian threshold), no TOPO correction factor will be close to 1
MIN_SAMPLE_COUNT_TOPO = 100

# Thresholds for BRDF correction. Pixels beyond this range will not be used for BRDF coefficients estimation.
SENSOR_ZENITH_MIN_DEG = 2

# Wavelengths for NDVI calculation, unit: nanometers
BAND_IR_NM = 850
BAND_RED_NM = 665

# Bad band range. Bands within these ranges will be treated as bad bands, unit: nanometers
# BAD_RANGE =  [[300,400],[1320,1430],[1800,1960],[2450,2600]]  #[[300,400],[1330,1430],[1800,1960],[2450,2600]]
BAD_RANGE = [[300, 400], [1337, 1430], [1800, 1960], [2450, 2600]]

# Bands used for outlier/abnormal flight lines
RGBIM_BAND_CHECK_OUTLIERS = [480, 560, 660, 850, 950, 1050, 1240, 1650, 2217]

# Name field of correction /  smoothing factors. It apprears in the header file of CORR product.
NAME_FIELD_SMOOTH = 'correction factors'

# NDVI range for NDVI-based BRDF coefficients interpolation / regression.
# NDVI bins outside the range will not be used for BRDF coefficients interpolation or regression.
BRDF_VEG_upper_bound = 0.85
BRDF_VEG_lower_bound = 0.25

# Cutting boundary percentiles for dyanmic NDVI Bins
DYN_NDVI_BIN_LOW_PERC = 10
DYN_NDVI_BIN_HIGH_PERC = 90

###########################################


def progbar(curr, total, full_progbar):

    frac = curr / total
    filled_progbar = round(frac * full_progbar)
    print('\r', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), '[{:>7.2%}]'.format(frac), end='')


# Calculate R Squared and rmse for a multiple regression
def cal_r2(y, x1, x2, a):

    nn = y.shape[0]
    if nn < 10:
        return DIAGNO_NAN_OUTPUT, nn, DIAGNO_NAN_OUTPUT

    est_y = a[0] * x1 + a[1] * x2 + a[2]

    avg_y = np.mean(y)

    ss_total = np.sum((y - avg_y)**2)
    ss_res = np.sum((y - est_y)**2)
    r_2 = 1.0 - ss_res / ss_total

    rmse = np.sqrt(ss_res / nn)

    return r_2, nn, rmse


# Calculate R Squared and rmse for a single regression
def cal_r2_single(x, y):

    nn = y.shape[0]

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    est_y = intercept + x * slope

    ss_res = np.sum((y - est_y)**2)

    rmse = np.sqrt(ss_res / nn)

    return slope, intercept, r_value, p_value, std_err, rmse


# convert projected coordinates to LAT/LON
def geoxy_2_latlon(x, y, img_proj):

    img_srs = osr.SpatialReference(wkt=img_proj)

    latlon_wgs84 = osr.SpatialReference()
    latlon_wgs84.ImportFromEPSG(4326)

    # transform coordinates from image to latlon
    imgsrs2latlon = osr.CoordinateTransformation(img_srs, latlon_wgs84)

    lon, lat, z = imgsrs2latlon.TransformPoint(x, y)

    return lon, lat


# get information of the image center
def get_center_info(hyObj, coord_center_method):

    '''
    coord_center_method:
    1: geometric center
    2: mass center
    '''

    geotransform = hyObj.transform
    dim1 = hyObj.lines
    dim2 = hyObj.columns

    if coord_center_method == 1:
        x_center = dim2 // 2
        y_center = dim1 // 2

    elif coord_center_method == 2:
        x_grid = np.arange(0, dim2)
        y_grid = np.arange(0, dim1)
        xv, yv = np.meshgrid(x_grid, y_grid)
        x_center = np.average(xv, weights=hyObj.mask.astype(np.int))
        y_center = np.average(yv, weights=hyObj.mask.astype(np.int))

    solar_zn_center = (hyObj.solar_zn)[int(y_center), int(x_center)]

    center_x = geotransform[1] * (x_center) + geotransform[2] * (y_center) + geotransform[0]
    center_y = geotransform[4] * (x_center) + geotransform[5] * (y_center) + geotransform[3]

    lon_center, lat_center = geoxy_2_latlon(center_x, center_y, hyObj.projection)
    return (lon_center, lat_center, solar_zn_center, solar_zn_center * 180 / np.pi)


# get the unique part of the image file names
def get_uniq_img_name(file_list):

    base_name_list = []

    for file_name in file_list:

        # file_name = Path(file_name.encode().decode(errors="surrogateescape"))
        # file_name = Path(file_name.encode().decode())
        base_name_list = base_name_list + [os.path.basename(file_name)]

    base_name_list1 = base_name_list.copy()

    ll = len(base_name_list[0])
    total = len(base_name_list)
    flag = False
    for cur in range(ll, 0, -1):
        stem = base_name_list[0][:cur]

        for jj in range(1, total):
            if stem in base_name_list[jj][:cur]:
                flag = True
                break
        if flag:
            break

    flag2 = False
    for cur2 in range(ll - 1, 0, -1):
        ext_str = base_name_list[0][cur2:]
        for kk in range(1, total):
            if ext_str != base_name_list[jj][cur2:]:
                flag2 = True
        if flag2:
            break

    for order, item in enumerate(base_name_list):
        base_name_list1[order] = item[:cur + 1]
        base_name_list[order] = item[:cur2 + 1]

    return base_name_list, base_name_list1


# make a circular / disc image for image convolusion in order to make mask buffer
def make_disc_for_buffer(window_radius_size):
    y_grid, x_grid = np.ogrid[-window_radius_size: window_radius_size + 1, -window_radius_size: window_radius_size + 1]
    return (x_grid**2 + y_grid**2 <= window_radius_size**2).astype(np.float)


# search flight line outliers
def kernel_ndvi_outlier_search(band_subset_outlier, sample_k_vol, sample_k_geom, sample_c1, sample_cos_i, sample_slope, sample_ndvi, sample_topo_msk, sample_img_tag, idxRand_dict, hyObj_pointer_dict_list, image_smooth):

    wave_all_samples = np.empty((len(band_subset_outlier), 0), float)
    img_name_list = [x.file_name for x in hyObj_pointer_dict_list]
    group_dict = {}
    group_dict["img_name_list"] = img_name_list
    # print(band_subset_outlier)
    group_dict["band_subset"] = band_subset_outlier

    for i in range(len(hyObj_pointer_dict_list)):
        print(hyObj_pointer_dict_list[i].file_name)
        if hyObj_pointer_dict_list[i].file_type == "ENVI":

            if hyObj_pointer_dict_list[i].interleave == 'bsq':
                spec_data = hyObj_pointer_dict_list[i].data[:, idxRand_dict[i][0], idxRand_dict[i][1]].transpose()
            elif hyObj_pointer_dict_list[i].interleave == 'bil':
                spec_data = hyObj_pointer_dict_list[i].data[idxRand_dict[i][0], :, idxRand_dict[i][1]]
            # hyObj.interleave=='bip':
            else:
                spec_data = hyObj_pointer_dict_list[i].data[idxRand_dict[i][0], idxRand_dict[i][1], :]
        elif hyObj_pointer_dict_list[i].file_type == "HDF":
            spec_data = hyObj_pointer_dict_list[i].data[idxRand_dict[i][0], idxRand_dict[i][1], :]
        else:
            return None

        wave_samples = spec_data[:, band_subset_outlier]
        wave_samples = wave_samples / image_smooth[i][band_subset_outlier]

        sub_index_img_tag = (sample_img_tag == i + 1)
        sample_cos_i_sub = sample_cos_i[sub_index_img_tag]
        sample_slope_sub = sample_slope[sub_index_img_tag]
        sample_c1_sub = sample_c1[sub_index_img_tag]

        topo_mask_sub = (sample_cos_i_sub > COSINE_I_MIN_THRESHOLD) & (sample_slope_sub > SLOPE_MIN_THRESHOLD)

        for iband in range(len(band_subset_outlier)):
            wave_samples_band = wave_samples[:, iband]

            topo_coeff, _, _ = generate_topo_coeff_band(wave_samples_band, (wave_samples_band > REFL_MIN_THRESHOLD) & (wave_samples_band < REFL_MAX_THRESHOLD) & topo_mask_sub, sample_cos_i_sub, non_negative=True)
            correctionFactor = (sample_c1_sub + topo_coeff) / (sample_cos_i_sub + topo_coeff)
            correctionFactor = correctionFactor * topo_mask_sub + 1.0 * (1 - topo_mask_sub)
            wave_samples[:, iband] = wave_samples_band * correctionFactor

        wave_all_samples = np.hstack((wave_all_samples, wave_samples.T))

    ndvi_mask = (sample_ndvi > 0.15) & (sample_ndvi <= 0.95)
    obs_mask = np.isfinite(sample_k_vol) & np.isfinite(sample_k_geom)
    temp_mask = (wave_all_samples[0] > REFL_MIN_THRESHOLD) & (wave_all_samples[0] < REFL_MAX_THRESHOLD) & (obs_mask) & (ndvi_mask)

    for iband in range(len(band_subset_outlier)):
        new_df = pd.DataFrame({'k_geom': sample_k_geom[temp_mask], 'k_vol': sample_k_vol[temp_mask],
                              'reflectance': wave_all_samples[iband, temp_mask], 'line_id': sample_img_tag[temp_mask],
                              "NDVI": sample_ndvi[temp_mask]})

        new_df['ndvi_cut_bins'] = pd.cut(new_df['NDVI'],
                                        bins=[0.15, 0.4, 0.7, 0.95],
                                        labels=['ndvi_1', 'ndvi_2', 'ndvi_3'])

        new_df['geom_cut_bins'] = pd.cut(new_df['k_geom'],
                                bins=np.percentile(sample_k_geom[temp_mask], [5, 33, 67, 95]),  # [5,33,67,95] #[5,25,50,75,95]
                                labels=['k_geom_1', 'k_geom_2', 'k_geom_3'])  # ,'k_geom_4'

        new_df['vol_cut_bins'] = pd.cut(new_df['k_vol'],
                                bins=np.percentile(sample_k_vol[temp_mask], [5, 33, 67, 95]),   # [5,25,50,75,95] # [5,33,67,95]
                                labels=['k_vol_1', 'k_vol_2', 'k_vol_3'])  # 'k_vol_4'

        new_df_bin_group_mean = new_df.groupby(['vol_cut_bins', 'geom_cut_bins', 'ndvi_cut_bins', 'line_id']).median()  # mean()

        new_df_bin_group_mean.reset_index(inplace=True)

        n_bin = new_df_bin_group_mean.shape[0] // len(hyObj_pointer_dict_list)

        ss = new_df_bin_group_mean["reflectance"].values

        bin_avg_array = np.reshape(ss, (n_bin, len(hyObj_pointer_dict_list)))

        bin_mean = np.nanmedian(bin_avg_array, axis=1)
        inds = np.where(np.isnan(bin_avg_array))

        # Place column means in the indices. Align the arrays using take
        bin_avg_array[inds] = np.take(bin_mean, inds[0])

        bin_avg_array = bin_avg_array / bin_mean[:, np.newaxis]

        bin_avg_array = bin_avg_array[~np.isnan(bin_avg_array[:, 0])]

        # Y = pdist(bin_avg_array.T, 'seuclidean', V=None)
        Y = pdist(bin_avg_array.T, 'euclidean', V=None)
        # Y = pdist(bin_avg_array.T, 'canberra')

        print(Y)

        return_dict = {}

        # H_s = hierarchy.single(Y)
        H_s = hierarchy.complete(Y)
        T_ = hierarchy.fcluster(H_s, 1.2, criterion='distance')
        print("Cluster thres 1.2", T_)

        return_dict["Cluster thres 1.2"] = T_.tolist()

        T_ = hierarchy.fcluster(H_s, 1.0, criterion='distance')
        print("Cluster thres 1.0", T_)

        return_dict["Cluster thres 1.0"] = T_.tolist()

        T_ = hierarchy.fcluster(H_s, 0.85, criterion='distance')
        print("Cluster thres 0.85", T_)

        return_dict["Cluster thres 0.9"] = T_.tolist()

        return_dict["distance of metrics"] = Y.tolist()

        major_label_id = np.bincount(np.array(T_)).argmax()

        outlier_img_tag = (np.array(T_) != major_label_id)

        return_dict["outlier_image_bool"] = outlier_img_tag.astype(int).tolist()
        return_dict["outlier_count"] = int(np.count_nonzero(outlier_img_tag))
        group_dict['b' + str(iband + 1)] = return_dict

    return group_dict


def main():
    '''
    Generate topographic and BRDF correction coefficients. Corrections can be calculated on individual images
    or groups of images.
    '''
    parser = argparse.ArgumentParser(description="In memory trait mapping tool.")
    parser.add_argument("--img", help="Input image/directory pathname", required=True, nargs='*', type=str)
    parser.add_argument("--obs", help="Input observables pathname", required=False, nargs='*', type=str)
    parser.add_argument("--od", help="Ouput directory", required=True, type=str)
    parser.add_argument("--pref", help="Coefficient filename prefix", required=True, type=str)
    parser.add_argument("--brdf", help="Perform BRDF correction", action='store_true')
    parser.add_argument("--kernels", help="Li and Ross kernel types", nargs=2, type=str)
    parser.add_argument("--topo", help="Perform topographic correction", action='store_true')
    parser.add_argument("--mask", help="Image mask type to use", action='store_true')
    parser.add_argument("--mask_threshold", help="Mask threshold value", nargs='*', type=float)
    parser.add_argument("--samp_perc", help="Percent of unmasked pixels to sample", type=float, default=1.0)
    parser.add_argument("--agmask", help="ag / urban mask file", required=False, type=str)
    parser.add_argument("--topo_sep", help="In multiple image mode, perform topographic correction in a image-based fasion", action='store_true')
    parser.add_argument("--mass_center", help="Use mass center to be the center coordinate, default is geometric center. It is only used in BRDF correction", action='store_true')
    parser.add_argument("--check_flight", help="Check abnormal flight lines if group mode BRDF correction is performed", action='store_true')

    parser.add_argument("--dynamicbin", help="Total Number of dynamic NDVI bins, with higher priority than mask_threshold", type=int, required=False)
    parser.add_argument("--buffer_neon", help="neon buffer", action='store_true')

    args = parser.parse_args()

    if not args.od.endswith("/"):
        args.od += "/"

    if len(args.img) == 1:
        image = args.img[0]

        # Load data objects memory
        if image.endswith(".h5"):
            hyObj = ht.openHDF(image, load_obs=True)
            smoothing_factor = np.ones(hyObj.bands)
        else:
            hyObj = ht.openENVI(image)
            hyObj.load_obs(args.obs[0])

            smoothing_factor = hyObj.header_dict[NAME_FIELD_SMOOTH]
            # CORR product has smoothing factor, and all bands are converted back to uncorrected / unsmoothed version by dividing the corr/smooth factors
            if isinstance(smoothing_factor, (list, tuple, np.ndarray)):
                smoothing_factor = np.array(smoothing_factor)
            # REFL version
            else:
                smoothing_factor = np.ones(hyObj.bands)

        hyObj.create_bad_bands(BAD_RANGE)

        # no data  / ignored values varies by product
        # hyObj.no_data = NO_DATA_VALUE

        hyObj.load_data()

        # Generate mask
        if args.mask:
            ir = hyObj.get_wave(BAND_IR_NM)
            red = hyObj.get_wave(BAND_RED_NM)
            ndvi = (1.0 * ir - red) / (1.0 * ir + red)

            ag_mask = np.array([0])
            if args.agmask:
                ag_mask = np.fromfile(args.agmask, dtype=np.uint8).reshape((hyObj.lines, hyObj.columns))

            if args.buffer_neon:
                buffer_edge = sig.convolve2d(ir <= 0.5 * hyObj.no_data, make_disc_for_buffer(30), mode='same', fillvalue=1)
                ag_mask = ag_mask or (buffer_edge > 0)

            hyObj.mask = (ndvi > NDVI_MIN_THRESHOLD) & (ndvi < NDVI_MAX_THRESHOLD) & (ir != hyObj.no_data) & (ag_mask == 0)

            del ir, red  # ,ndvi
        else:
            hyObj.mask = np.ones((hyObj.lines, hyObj.columns)).astype(bool)
            print("Warning no mask specified, results may be unreliable!")

        # Generate cosine i and c1 image for topographic correction

        if args.topo:

            topo_coeffs = {}
            topo_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist()
            topo_coeffs['c'] = []
            topo_coeffs['slope'] = []
            topo_coeffs['intercept'] = []
            cos_i = calc_cosine_i(hyObj.solar_zn, hyObj.solar_az, hyObj.aspect, hyObj.slope)
            c1 = np.cos(hyObj.solar_zn)
            c2 = np.cos(hyObj.slope)

            terrain_msk = (cos_i > COSINE_I_MIN_THRESHOLD) & (hyObj.slope > SLOPE_MIN_THRESHOLD)
            topomask = hyObj.mask  # & (cos_i > 0.12)  & (hyObj.slope > 0.087)

        # Generate scattering kernel images for brdf correction
        if args.brdf:

            if args.dynamicbin:
                total_bin = args.dynamicbin
                perc_range = DYN_NDVI_BIN_HIGH_PERC - DYN_NDVI_BIN_LOW_PERC + 1
                ndvi_break_dyn_bin = np.percentile(ndvi[ndvi > 0], np.arange(DYN_NDVI_BIN_LOW_PERC, DYN_NDVI_BIN_HIGH_PERC + 1, perc_range // (total_bin - 2)))
                ndvi_thres = sorted([NDVI_BIN_MIN_THRESHOLD] + ndvi_break_dyn_bin.tolist() + [NDVI_BIN_MAX_THRESHOLD])

            else:
                if args.mask_threshold:
                    ndvi_thres = [NDVI_BIN_MIN_THRESHOLD] + args.mask_threshold + [NDVI_BIN_MAX_THRESHOLD]
                else:
                    ndvi_thres = [NDVI_BIN_MIN_THRESHOLD, NDVI_BIN_MAX_THRESHOLD]

            ndvi_thres = sorted(list(set(ndvi_thres)))
            total_bin = len(ndvi_thres) - 1
            brdfmask = np.ones((total_bin, hyObj.lines, hyObj.columns)).astype(bool)

            for ibin in range(total_bin):
                brdfmask[ibin, :, :] = hyObj.mask & (ndvi > ndvi_thres[ibin]) & (ndvi <= ndvi_thres[ibin + 1]) & (hyObj.sensor_zn > np.radians(SENSOR_ZENITH_MIN_DEG))

            li, ross = args.kernels
            # Initialize BRDF dictionary

            brdf_coeffs_List = []  # initialize
            brdf_mask_stat = np.zeros(total_bin)

            for ibin in range(total_bin):
                brdf_mask_stat[ibin] = np.count_nonzero(brdfmask[ibin, :, :])

                brdf_coeffs = {}
                brdf_coeffs['li'] = li
                brdf_coeffs['ross'] = ross
                brdf_coeffs['ndvi_lower_bound'] = ndvi_thres[ibin]
                brdf_coeffs['ndvi_upper_bound'] = ndvi_thres[ibin + 1]
                brdf_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist()
                brdf_coeffs['fVol'] = []
                brdf_coeffs['fGeo'] = []
                brdf_coeffs['fIso'] = []
                brdf_coeffs_List.append(brdf_coeffs)

            k_vol = generate_volume_kernel(hyObj.solar_az, hyObj.solar_zn, hyObj.sensor_az, hyObj.sensor_zn, ross=ross)
            k_geom = generate_geom_kernel(hyObj.solar_az, hyObj.solar_zn, hyObj.sensor_az, hyObj.sensor_zn, li=li)
            k_finite = np.isfinite(k_vol) & np.isfinite(k_geom)

            if args.mass_center:
                coord_center_method = 2
            else:
                coord_center_method = 1

            img_center_info = get_center_info(hyObj, coord_center_method)
            print(img_center_info)
            # csv_img_center_info = np.array([os.path.basename(args.img[0]).split('_')[0]]+list(img_center_info))[:,np.newaxis]
            csv_img_center_info = np.array([os.path.basename(args.img[0])] + [args.pref] * 2 + list(img_center_info))[:, np.newaxis]
            np.savetxt("%s%s_lat_sza.csv" % (args.od, args.pref), csv_img_center_info.T, header="image_name,uniq_name,uniq_name_short,LON,LAT,solar_zn_rad,ssolar_zn_deg", delimiter=',', fmt='%s', comments='')

        # Cycle through the bands and calculate the topographic and BRDF correction coefficients
        print("Calculating image correction coefficients.....")
        iterator = hyObj.iterate(by='band')

        if args.topo or args.brdf:
            while not iterator.complete:

                if hyObj.bad_bands[iterator.current_band + 1]:  # load data to RAM, if it is a goog band
                    band = iterator.read_next()
                    band = band / smoothing_factor[iterator.current_band]
                    band_msk = (band > REFL_MIN_THRESHOLD) & (band < REFL_MAX_THRESHOLD)

                else:  # similar to .read_next(), but do not load data to RAM, if it is a bad band
                    iterator.current_band += 1
                    if iterator.current_band == hyObj.bands - 1:
                        iterator.complete = True

                progbar(iterator.current_band + 1, len(hyObj.wavelengths), 100)
                # Skip bad bands
                if hyObj.bad_bands[iterator.current_band]:
                    # Generate topo correction coefficients
                    if args.topo:

                        topomask_b = topomask & band_msk

                        if np.count_nonzero(topomask_b & terrain_msk) > MIN_SAMPLE_COUNT_TOPO:
                            topo_coeff, reg_slope, reg_intercept = generate_topo_coeff_band(band, topomask_b & terrain_msk, cos_i, non_negative=True)
                            # topo_coeff, reg_slope, reg_intercept= generate_topo_coeff_band(band,topomask_b & terrain_msk,cos_i)
                        else:
                            topo_coeff = FLAT_COEFF_C
                            reg_slope, reg_intercept = (-9999, -9999)
                            print("fill with FLAT_COEFF_C")

                        topo_coeffs['c'].append(topo_coeff)
                        topo_coeffs['slope'].append(reg_slope)
                        topo_coeffs['intercept'].append(reg_intercept)

                    # Gernerate BRDF correction coefficients
                    if args.brdf:
                        if args.topo:
                            # Apply topo correction to current bands
                            correctionFactor = (c2 * c1 + topo_coeff) / (cos_i + topo_coeff)
                            correctionFactor = correctionFactor * topomask_b + 1.0 * (1 - topomask_b)  # only apply to orographic area
                            band = band * correctionFactor

                        for ibin in range(total_bin):

                            if brdf_mask_stat[ibin] < MIN_SAMPLE_COUNT:
                                continue

                            band_msk_new = (band > REFL_MIN_THRESHOLD) & (band < REFL_MAX_THRESHOLD)

                            if np.count_nonzero(brdfmask[ibin, :, :] & band_msk & k_finite & band_msk_new) < MIN_SAMPLE_COUNT:
                                brdf_mask_stat[ibin] = DIAGNO_NAN_OUTPUT
                                continue

                            fVol, fGeo, fIso = generate_brdf_coeff_band(band, brdfmask[ibin, :, :] & band_msk & k_finite & band_msk_new, k_vol, k_geom)
                            brdf_coeffs_List[ibin]['fVol'].append(fVol)
                            brdf_coeffs_List[ibin]['fGeo'].append(fGeo)
                            brdf_coeffs_List[ibin]['fIso'].append(fIso)

    # Compute topographic and BRDF coefficients using data from multiple scenes
    elif len(args.img) > 1:

        image_uniq_name_list, image_uniq_name_list_short = get_uniq_img_name(args.img)

        if args.check_flight:
            args.brdf = True
            args.topo = True
            args.topo_sep = True
            print("Automatically enable TOPO and BRDF mode if 'check_flight' is enabled. ")

        if args.brdf:

            li, ross = args.kernels

            ndvi_thres_complete = False
            if args.dynamicbin:
                total_bin = args.dynamicbin
                perc_range = DYN_NDVI_BIN_HIGH_PERC - DYN_NDVI_BIN_LOW_PERC + 1
                # ndvi_break_dyn_bin= np.percentile(ndvi[ndvi>0], np.arange(DYN_NDVI_BIN_LOW_PERC,DYN_NDVI_BIN_HIGH_PERC+1,perc_range//(total_bin-2)))
                ndvi_thres = [NDVI_BIN_MIN_THRESHOLD] + [None] * (total_bin - 1) + [NDVI_BIN_MAX_THRESHOLD]
                print("NDVI bins:", ndvi_thres)
            else:
                ndvi_thres_complete = True
                if args.mask_threshold:
                    ndvi_thres = sorted([NDVI_BIN_MIN_THRESHOLD] + args.mask_threshold + [NDVI_BIN_MAX_THRESHOLD])
                    total_bin = len(args.mask_threshold) + 1
                else:
                    ndvi_thres = [NDVI_BIN_MIN_THRESHOLD, NDVI_BIN_MAX_THRESHOLD]
                    total_bin = 1

            brdf_coeffs_List = []  # initialize
            brdf_mask_stat = np.zeros(total_bin)

            for ibin in range(total_bin):
                brdf_coeffs = {}
                brdf_coeffs['li'] = li
                brdf_coeffs['ross'] = ross
                brdf_coeffs['ndvi_lower_bound'] = None  # ndvi_thres[ibin]
                brdf_coeffs['ndvi_upper_bound'] = None  # ndvi_thres[ibin+1]
                # brdf_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist()
                brdf_coeffs['fVol'] = []
                brdf_coeffs['fGeo'] = []
                brdf_coeffs['fIso'] = []
                brdf_coeffs['flight_box_avg_sza'] = []
                brdf_coeffs_List.append(brdf_coeffs)

            if args.mass_center:
                coord_center_method = 2
            else:
                coord_center_method = 1
            csv_group_center_info = np.empty((7, 0))

        hyObj_dict = {}
        sample_dict = {}
        idxRand_dict = {}
        sample_k_vol = []
        sample_k_geom = []
        sample_cos_i = []
        sample_c1 = []
        sample_slope = []
        sample_ndvi = []
        sample_index = [0]
        sample_img_tag = []  # record which image that sample is drawn from
        sub_total_sample_size = 0
        ndvi_mask_dict = {}
        image_smooth = []
        band_subset_outlier = []
        hyObj_pointer_dict_list = []

        for i, image in enumerate(args.img):
            # Load data objects memory
            if image.endswith(".h5"):
                hyObj = ht.openHDF(image, load_obs=True)
                smoothing_factor = np.ones(hyObj.bands)
                image_smooth += [smoothing_factor]
            else:
                hyObj = ht.openENVI(image)
                hyObj.load_obs(args.obs[i])

                smoothing_factor = hyObj.header_dict[NAME_FIELD_SMOOTH]
                # CORR product has smoothing factor, and all bands are converted back to uncorrected / unsmoothed version by dividing the corr/smooth factors
                if isinstance(smoothing_factor, (list, tuple, np.ndarray)):
                    smoothing_factor = np.array(smoothing_factor)
                # REFL version
                else:
                    smoothing_factor = np.ones(hyObj.bands)
                image_smooth += [smoothing_factor]

            hyObj.create_bad_bands(BAD_RANGE)
            hyObj.no_data = NO_DATA_VALUE
            hyObj.load_data()
            hyObj_pointer_dict_list = hyObj_pointer_dict_list + [copy.copy(hyObj)]

            if args.brdf:
                img_center_info = get_center_info(hyObj, coord_center_method)
                print(img_center_info)
                csv_img_center_info = np.array([os.path.basename(args.img[i])] + [image_uniq_name_list[i]] + [image_uniq_name_list_short[i]] + list(img_center_info))[:, np.newaxis]
                csv_group_center_info = np.hstack((csv_group_center_info, csv_img_center_info))
            # continues

            # Generate mask
            if args.mask:
                ir = hyObj.get_wave(BAND_IR_NM)
                red = hyObj.get_wave(BAND_RED_NM)
                ndvi = (1.0 * ir - red) / (1.0 * ir + red)

                ag_mask = 0
                if args.buffer_neon:
                    buffer_edge = sig.convolve2d(ir <= 0.5 * hyObj.no_data, make_disc_for_buffer(30), mode='same', fillvalue=1)
                    ag_mask = ag_mask or (buffer_edge > 0)

                hyObj.mask = (ndvi > NDVI_MIN_THRESHOLD) & (ndvi <= NDVI_MAX_THRESHOLD) & (ir != hyObj.no_data) & (ag_mask == 0)
                hyObj.mask.tofile(args.od + '/' + args.pref + str(i) + '_msk.bin')
                del ir, red  # ,ndvi
            else:
                hyObj.mask = np.ones((hyObj.lines, hyObj.columns)).astype(bool)
                print("Warning no mask specified, results may be unreliable!")

            # Generate sampling mask
            sampleArray = np.zeros(hyObj.mask.shape).astype(bool)
            idx = np.array(np.where(hyObj.mask == True)).T

            # np.random.seed(0)  # just for test

            idxRand = idx[np.random.choice(range(len(idx)), int(len(idx) * args.samp_perc), replace=False)].T  # actually used
            sampleArray[idxRand[0], idxRand[1]] = True
            sample_dict[i] = sampleArray
            idxRand_dict[i] = idxRand

            print(idxRand.shape)
            sub_total_sample_size += idxRand.shape[1]
            sample_index = sample_index + [sub_total_sample_size]

            # Initialize and store band iterator
            hyObj_dict[i] = copy.copy(hyObj).iterate(by='band')

            # Generate cosine i and slope samples
            sample_cos_i += calc_cosine_i(hyObj.solar_zn, hyObj.solar_az, hyObj.aspect, hyObj.slope)[sampleArray].tolist()
            sample_slope += (hyObj.slope)[sampleArray].tolist()

            # Generate c1 samples for topographic correction
            if args.topo:
                # Initialize topographic correction dictionary
                topo_coeffs = {}
                topo_coeffs['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist()
                topo_coeffs['c'] = []
                topo_coeffs['slope'] = []
                topo_coeffs['intercept'] = []
                sample_c1 += (np.cos(hyObj.solar_zn) * np.cos(hyObj.slope))[sampleArray].tolist()

            # Gernerate scattering kernel samples for brdf correction
            if args.brdf or args.check_flight:

                sample_ndvi += (ndvi)[sampleArray].tolist()
                sample_img_tag += [i + 1] * idxRand.shape[1]  # start from 1

                for ibin in range(total_bin):
                    brdf_coeffs_List[ibin]['wavelengths'] = hyObj.wavelengths[hyObj.bad_bands].tolist()

                sample_k_vol += generate_volume_kernel(hyObj.solar_az, hyObj.solar_zn, hyObj.sensor_az, hyObj.sensor_zn, ross=ross)[sampleArray].tolist()
                sample_k_geom += generate_geom_kernel(hyObj.solar_az, hyObj.solar_zn, hyObj.sensor_az, hyObj.sensor_zn, li=li)[sampleArray].tolist()

            # del ndvi, topomask, brdfmask
            if args.mask:
                del ndvi

            # find outliers, initialize band_subset_outlier
            if args.check_flight and i == 0:
                for wave_i in RGBIM_BAND_CHECK_OUTLIERS:
                    band_num = int(hyObj.wave_to_band(wave_i))
                    band_subset_outlier = band_subset_outlier + [band_num]  # zero based

        sample_k_vol = np.array(sample_k_vol)
        sample_k_geom = np.array(sample_k_geom)
        sample_cos_i = np.array(sample_cos_i)
        sample_c1 = np.array(sample_c1)
        sample_slope = np.array(sample_slope)
        sample_ndvi = np.array(sample_ndvi)
        sample_img_tag = np.array(sample_img_tag)

        sample_topo_msk = (sample_cos_i > COSINE_I_MIN_THRESHOLD) & (sample_slope > SLOPE_MIN_THRESHOLD)

        if args.brdf or args.check_flight:
            group_summary_info = [args.pref] * 3 + np.mean(csv_group_center_info[3:, :].astype(np.float), axis=1).tolist()
            csv_group_center_info = np.insert(csv_group_center_info.T, 0, group_summary_info, axis=0)
            np.savetxt("%s%s_lat_sza.csv" % (args.od, args.pref), csv_group_center_info, header="image_name,uniq_name,uniq_name_short,LON,LAT,solar_zn_rad,solar_zn_deg", delimiter=',', fmt='%s', comments='')

            if not ndvi_thres_complete:
                ndvi_break_dyn_bin = np.percentile(sample_ndvi[sample_ndvi > 0], np.arange(DYN_NDVI_BIN_LOW_PERC, DYN_NDVI_BIN_HIGH_PERC + 1, perc_range // (total_bin - 2)))

                ndvi_thres = sorted([NDVI_BIN_MIN_THRESHOLD] + ndvi_break_dyn_bin.tolist() + [NDVI_BIN_MAX_THRESHOLD])
                ndvi_thres = sorted(list(set(ndvi_thres)))  # remove duplicates
                total_bin = len(ndvi_thres) - 1

            for ibin in range(total_bin):
                brdf_coeffs_List[ibin]['flight_box_avg_sza'] = csv_group_center_info[0, -1]
                # print('last',csv_group_center_info[0,-1])

                brdf_coeffs_List[ibin]['ndvi_lower_bound'] = ndvi_thres[ibin]
                brdf_coeffs_List[ibin]['ndvi_upper_bound'] = ndvi_thres[ibin + 1]

        if args.topo_sep:
            topo_coeff_list = []
            for _ in range(len(args.img)):
                topo_coeff_list += [{'c': [], 'slope':[], 'intercept':[], 'wavelengths':hyObj.wavelengths[hyObj.bad_bands].tolist()}]

        # initialize BRDF coeffs
        if args.brdf:
            for ibin in range(total_bin):
                ndvi_mask = (sample_ndvi > brdf_coeffs_List[ibin]['ndvi_lower_bound']) & (sample_ndvi <= brdf_coeffs_List[ibin]['ndvi_upper_bound'])
                ndvi_mask_dict[ibin] = ndvi_mask
                count_ndvi = np.count_nonzero(ndvi_mask)
                brdf_mask_stat[ibin] = count_ndvi

            # initialize arrays for BRDF coefficient estimation diagnostic files
            total_image = len(args.img)

            # r-squared array
            r_squared_array = np.ndarray((total_bin * (total_image + 1), 3 + len(hyObj.wavelengths)), dtype=object)  # total + flightline by flightline
            r_squared_array[:] = 0

            r_squared_array[:, 0] = (0.5 * (np.array(ndvi_thres[:-1]) + np.array(ndvi_thres[1:]))).tolist() * (total_image + 1)
            r_squared_array[:total_bin, 1] = 'group'
            r_squared_array[total_bin:, 1] = np.repeat(np.array([os.path.basename(x) for x in args.img]), total_bin, axis=0)

            r2_header = 'NDVI_Bin_Center,Flightline,Sample_Size,' + ','.join('B' + str(wavename) for wavename in hyObj.wavelengths)

            # RMSE array
            rmse_array = np.copy(r_squared_array)

            # BRDF coefficient array, volumetric+geometric+isotopic
            brdf_coeff_array = np.ndarray((total_bin + 8, 3 + 3 * len(hyObj.wavelengths)), dtype=object)
            brdf_coeff_array[:] = 0
            brdf_coeff_array[:total_bin, 0] = 0.5 * (np.array(ndvi_thres[:-1]) + np.array(ndvi_thres[1:]))

            brdf_coeffs_header = 'NDVI_Bin_Center,Sample_Size,' + ','.join('B' + str(wavename) + '_vol' for wavename in hyObj.wavelengths) + ',' + ','.join('B' + str(wavename) + '_geo' for wavename in hyObj.wavelengths) + ',' + ','.join('B' + str(wavename) + '_iso' for wavename in hyObj.wavelengths)
            brdf_coeff_array[total_bin + 1:total_bin + 7, 0] = ['slope', 'intercept', 'r_value', 'p_value', 'std_err', 'rmse']

        # find outliers
        img_tag_used = (sample_img_tag > -1)  # All True, will be changed if there is any abnormal lines.
        if args.check_flight:
            print("Searching for fight line outliers.....")

            outlier_dict = kernel_ndvi_outlier_search(band_subset_outlier, sample_k_vol, sample_k_geom, sample_c1, sample_cos_i, sample_slope, sample_ndvi, sample_topo_msk, sample_img_tag, idxRand_dict, hyObj_pointer_dict_list, image_smooth)
            outlier_json = "%s%s_outliers.json" % (args.od, args.pref)
            with open(outlier_json, 'w') as outfile:
                json.dump(outlier_dict, outfile)

            if outlier_dict["b1"]["outlier_count"] > 0:
                outlier_image_bool = np.array(outlier_dict["b1"]["outlier_image_bool"]).astype(bool)
                img_tag_used = ~outlier_image_bool[np.arange(len(args.img))[sample_img_tag - 1]]
                print(np.unique(sample_img_tag[img_tag_used]))

                if outlier_dict["b1"]["outlier_count"] > len(args.img) / 2:
                    print("More than half of the lines are abnormal lines, please check the information in {}{}_outliers.json. ".format(args.od, args.pref))
                    print("Flight line outliers checking Finishes.")
                    return  # exit the script, halt the procedure of coefficients estimation.

            print("Flight line outliers checking finishes.")

        else:
            print("Use all fight lines.....")
            
            # wave9_samples = np.empty((9,0),float)
            # if
            # singleband_kernel_ndvi_outlier_search(wave_samples, args.od,args.pref)

        # Calculate bandwise correction coefficients
        print("Calculating image correction coefficients.....")
        current_progress = 0

        for w, wave in enumerate(hyObj.wavelengths):
            progbar(current_progress, len(hyObj.wavelengths) * len(args.img), 100)
            wave_samples = []
            for i, image in enumerate(args.img):

                if hyObj.bad_bands[hyObj_dict[i].current_band + 1]:  # load data to RAM, if it is a goog band
                    wave_samples += hyObj_dict[i].read_next()[sample_dict[i]].tolist()

                else:  # similar to .read_next(), but do not load data to RAM, if it is a bad band
                    hyObj_dict[i].current_band += 1
                    if hyObj_dict[i].current_band == hyObj_dict[i].bands - 1:
                        hyObj_dict[i].complete = True

                current_progress += 1

            if hyObj.bad_bands[hyObj_dict[i].current_band]:

                wave_samples = np.array(wave_samples)

                for i_img_tag in range(len(args.img)):
                    img_tag_true = sample_img_tag == i_img_tag + 1
                    wave_samples[img_tag_true] = wave_samples[img_tag_true] / image_smooth[i_img_tag][w]

                # Generate cosine i and c1 image for topographic correction
                if args.topo:
                    if not args.topo_sep:
                        topo_coeff, coeff_slope, coeff_intercept = generate_topo_coeff_band(wave_samples, (wave_samples > REFL_MIN_THRESHOLD) & (wave_samples < REFL_MAX_THRESHOLD) & sample_topo_msk, sample_cos_i, non_negative=True)
                        topo_coeffs['c'].append(topo_coeff)
                        topo_coeffs['slope'].append(coeff_slope)
                        topo_coeffs['intercept'].append(coeff_intercept)
                        correctionFactor = (sample_c1 + topo_coeff) / (sample_cos_i + topo_coeff)
                        correctionFactor = correctionFactor * sample_topo_msk + 1.0 * (1 - sample_topo_msk)
                        wave_samples = wave_samples * correctionFactor
                    else:
                        for i in range(len(args.img)):
                            wave_samples_sub = wave_samples[sample_index[i]:sample_index[i + 1]]
                            sample_cos_i_sub = sample_cos_i[sample_index[i]:sample_index[i + 1]]
                            # sample_slope_sub = sample_slope[sample_index[i]:sample_index[i + 1]]
                            sample_c1_sub = sample_c1[sample_index[i]:sample_index[i + 1]]

                            sample_topo_msk_sub = sample_topo_msk[sample_index[i]:sample_index[i + 1]]

                            if np.count_nonzero(sample_topo_msk_sub) > MIN_SAMPLE_COUNT_TOPO:
                                topo_coeff, coeff_slope, coeff_intercept = generate_topo_coeff_band(wave_samples_sub, (wave_samples_sub > REFL_MIN_THRESHOLD) & (wave_samples_sub < REFL_MAX_THRESHOLD) & sample_topo_msk_sub, sample_cos_i_sub, non_negative=True)
                            else:
                                topo_coeff = FLAT_COEFF_C

                            topo_coeff_list[i]['c'].append(topo_coeff)
                            topo_coeff_list[i]['slope'].append(coeff_slope)
                            topo_coeff_list[i]['intercept'].append(coeff_intercept)

                            correctionFactor = (sample_c1_sub + topo_coeff) / (sample_cos_i_sub + topo_coeff)
                            correctionFactor = correctionFactor * sample_topo_msk_sub + 1.0 * (1 - sample_topo_msk_sub)
                            wave_samples[sample_index[i]:sample_index[i + 1]] = wave_samples_sub * correctionFactor

                # Gernerate scattering kernel images for brdf correction
                if args.brdf:

                    wave_samples = wave_samples[img_tag_used]
                    temp_mask = (wave_samples > REFL_MIN_THRESHOLD) & (wave_samples < REFL_MAX_THRESHOLD) & np.isfinite(sample_k_vol[img_tag_used]) & np.isfinite(sample_k_geom[img_tag_used])
                    temp_mask = temp_mask & (sample_cos_i[img_tag_used] > COSINE_I_MIN_THRESHOLD) & (sample_slope[img_tag_used] > SAMPLE_SLOPE_MIN_THRESHOLD)

                    for ibin in range(total_bin):

                        # skip BINs that has not enough samples in diagnostic output
                        if brdf_mask_stat[ibin] < MIN_SAMPLE_COUNT or np.count_nonzero(temp_mask) < MIN_SAMPLE_COUNT:
                            r_squared_array[range(ibin, total_bin * (total_image + 1), total_bin), w + 3] = DIAGNO_NAN_OUTPUT
                            rmse_array[range(ibin, total_bin * (total_image + 1), total_bin), w + 3] = DIAGNO_NAN_OUTPUT
                            brdf_mask_stat[ibin] = brdf_mask_stat[ibin] + DIAGNO_NAN_OUTPUT
                            continue

                        if np.count_nonzero(temp_mask & ndvi_mask_dict[ibin][img_tag_used]) < MIN_SAMPLE_COUNT:
                            fVol, fGeo, fIso = (0, 0, 1)
                        else:
                            fVol, fGeo, fIso = generate_brdf_coeff_band(wave_samples, temp_mask & ndvi_mask_dict[ibin][img_tag_used], sample_k_vol[img_tag_used], sample_k_geom[img_tag_used])

                        mask_sub = temp_mask & ndvi_mask_dict[ibin][img_tag_used]
                        r_squared_array[ibin, 2] = wave_samples[mask_sub].shape[0]
                        est_r2, sample_nn, rmse_total = cal_r2(wave_samples[mask_sub], sample_k_vol[img_tag_used][mask_sub], sample_k_geom[img_tag_used][mask_sub], [fVol, fGeo, fIso])
                        r_squared_array[ibin, w + 3] = est_r2
                        rmse_array[ibin, w + 3] = rmse_total
                        rmse_array[ibin, 2] = r_squared_array[ibin, 2]

                        brdf_coeff_array[ibin, 1] = wave_samples[mask_sub].shape[0]

                        # update diagnostic information scene by scene
                        for img_order in range(total_image):

                            img_mask_sub = (sample_img_tag[img_tag_used] == (img_order + 1)) & mask_sub

                            est_r2, sample_nn, rmse_bin = cal_r2(wave_samples[img_mask_sub], sample_k_vol[img_tag_used][img_mask_sub], sample_k_geom[img_tag_used][img_mask_sub], [fVol, fGeo, fIso])
                            r_squared_array[ibin + (img_order + 1) * total_bin, w + 3] = est_r2
                            r_squared_array[ibin + (img_order + 1) * total_bin, 2] = max(sample_nn, int(r_squared_array[ibin + (img_order + 1) * total_bin, 2]))  # update many times

                            rmse_array[ibin + (img_order + 1) * total_bin, w + 3] = rmse_bin
                            rmse_array[ibin + (img_order + 1) * total_bin, 2] = r_squared_array[ibin + (img_order + 1) * total_bin, 2]

                        brdf_coeffs_List[ibin]['fVol'].append(fVol)
                        brdf_coeffs_List[ibin]['fGeo'].append(fGeo)
                        brdf_coeffs_List[ibin]['fIso'].append(fIso)

                        # save the same coefficient information in diagnostic arrays
                        brdf_coeff_array[ibin, 2 + w] = fVol
                        brdf_coeff_array[ibin, 2 + w + len(hyObj.wavelengths)] = fGeo
                        brdf_coeff_array[ibin, 2 + w + 2 * len(hyObj.wavelengths)] = fIso

                    # update array for BRDF output diagnostic files
                    mid_ndvi_list = brdf_coeff_array[:total_bin, 0].astype(np.float)

                    # check linearity( NDVI as X v.s. kernel coefficients as Y ), save to diagnostic file, BIN by BIN, and wavelength by wavelength
                    if np.count_nonzero(brdf_coeff_array[:, 2 + w]) > 3:
                        # volumetric coefficients
                        temp_y = brdf_coeff_array[:total_bin, 2 + w].astype(np.float)
                        slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin, rmse_ndvi_bin = cal_r2_single(mid_ndvi_list[(mid_ndvi_list > BRDF_VEG_lower_bound) & (temp_y != 0)], temp_y[(mid_ndvi_list > BRDF_VEG_lower_bound) & (temp_y != 0)])
                        brdf_coeff_array[total_bin + 1:total_bin + 7, 2 + w] = slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin, rmse_ndvi_bin

                        # geometric coefficients
                        temp_y = brdf_coeff_array[:total_bin, 2 + w + 1 * len(hyObj.wavelengths)].astype(np.float)
                        slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin, rmse_ndvi_bin = cal_r2_single(mid_ndvi_list[(mid_ndvi_list > BRDF_VEG_lower_bound) & (temp_y != 0)], temp_y[(mid_ndvi_list > BRDF_VEG_lower_bound) & (temp_y != 0)])
                        brdf_coeff_array[total_bin + 1:total_bin + 7, 2 + w + len(hyObj.wavelengths)] = slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin, rmse_ndvi_bin

                        # isotropic coefficients
                        temp_y = brdf_coeff_array[:total_bin, 2 + w + 2 * len(hyObj.wavelengths)].astype(np.float)
                        slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin, rmse_ndvi_bin = cal_r2_single(mid_ndvi_list[(mid_ndvi_list > BRDF_VEG_lower_bound) & (temp_y != 0)], temp_y[(mid_ndvi_list > BRDF_VEG_lower_bound) & (temp_y != 0)])
                        brdf_coeff_array[total_bin + 1:total_bin + 7, 2 + w + 2 * len(hyObj.wavelengths)] = slope_ndvi_bin, intercept_ndvi_bin, r_value_ndvi_bin, p_value_ndvi_bin, std_err_ndvi_bin, rmse_ndvi_bin

    # Export coefficients to JSON
    if args.topo:
        if (not args.topo_sep) or (len(args.img) == 1):
            topo_json = "%s%s_topo_coeffs.json" % (args.od, args.pref)
            with open(topo_json, 'w') as outfile:
                json.dump(topo_coeffs, outfile)
        else:
            for i_img in range(len(args.img)):
                # filename_pref = (os.path.basename(args.img[i_img])).split('_')[0]
                filename_pref = image_uniq_name_list[i_img]
                topo_json = "%s%s_topo_coeffs.json" % (args.od, filename_pref)
                with open(topo_json, 'w') as outfile:
                    json.dump(topo_coeff_list[i_img], outfile)

    if args.brdf:

        if len(args.img) > 1:
            # In grouping mode, save arrays for BRDF diagnostic information to ascii files
            np.savetxt("%s%s_brdf_coeffs_r2.csv" % (args.od, args.pref), r_squared_array, header=r2_header, delimiter=',', fmt='%s', comments='')
            np.savetxt("%s%s_brdf_coeffs_rmse.csv" % (args.od, args.pref), rmse_array, header=r2_header, delimiter=',', fmt='%s', comments='')
            np.savetxt("%s%s_brdf_coeffs_fit.csv" % (args.od, args.pref), brdf_coeff_array, header=brdf_coeffs_header, delimiter=',', fmt='%s', comments='')

        if total_bin > 0:
            for ibin in range(total_bin):
                if brdf_mask_stat[ibin] < MIN_SAMPLE_COUNT:
                    continue
                brdf_json = "%s%s_brdf_coeffs_%s.json" % (args.od, args.pref, str(ibin + 1))
                with open(brdf_json, 'w') as outfile:
                    json.dump(brdf_coeffs_List[ibin], outfile)
        else:
            brdf_json = "%s%s_brdf_coeffs_1.json" % (args.od, args.pref)
            with open(brdf_json, 'w') as outfile:
                json.dump(brdf_coeffs, outfile)


if __name__ == "__main__":
    main()
