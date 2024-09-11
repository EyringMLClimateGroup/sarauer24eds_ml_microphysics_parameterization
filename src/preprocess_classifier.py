######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: preprocess classifier.py                                                                 #
######################################################################################################
# In this script we preprocess our data for the Microphysics Trigger Classifier Model.               #
# We load our netcdf simulation file and apply preselection criteria.                                #
# We split data in test, train and validation sets and save them.                                    #
# For more information, please check Methodology section in our paper.                               #
######################################################################################################

# Import
import xarray as xr
import numpy as np
import pandas as pd
import glob
from sklearn.utils import shuffle

# Load netcdf files
data_path = "path/to/data/"
# ml_varlist = 'ps', 'psl', 'rsdt', 'rsut', 'rsutcs', 'rlut', 'rlutcs',
#              'rsds', 'rsdscs', 'rlds', 'rldscs', 'rsus', 'rsuscs', 'rlus',
#              'ts', 'sic', 'sit', 'clt', 'prlr', 'prls', 'pr', 'prw',
#              'cllvi', 'clivi', 'qgvi', 'qrvi', 'qsvi', 'cptgzvi', 'hfls',
#              'hfss', 'evspsbl', 'tauu', 'tauv', 'sfcwind', 'uas', 'vas',
#              'tas', 'pr_rain', 'pr_ice', 'pr_snow', 'pr_grpl'
atm_2d_general_vars_path = glob.glob(data_path+"*atm_2d_general_vars_ml_20200207*")
# ml_varlist = 'rho', 'ta', 'ua', 'va', 'tv', 'omega', 'hus', 'hur', 'clw',
#              'cli', 'cl'
atm_3d_general_vars_path = glob.glob(data_path+"*atm_cl_ml_20200207*")
# ml_varlist = 'dz_mig', 'rho_mig', 'pf_mig', 'cpair_mig', 'ta_mig', 'qv_mig',
#              'qc_mig', 'qi_mig', 'qr_mig', 'qs_mig', 'qg_mig'
atm_mig_inputs_path = glob.glob(data_path+"*mig_inputs_ml_20200207*")
# ml_varlist = 'tend_ta_mig', 'tend_qhus_mig', 'tend_qclw_mig',
#              'tend_qcli_mig', 'tend_qr_mig', 'tend_qs_mig', 'tend_qg_mig'
atm_mig_tendencies_path = glob.glob(data_path+"*mig_tendencies_ml_20200207*")

# Create data arrays from nc files
def create_data_array(path_to_files, varname):
    joined_arr = np.zeros(32768000,)
    for path in path_to_files:
        file = xr.open_dataset(path)
        raw_arr = file[varname]
        del_100_arr = raw_arr[:,1:51,:]
        cut_arr = np.array(del_100_arr)
        out_arr = np.reshape(cut_arr, (32768000,))
        joined_arr = np.concatenate((joined_arr,out_arr))
    print(f"Final array shape of {varname}: {joined_arr.shape}")
    return(joined_arr)

# Create arrays
print("create input arrays")
dz_mig = create_data_array(atm_mig_inputs_path, "dz_mig")
pf_mig = create_data_array(atm_mig_inputs_path, "pf_mig")
ta_mig = create_data_array(atm_mig_inputs_path, "ta_mig")
qv_mig = create_data_array(atm_mig_inputs_path, "qv_mig")
qc_mig = create_data_array(atm_mig_inputs_path, "qc_mig")
qi_mig = create_data_array(atm_mig_inputs_path, "qi_mig")
qr_mig = create_data_array(atm_mig_inputs_path, "qr_mig")
qs_mig = create_data_array(atm_mig_inputs_path, "qs_mig")
qg_mig = create_data_array(atm_mig_inputs_path, "qg_mig")
#tv = create_data_array(atm_3d_general_vars_path,"tv")
#omega = create_data_array(atm_3d_general_vars_path,"wap")
#ua = create_data_array(atm_3d_general_vars_path,"ua")
#va = create_data_array(atm_3d_general_vars_path,"va")
#hus = create_data_array(atm_3d_general_vars_path,"hus")
print("create output arrays")
tend_ta_mig = create_data_array(atm_mig_tendencies_path, "tend_ta_mig")
tend_qv_mig = create_data_array(atm_mig_tendencies_path,"tend_qhus_mig")
tend_qc_mig = create_data_array(atm_mig_tendencies_path,"tend_qclw_mig")
tend_qi_mig = create_data_array(atm_mig_tendencies_path,"tend_qcli_mig")
tend_qr_mig = create_data_array(atm_mig_tendencies_path,"tend_qr_mig")
tend_qs_mig = create_data_array(atm_mig_tendencies_path,"tend_qs_mig")
tend_qg_mig = create_data_array(atm_mig_tendencies_path,"tend_qg_mig")

# Fill dataframe
print("fill dataframe")
df_mig = pd.DataFrame()
df_mig["dz_mig"] = dz_mig
df_mig["pf_mig"] = pf_mig
df_mig["ta_mig"] = ta_mig
df_mig["qv_mig"] = qv_mig
df_mig["qc_mig"] = qc_mig
df_mig["qi_mig"] = qi_mig
df_mig["qr_mig"] = qr_mig
df_mig["qs_mig"] = qs_mig
df_mig["qg_mig"] = qg_mig
#df_mig["tv"] = tv
#df_mig["omega"] = omega
#df_mig["ua"] = ua
#df_mig["va"] = va
#df_mig["hus"] = hus
df_mig["tend_ta_mig"] = tend_ta_mig
df_mig["tend_qv_mig"] = tend_qv_mig
df_mig["tend_qc_mig"] = tend_qc_mig
df_mig["tend_qi_mig"] = tend_qi_mig
df_mig["tend_qr_mig"] = tend_qr_mig
df_mig["tend_qs_mig"] = tend_qs_mig
df_mig["tend_qg_mig"] = tend_qg_mig

# Apply preselection criteria
print("apply basic preselection criteria")
df_mig = df_mig.dropna()
print(df_mig.shape)
#df_mig_no_zero = df_mig.loc[(df_mig != 0).all(axis=1)]
df_mig_cld_sig = df_mig
cond1 = abs(df_mig_cld_sig['tend_ta_mig']) > 10**-6
cond2 = (abs(df_mig_cld_sig['tend_qv_mig']) + abs(df_mig_cld_sig['tend_qc_mig'])) > 10**-9
cond3 = (abs(df_mig_cld_sig['tend_qi_mig']) + abs(df_mig_cld_sig['tend_qr_mig']) + abs(df_mig_cld_sig['tend_qs_mig']) + abs(df_mig_cld_sig['tend_qg_mig'])) > 10**-10
df_mig_cld_sig['mig_active'] = np.where(cond1 & cond2 & cond3 , 1, 0)
df_mig_cld_sig = df_mig
print("Number of 1 entries:", df_mig_cld_sig['mig_active'].sum())
print(f"Size of dataframe: {df_mig_cld_sig.size}")
final_df = df_mig_cld_sig[["dz_mig","pf_mig","ta_mig","qv_mig","qc_mig","qi_mig","qr_mig","qs_mig","qg_mig", "mig_active"]]
print("Split in train, val, test.")
print(final_df.shape)

# Convert to numpy and shuffle
final_array = final_df.to_numpy()
total_num_samples = 30000000
num_train_samples = int(10*total_num_samples/12)
num_val_samples = int((total_num_samples-num_train_samples)/2)
num_test_samples = total_num_samples - num_train_samples - num_val_samples
data_final = shuffle(final_array, n_samples=total_num_samples)
set_train = data_final[:num_train_samples]
set_val = data_final[num_train_samples:(num_train_samples+num_val_samples), :]
set_test = data_final[(num_train_samples+num_val_samples):, :]

# Save preprocessed files
out_path = "path/to/out/"
np.save(out_path + "df_nextgems_mig_subset_classify_train.npy", set_train)
np.save(out_path + "df_nextgems_mig_subset_classify_val.npy", set_val)
np.save(out_path + "df_nextgems_mig_subset_classify_test.npy", set_test)

# Check shapes
print(set_train.shape)
print(set_test.shape)
print(set_val.shape)