import os, sys
import pandas as pd
from data_preprocessing import data_preprocessing
from decoder_model import decoder
from vae_arch import vae_kl, vae_js,vae_wd
from plots_results_training import result_plots,plot

ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
TEST_CASE = '1D_heat'
TEST_CASE_DIR = os.path.join(OUTPUT_DIR,TEST_CASE) 
PLOT_DIR = os.path.join(TEST_CASE_DIR,'plots')
n_poi = 1

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_CASE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Load the data. NOTE: Needs to be shifted to numpy array in binary format
data = pd.read_csv(os.path.join(DATA_DIR,TEST_CASE+'.csv')).to_numpy()
#data preprocessing , update the percentage as required
#u_train_noisy, u_test_noisy, u_train, u_test, k_train, k_test, k_train_prior, k_test_prior = data_preprocessing(data, percentage=0.25)
train_data_scaled, test_data_scaled, valid_data_scaled, poi_train_prior, poi_test_prior, poi_valid_prior = data_preprocessing(data, n_poi=n_poi)

#training the decoder model with non noisy data as a surrogate
history_decoder = decoder(train_data_scaled, test_data_scaled, valid_data_scaled, n_poi=n_poi)
plot_path = os.path.join(PLOT_DIR, 'decoder_training.png')
plot(history_decoder,save_path=plot_path)

u_train = train_data_scaled[:,:-n_poi]
u_test = test_data_scaled[:,:-n_poi]
u_valid = valid_data_scaled[:,:-n_poi]

def process_vae(vae_function, name, u_train = u_train, u_test=u_test, u_valid = u_valid, poi_train_prior= poi_train_prior, 
                poi_test_prior= poi_test_prior, poi_valid_prior=poi_valid_prior):
    history_vae, z_mean, z_log_var, z_mean_t, z_log_var_t = vae_function(u_train, u_test, poi_train_prior, poi_test_prior)
    plot_path = os.path.join(PLOT_DIR, f"{name}_plot.png")
    plot(history_vae, save_path=plot_path)
    result_plot_path_1 = os.path.join(OUTPUT_DIR, f"{name}_train_results.png")
    result_plot_path_2 = os.path.join(OUTPUT_DIR, f"{name}_test_results.png")
    mse, r2, mse_t, r2_t = result_plots(z_mean, z_log_var, z_mean_t, z_log_var_t, u_train, u_test, save_path_train=result_plot_path_1, save_path_test=result_plot_path_2)
    return mse, r2, mse_t, r2_t
def process_vae_2(vae_function, name):
    history_vae, z_mean, z_log_var, z_mean_t, z_log_var_t = vae_function(u_train_noisy, u_test_noisy,k_train,k_test, k_train_prior, k_test_prior)
    plot_path = os.path.join(PLOT_DIR, f"{name}_plot.png")
    plot(history_vae, save_path=plot_path)
    result_plot_path_1 = os.path.join(OUTPUT_DIR, f"{name}_train_results.png")
    result_plot_path_2 = os.path.join(OUTPUT_DIR, f"{name}_test_results.png")
    mse, r2, mse_t, r2_t = result_plots(z_mean, z_log_var, z_mean_t, z_log_var_t, k_train, k_test, save_path_train=result_plot_path_1, save_path_test=result_plot_path_2)
    return mse, r2, mse_t, r2_t

# Process VAE models
mse_k, r2_k, mse_t_k, r2_t_k = process_vae(vae_kl, "vae_kl")
#mse_j, r2_j, mse_t_j, r2_t_j = process_vae_2(vae_js, "vae_js")
#mse_w, r2_w, mse_t_w, r2_t_w = process_vae_2(vae_wd, "vae_wd")

# Print results
print(f"VAE KL - MSE for train data is {mse_k} and R2 score is {r2_k}")
print(f"VAE KL - MSE for test data is {mse_t_k} and R2 score is {r2_t_k}")
#print(f"VAE JS - MSE for train data is {mse_j} and R2 score is {r2_j}")
#print(f"VAE JS - MSE for test data is {mse_t_j} and R2 score is {r2_t_j}")
#print(f"VAE WD - MSE for train data is {mse_w} and R2 score is {r2_w}")
#print(f"VAE WD - MSE for test data is {mse_t_w} and R2 score is {r2_t_w}")
