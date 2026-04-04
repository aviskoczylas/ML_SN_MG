import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from scipy.special import roots_legendre
from sklearn.preprocessing import MinMaxScaler
from scipy.special import lpmv
from numpy.polynomial.legendre import leggauss

#constants
num_groups = 8
num_nodes = 100
num_ordinates = 64
bc_order = 4
leg_order = 4
source_order = 4
bc_data_in_y = 2*bc_order*num_groups
num_regions = 8
min_fuel_frac = 0.6
tolerance = 1e-6


#set up indexing

# output structure:
#   left bc angular flux moments  -> bc_order * num_groups
#   right bc angular flux moments -> bc_order * num_groups  
#   scalar flux fourier coeffs -> (2*source_order+1) * num_groups

# input structure based on sn_solver_fixed_source
#   groupwise sig_t -> num_groups
#   group to group scattering xs -> leg_order * num_groups * num_groups
#   left bc angular flux (if inflow) -> num_ordinates//2 * num_groups
#   right bc angular flux (if inflow) -> num_ordinates//2 * num_groups
#   source coeffs -> (2 * source_order + 1) * num_groups
#   NH -> 1 col
#   FIXME Not here - is it necessary? bc type, 2 col categorical. Included for now.

sig_t_size = num_groups
xs_gtg_size = leg_order * num_groups * num_groups
left_bc_size = (num_ordinates // 2) * num_groups
right_bc_size = (num_ordinates // 2) * num_groups
source_coeffs_size = (2 * source_order + 1) * num_groups
NA_size = 1
bc_type_size =2 # NEW INPUT FOR FIXED SOURCE!!
# Cumulative indices
idx = 0
sig_t_idx = (idx, idx + sig_t_size)
idx += sig_t_size
xs_gtg_idx = (idx, idx + xs_gtg_size)
idx += xs_gtg_size
left_bc_idx = (idx, idx + left_bc_size)
idx += left_bc_size
right_bc_idx = (idx, idx + right_bc_size)
idx += right_bc_size
source_coeffs_idx = (idx, idx + source_coeffs_size)
idx += source_coeffs_size
NA_idx = (idx, idx + NA_size)
idx += NA_size
bc_type_idx = (idx, idx + bc_type_size)
idx += bc_type_size




#read material propeties from xtrain. 
# FIXME Right now, both moderator and fuel properties are the same (taken from xtest) but of course that will need to change
material_properties_file = pd.read_csv("data/xtest1.csv").to_numpy()
sig_t_fuel = material_properties_file[:, sig_t_idx[0]:sig_t_idx[1]]
xs_gtg_fuel = material_properties_file[:, xs_gtg_idx[0]:xs_gtg_idx[1]]
source_coeffs = material_properties_file[:, source_coeffs_idx[0]:source_coeffs_idx[1]]
N_fuel = material_properties_file[:, NA_idx[0]:NA_idx[1]]
sig_t_mod = sig_t_fuel
xs_gtg_mod = xs_gtg_fuel
N_mod = N_fuel

#load models
FUEL_MODEL_FILE = "models/best_fuel_model.model.keras"
if os.path.exists(FUEL_MODEL_FILE):
    fuel_model = load_model(FUEL_MODEL_FILE, compile = False)
MOD_MODEL_FILE = "models/best_H_model.model.keras"
if os.path.exists(MOD_MODEL_FILE):
    moderator_model = load_model(MOD_MODEL_FILE, compile = False)

#for making input sizes work while I'm initially writing this 
def create_debug_model(input_size, output_size):
    model = Sequential([
        tf.keras.layers.Input(shape=(input_size,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

input_size = idx
output_size = bc_data_in_y + (2*source_order+1)*num_groups
fuel_model = create_debug_model(input_size, output_size)
moderator_model = create_debug_model(input_size, output_size)

#set up storage arrays
#FIXME temporarily set to 16 bit. my computer doesnt have enough ram to store fluxes for all regions at once.
#So there's probably some optimization for ram to be done here.
#either way, this has to change to prevent overflow when running for real
num_samples = material_properties_file.shape[0]
saved_phi = np.zeros((num_groups, num_samples, num_nodes, num_regions), dtype=np.float16)
saved_phi_prev = np.zeros_like(saved_phi)
saved_bc_l = np.zeros((num_samples, left_bc_size, num_regions), dtype=np.float16)
saved_bc_r = np.zeros((num_samples, right_bc_size, num_regions), dtype=np.float16)

flux_preds = [0] * num_regions
xscaler = MinMaxScaler()




#generate random fuel/moderator configuration with a certain percentage fuel guaranteed
regions = np.zeros(num_regions)

while regions.sum() < min_fuel_frac*num_regions:
    #1 = fuel, 0 = moderator
    regions = (np.random.random(num_regions) < 0.5).astype(int)
np.savetxt("fuel_mod_layout.csv", regions)





#functions for handling legendre/fourier expansions
quadrature_points, weight = leggauss(num_ordinates)
legendre = np.zeros((bc_order, num_ordinates))
for l in range(bc_order):
    for n in range(num_ordinates):
        legendre[l, n] = lpmv(0, l, quadrature_points[n])

def np_legendre(moments, order, num_groups, dir):
    #moments arrive flattened, reshape to [num_samples, order, num_groups]
    if dir == -1:
        legendre_values = legendre[:,:num_ordinates//2]
    elif dir == 1:
        legendre_values = legendre[:,num_ordinates//2:]
    moments = np.reshape(moments, [-1, order, num_groups])
    flux = []
    for g in range(num_groups):
        flux.append(np.zeros((moments.shape[0], num_ordinates//2)))
        for l in range(order):
            flux[g]+=(2*l+1)/2*np.outer(moments[:,l,g],legendre_values[l,:])
    return flux

def fourier_expansion(coeffs,order,num_nodes, num_groups):
    if coeffs.ndim == 1:
        coeffs = np.expand_dims(coeffs,axis = 0)
    x = np.linspace(0, 1, num_nodes)
    coeffs = np.reshape(coeffs, [-1, order*2+1, num_groups]) 
    # Compute source by Fourier expansion
    flux = []
    for g in range(num_groups):
        flux.append(np.zeros((coeffs.shape[0], num_nodes)))
        flux[g] += np.expand_dims(coeffs[:,0,g], axis = -1)
        for k in range(order):
            flux[g] += np.outer(coeffs[:, 2 * k + 1, g], np.cos(np.pi * (k + 1) * x))
            flux[g] += np.outer(coeffs[:, 2 * k + 2, g], np.sin(np.pi * (k + 1) * x))
    return flux




converged = False
iteration = 0
while not converged:
    iteration += 1
    #sweep left to right, then right to left
    for i in range(2):
        if i == 0:
            sweep_range = range(num_regions)
        else:
            sweep_range = reversed(range(num_regions))
        for r in sweep_range:
            #first, assign bcs from previous region
            x_data = np.zeros((num_samples, input_size))
            x_data[:, left_bc_idx[0] : left_bc_idx[1]] = saved_bc_l[:, :, r]
            x_data[:, right_bc_idx[0]:right_bc_idx[1]] = saved_bc_r[:, :, r]
            if r:
                x_data[:, -2] = 1 #corresponds to left vacuum
            else:
                x_data[:, -2] = 2 #corresponds to left inflow
            if r < num_regions - 1:
                x_data[:, -1] = 1 #corresponds to right vacuum
            else:
                x_data[:, -1] = 2 #corresponds to right inflow

            if regions[r]:
                #assign material properties, presumably constant with respect to region for given sample
                x_data[:, sig_t_idx[0]:sig_t_idx[1]] = sig_t_fuel
                x_data[:, xs_gtg_idx[0]:xs_gtg_idx[1]] = xs_gtg_fuel

                #FIXME not sure how source coeffs are supposed to be handled. They likely vary by region, but 
                #how is that currently handled as input? I'll call it uniform by region for now.
                x_data[:, source_coeffs_idx[0]:source_coeffs_idx[1]] = source_coeffs
                x_data[:, NA_idx[0]:NA_idx[1]] = N_fuel

                model = fuel_model
            else:
                #assign material properties, presumably constant with respect to region for given sample
                x_data[:, sig_t_idx[0]:sig_t_idx[1]] = sig_t_mod
                x_data[:, xs_gtg_idx[0]:xs_gtg_idx[1]] = xs_gtg_mod

                x_data[:, source_coeffs_idx[0]:source_coeffs_idx[1]] = 0
                x_data[:, NA_idx[0]:NA_idx[1]] = N_mod

                model = moderator_model

            #FIXME this should be fitted in the same way as the data was when the model was initially trained, right?
            # how do i get the transformation from training to apply in this file?
            #for now, i just apply fit transform every time
            x_data = xscaler.fit_transform(x_data)
            flux_pred = model.predict(x_data)
            flux_preds[r] = flux_pred

            phi = fourier_expansion(flux_pred[:, bc_data_in_y:],source_order,num_nodes,num_groups)
            saved_phi[:, :, :, r]  = np.array([ phi[g] for g in range(num_groups) ])
            if r:
                #predicted left boundary outgoing flux becomes next incoming right boundary flux
                bc_l = np_legendre(flux_pred[:,:bc_data_in_y//2], bc_order, num_groups, -1)
                saved_bc_l[:, :, r-1] = np.concatenate(bc_l, axis=1)
            if r < num_regions - 1:
                #predicted right boundary outgoing flux becomes next incoming left boundary flux
                bc_r = np_legendre(flux_pred[:,bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)
                saved_bc_r[:, :, r+1] = np.concatenate(bc_r, axis=1)
    
    #check for convergence between sweeps
    l2_norms = np.array([
        np.linalg.norm(saved_phi[:, :, :, r] - saved_phi_prev[:, :, :, r])
        for r in range(num_regions)
    ])
    converged = np.all(l2_norms < tolerance)
    saved_phi_prev = saved_phi.copy()
    
    print(f"Iteration {iteration}: L2 norms = {l2_norms}")

for r in range(num_regions):
    pd.DataFrame(flux_preds[r]).to_csv(f"flux_pred_region_{r}.csv",index = False)
