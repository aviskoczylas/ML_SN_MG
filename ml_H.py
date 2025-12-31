import time
start = time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from scipy.special import roots_legendre
from sklearn.preprocessing import MinMaxScaler
from scipy.special import lpmv
from numpy.polynomial.legendre import leggauss
print("Start time: "+str(time.time()-start))
use_stored_model = False
a = 1 #thickness, not given in xdata
num_groups = 8
num_nodes = 100
num_ordinates = 64
epochs = 50
num_hp_trials = 100
source_order = 4
bc_order = 4
bc_data_in_y = 2*bc_order*num_groups
eps = 1e-10

# load data
xtrain = pd.read_csv("data/xtrain.csv").to_numpy()
ytrain = pd.read_csv("data/Hydrogen/ytrain.csv").to_numpy()
xtest = pd.read_csv("data/xtest.csv").to_numpy()
ytest = pd.read_csv("data/Hydrogen/ytest.csv").to_numpy()


xscaler = MinMaxScaler()
xtrain = xscaler.fit_transform(xtrain)
xtest = xscaler.transform(xtest)
'''yscaler = MinMaxScaler()
ytrain = yscaler.fit_transform(ytrain)
ytest = yscaler.transform(ytest)
'''
# search space
param_space = {
    'learning_rate': np.random.uniform(0.0001, 0.001, 10).tolist(),
    'num_layers': np.random.randint(1, 6, 6).tolist(),
    'num_layer_nodes': [16, 32, 64, 128, 256, 512],
    'batch_size': [8, 16, 32, 64, 128]
}

def groupwise_fourier_expansion(coeffs, order, num_nodes, num_groups, x):
    #groupwise data is initially flattened in coeffs
    x = tf.reshape(x, [-1, num_nodes, 1])
    x = tf.cast(x, tf.float32)
    coeffs = tf.reshape(coeffs, [-1, order*2+1, num_groups])
    # First coefficient (broadcasted across x)
    expansion = tf.expand_dims(coeffs[:, 0, :], axis=1)  # Shape: [batch, 1, num_groups]

    for k in range(order):
        cos_term = tf.expand_dims(coeffs[:, 2 * k + 1, :], axis=1) * tf.cos(np.pi * (k + 1) * x)
        sin_term = tf.expand_dims(coeffs[:, 2 * k + 2, :], axis=1) * tf.sin(np.pi * (k + 1) * x)
        expansion += cos_term + sin_term

    return expansion  # Shape: [batch, num_nodes, num_groups]

quadrature_points, weight = leggauss(num_ordinates)
legendre = np.zeros((bc_order, num_ordinates))
for l in range(bc_order):
    for n in range(num_ordinates):
        legendre[l, n] = lpmv(0, l, quadrature_points[n])
legendre = tf.convert_to_tensor(legendre, dtype=tf.float32)

def legendre_expansion(moments, bc_order, num_groups, dir):
    moments = tf.reshape(moments, [-1, bc_order, num_groups]) #shape [batch, bc_order, num_groups]
    weights = tf.cast(tf.expand_dims(((2*tf.range(bc_order)+1)/2), -1), tf.float32)
    weighted_moments = weights*moments  #shape [batch, bc_order, num_groups]
    if dir == -1:
        legendre_values = legendre[:,:num_ordinates//2]
    elif dir == 1:
        legendre_values = legendre[:,num_ordinates//2:]
    angular_flux = tf.matmul(tf.transpose(weighted_moments, perm = [0,2,1]), legendre_values)  #shape [batch, num_groups, num_ordinates//2]
    angular_flux = tf.transpose(angular_flux, perm = [0,2,1])  #shape [batch, num_ordinates//2, num_groups]
    return angular_flux

def loss_func(ytrue, ypred):
    """
    Computes L2 loss on the Gauss-Legendre integral of the reconstructed spectrum.
    
    - y_true, y_pred: Coefficients to reconstruct the spectrum (shape: [batch, num_coeffs])
    - s_order, b_order, num_nodes: Integers used in spectrum reconstruction.
    
    Returns:
    - Scalar loss (L2 norm of integral error between reconstructed spectra).
    """
    # Gauss-Legendre Quadrature (same method as before)
    xi, wi = roots_legendre(num_nodes)
    xi = tf.convert_to_tensor(xi, dtype=tf.float32) 
    wi = tf.convert_to_tensor(wi, dtype=tf.float32)
    x_batch = (a / 2) * (xi + 1) 
    phi     = groupwise_fourier_expansion(ytrue[:,bc_data_in_y:], source_order, num_nodes, num_groups, x_batch)  
    phi_pred = groupwise_fourier_expansion(ypred[:,bc_data_in_y:], source_order, num_nodes, num_groups, x_batch)  

    # Evaluate loss separately for each group
    phi_loss = 0
    # for i in range(num_groups):
    #    phi_integral     = tf.matmul(phi[:,:,i],     tf.expand_dims(wi, axis=-1))  
    #    phi_pred_integral = tf.matmul(phi_pred[:,:,i], tf.expand_dims(wi, axis=-1))  
    #    phi_loss += tf.reduce_mean(tf.square(phi_integral - phi_pred_integral)) 

        # Compute L2 norm of integral error

    #phi_loss = tf.reduce_mean(tf.abs(phi - phi_pred) / tf.abs(phi+eps)) 
    phi_loss += tf.reduce_mean(tf.square(phi - phi_pred)) 
    #for some reason this works better than the commented groupwise method?

    bc_loss = 0
    bc_l     = legendre_expansion(ytrue[:,:bc_data_in_y//2], bc_order, num_groups, -1)
    bc_l_pred = legendre_expansion(ypred[:,:bc_data_in_y//2], bc_order, num_groups, -1)
    #bc_loss += tf.reduce_mean(tf.abs(bc_l - bc_l_pred) / tf.abs(bc_l+eps)) 
    bc_loss += tf.reduce_mean(tf.square(bc_l - bc_l_pred)) 

    bc_r     = legendre_expansion(ytrue[:,bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)
    bc_r_pred = legendre_expansion(ypred[:,bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)
    #bc_loss += tf.reduce_mean(tf.abs(bc_r - bc_r_pred) / tf.abs(bc_r+eps)) 
    bc_loss += tf.reduce_mean(tf.square(bc_r - bc_r_pred)) 

    total_loss = phi_loss + bc_loss

    return total_loss

# hp tuning
BEST_PARAMS_FILE = "models/H_tf_nn_hps.npy"
if os.path.exists(BEST_PARAMS_FILE):
    best_params = np.load(BEST_PARAMS_FILE, allow_pickle=True).item()
    print("Loaded best hyperparameters:", best_params)
else:
    best_loss = float("inf")
    best_params = {}
    for _ in range(num_hp_trials):  
        lr = np.random.choice(param_space['learning_rate'])
        num_layers = np.random.choice(param_space['num_layers'])
        num_layer_nodes = np.random.choice(param_space['num_layer_nodes'])
        batch_size = np.random.choice(param_space['batch_size'])
        model = Sequential()
        model.add(Input(shape=(xtrain.shape[1],)))
        for _ in range(num_layers):
            model.add(Dense(num_layer_nodes, activation='relu'))
        model.add(Dense(ytrain.shape[1], activation='linear'))

        model.compile(optimizer=Adam(learning_rate=lr), 
                        loss=loss_func, 
                        metrics=['mae'])

        history = model.fit(xtrain, 
                ytrain, 
                epochs=int(epochs/5), 
                batch_size=batch_size, 
                validation_data=(xtest, ytest), 
                verbose=1,)

        val_loss = min(history.history['val_loss'])
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = {'learning_rate': lr, 
                    'num_layers': num_layers, 
                    'num_layer_nodes': num_layer_nodes, 
                    'batch_size': batch_size}
        tf.keras.backend.clear_session()
    # Save 
    np.save(BEST_PARAMS_FILE, best_params)
    print("Saved best hyperparameters:", best_params)

MODEL_FILE = "models/best_H_model.model.keras"
if use_stored_model and os.path.exists(MODEL_FILE):
    model = load_model(MODEL_FILE, custom_objects={'loss_func': loss_func})
else:
    model = Sequential()
    model.add(Input(shape=(xtrain.shape[1],)))
    for _ in range(best_params['num_layers']):
        model.add(Dense(best_params['num_layer_nodes'], activation='relu'))
    model.add(Dense(ytrain.shape[1], activation='linear'))
    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), 
                                loss=loss_func,
                                metrics=['mae'])

    model_checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_FILE,
        monitor='mae',
        mode='min',
        save_best_only=True)

    history = model.fit(
                    xtrain, 
                    ytrain, 
                    epochs=epochs, 
                    batch_size=best_params['batch_size'], 
                    validation_data=(xtest, ytest),
                    callbacks = [model_checkpoint_callback],
                    verbose = 1)

    # learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.yscale("log")
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    plt.savefig("charts/ml_H/tf_nn_learning_curve.png")
    plt.close()

def np_legendre(moments, bc_order, num_groups, dir):
    #moments arrive flattened, reshape to [num_samples, bc_order, num_groups]
    if dir == -1:
        legendre_values = legendre[:,:num_ordinates//2]
    elif dir == 1:
        legendre_values = legendre[:,num_ordinates//2:]
    moments = np.reshape(moments, [-1, bc_order, num_groups])
    flux = []
    for g in range(num_groups):
        flux.append(np.zeros((moments.shape[0], num_ordinates//2)))
        for l in range(bc_order):
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
    #group_flux += np.expand_dims(coeffs[:, 0, :], axis = -1)
    # for k in range(source_order):
    #     group_flux += np.multiply.outer(coeffs[:, 2 * k + 1, :], np.cos(np.pi * (k + 1) * x))
    #     group_flux += np.multiply.outer(coeffs[:, 2 * k + 2, :], np.sin(np.pi * (k + 1) * x))
    # return np.sum(group_flux, axis = 1)

#L2 norm on coeffs
ypred = model.predict(xtest)
'''ypred = yscaler.inverse_transform(ypred)
ytest = yscaler.inverse_transform(ytest)
'''
df_pred = pd.DataFrame(ypred)
df_pred.to_csv("ypred_H.csv",index = False)
l2_errors = np.linalg.norm(ytest - ypred, axis=1)
print(f"L2 Error on Output: {np.mean(l2_errors):.5f}")

# dataset L2 error
phi     = fourier_expansion(ytest[:, bc_data_in_y:],source_order,num_nodes,num_groups)
phi_hat = fourier_expansion(ypred[:, bc_data_in_y:],source_order,num_nodes,num_groups)
bc_l = np_legendre(ytest[:,:bc_data_in_y//2], bc_order, num_groups, -1)
bc_l_hat = np_legendre(ypred[:,:bc_data_in_y//2], bc_order, num_groups, -1)
bc_r = np_legendre(ytest[:,bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)
bc_r_hat = np_legendre(ypred[:,bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)

L2_phis = [np.mean(np.linalg.norm(phi[g] - phi_hat[g], axis=1)) for g in range(num_groups)]
L2_phi = np.mean(L2_phis)
print(f"L2 error on scalar flux = {np.round(L2_phi,5)}")
L2_bc_ls = [np.mean(np.linalg.norm(bc_l[g] - bc_l_hat[g], axis=1)) for g in range(num_groups)]
L2_bc_l = np.mean(L2_bc_ls)
print(f"L2 error on left boundary = {np.round(L2_bc_l,5)}")
L2_bc_rs = [np.mean(np.linalg.norm(bc_r[g] - bc_r_hat[g], axis=1)) for g in range(num_groups)]
L2_bc_r = np.mean(L2_bc_rs)
print(f"L2 error on right boundary = {np.round(L2_bc_r,5)}")
# residuals
#phi_percent_errors = [np.mean(np.abs(phi[g] - phi_hat[g]) / (np.abs(phi[g]) + eps), axis=1) for g in range(num_groups)]

#phi, bc, etc is list of num_groups arrays, each array is of shape [num_samples, num_nodes or num_ordinates//2]
phi_percent_errors = [np.abs(phi[g] - phi_hat[g]) / (np.abs(phi[g]) + eps) for g in range(num_groups)] 
bc_l_percent_errors = [np.abs(bc_l[g] - bc_l_hat[g]) / (np.abs(bc_l[g]) + eps) for g in range(num_groups)] 
bc_r_percent_errors = [np.abs(bc_r[g] - bc_r_hat[g]) / (np.abs(bc_r[g]) + eps) for g in range(num_groups)] 
group_percent_errors = []
phi_group_percent_errors = []
for g in range(num_groups):
    phi_group_percent_errors.append(np.mean(phi_percent_errors[g], axis = 1))
    bc_l_group_percent_error = np.mean(bc_l_percent_errors[g], axis = 1)
    bc_r_group_percent_error = np.mean(bc_r_percent_errors[g], axis = 1)
    group_percent_errors.append((phi_group_percent_errors[g] + (bc_l_group_percent_error + bc_r_group_percent_error) / 2) / 2)

phi_percent_errors = np.mean(np.mean(phi_percent_errors, axis = 2), axis = 0)
bc_l_percent_errors = np.mean(np.mean(bc_l_percent_errors, axis = 2), axis = 0)
bc_r_percent_errors = np.mean(np.mean(bc_r_percent_errors, axis = 2), axis = 0)
percent_errors = (phi_percent_errors + (bc_l_percent_errors + bc_r_percent_errors) / 2) / 2

best_idx = np.argmin(np.abs(percent_errors))
worst_idx = np.argmax(np.abs(percent_errors))
print(f"Best Index: {best_idx}, Worst Index: {worst_idx}")
x = np.linspace(0, 1, num_nodes)

def plot_flux(x, t, ytest, ypred, flux_order, num_nodes, idx, title):
    """Plot expansion and TT solution."""
    true_fluxes = fourier_expansion(ytest[bc_data_in_y:], flux_order, num_nodes, num_groups)
    pred_fluxes = fourier_expansion(ypred[bc_data_in_y:], flux_order, num_nodes, num_groups)
    true_boundary_l = np_legendre(ytest[:bc_data_in_y//2], bc_order, num_groups, -1)
    pred_boundary_l = np_legendre(ypred[:bc_data_in_y//2], bc_order, num_groups, -1)
    true_boundary_r = np_legendre(ytest[bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)
    pred_boundary_r = np_legendre(ypred[bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)
    for g in range(num_groups):
        plt.clf()
        plt.plot(
            t * x,
            np.squeeze(true_fluxes[g]),
            label="Y-Test",
        )
        plt.plot(
            t * x,
            np.squeeze(pred_fluxes[g]),
            label="Y-Pred",
        )
        plt.xlabel("$x$")
        plt.ylabel("$\\phi(x)$")
        plt.legend()
        plt.title(f"{title} vs x, Group {g+1}")
        plt.grid()
        plt.savefig(f"charts/ml_H/tf_ytest_ypred_{idx}_phi_group{g+1}.png")
        plt.close()

        plt.clf()
        plt.plot(
            quadrature_points[:num_ordinates//2],
            np.squeeze(true_boundary_l[g]),
            label="Y-Test",
        )
        plt.plot(
            quadrature_points[:num_ordinates//2],
            np.squeeze(pred_boundary_l[g]),
            label="Y-Pred",
        )
        plt.xlabel("$\\mu$")
        plt.ylabel("$\\psi(\\mu)$")
        plt.legend()
        plt.title(f"{title} vs $\\mu$, Group {g+1}, Left Boundary")
        plt.grid()
        plt.savefig(f"charts/ml_H/tf_ytest_ypred_{idx}_lb_group{g+1}.png")
        plt.close()


        plt.clf()
        plt.plot(
            quadrature_points[num_ordinates//2:],
            np.squeeze(true_boundary_r[g]),
            label="Y-Test",
        )
        plt.plot(
            quadrature_points[num_ordinates//2:],
            np.squeeze(pred_boundary_r[g]),
            label="Y-Pred",
        )
        plt.xlabel("$\\mu$")
        plt.ylabel("$\\psi(\\mu)$")
        plt.legend()
        plt.title(f"{title} vs $\\mu$, Group {g+1}, Right Boundary")
        plt.grid()
        plt.savefig(f"charts/ml_H/tf_ytest_ypred_{idx}_rb_group{g+1}.png")
        plt.close()

plot_flux(
    x,
    a,   # thickness
    ytest[best_idx, :],  # ytest coeffs
    ypred[best_idx, :],   # ypred_coeffs
    source_order,
    num_nodes,
    best_idx,
    f"Y-Test and Y-Pred, Best Index"
)

plot_flux(
    x,
    a,   # thickness
    ytest[worst_idx, :],  # ytest coeffs
    ypred[worst_idx, :],   # ypred_coeffs
    source_order,
    num_nodes,
    worst_idx,
    f"Y-Test and Y-Pred, Worst Index"
)

percent_errors *= 100
plt.hist(percent_errors, bins = 100, range = (0,10), weights = np.ones_like(percent_errors)/len(percent_errors))
plt.xlabel("Percent Error")
plt.ylabel("Fraction of Samples")
plt.title("Percent Error Histogram")
plt.grid()
plt.savefig(f"charts/ml_H/hist.png")
plt.clf()
plt.close()
plt.hist(percent_errors, bins = 100, range = (0,100), weights = np.ones_like(percent_errors)/len(percent_errors))
plt.xlabel("Percent Error")
plt.ylabel("Fraction of Samples")
plt.title("Percent Error Histogram")
plt.grid()
plt.savefig(f"charts/ml_H/full_hist.png")
plt.clf()
plt.close()
percent_over_100 = len([i for i in percent_errors if i >= 100])/len(percent_errors)*100
percent_over_10 = len([i for i in percent_errors if i >= 10])/len(percent_errors)*100
percent_over_5 = len([i for i in percent_errors if i >= 5])/len(percent_errors)*100
percent_less_than_5 = len([i for i in percent_errors if i < 5])/len(percent_errors)*100
print(str(percent_over_100)+"% of samples over 100% relative error")
print(str(percent_over_10)+"% of samples over 10% relative error")
print(str(percent_over_5)+"% of samples over 5% relative error")
print(str(percent_less_than_5)+"% of samples less than 5% relative error")

phi_percent_errors *= 100
plt.hist(phi_percent_errors, bins = 100, range = (0,10), weights = np.ones_like(percent_errors)/len(percent_errors))
plt.xlabel("Percent Error")
plt.ylabel("Fraction of Samples")
plt.title("Percent Error Histogram on Flux")
plt.grid()
plt.savefig(f"charts/ml_H/phi_hist.png")
plt.clf()
plt.close()
plt.hist(phi_percent_errors, bins = 100, range = (0,100), weights = np.ones_like(phi_percent_errors)/len(phi_percent_errors))
plt.xlabel("Percent Error")
plt.ylabel("Fraction of Samples")
plt.title("Percent Error Histogram on Flux")
plt.grid()
plt.savefig(f"charts/ml_H/phi_full_hist.png")
plt.clf()
plt.close()
print("Here, percent error is only evaluated using phi")
percent_over_100 = len([i for i in phi_percent_errors if i >= 100])/len(phi_percent_errors)*100
percent_over_10 = len([i for i in phi_percent_errors if i >= 10])/len(phi_percent_errors)*100
percent_over_5 = len([i for i in phi_percent_errors if i >= 5])/len(phi_percent_errors)*100
percent_less_than_5 = len([i for i in phi_percent_errors if i < 5])/len(phi_percent_errors)*100
print(str(percent_over_100)+"% of samples over 100% relative error")
print(str(percent_over_10)+"% of samples over 10% relative error")
print(str(percent_over_5)+"% of samples over 5% relative error")
print(str(percent_less_than_5)+"% of samples less than 5% relative error")

group_percent_errors = [x*100 for x in group_percent_errors]
phi_group_percent_errors = [x*100 for x in phi_group_percent_errors]
for g in range(num_groups):
    plt.hist(group_percent_errors[g], bins = 100, range = (0,10), weights = np.ones_like(group_percent_errors[g])/len(group_percent_errors[g]), alpha=.35, label = f"Group {g+1}")
plt.xlabel("Percent Error")
plt.ylabel("Fraction of Samples")
plt.title(f"Percent Error Histogram, Group {g}")
plt.grid()
plt.legend()
plt.savefig(f"charts/ml_H/one_plot_histograms.png")
plt.clf()
plt.close()

for g in range(num_groups):
    plt.hist(group_percent_errors[g], bins = 100, range = (0,10), weights = np.ones_like(group_percent_errors[g])/len(group_percent_errors[g]))
    plt.xlabel("Percent Error")
    plt.ylabel("Fraction of Samples")
    plt.xlim([0,10])
    plt.title(f"Percent Error Histogram, Group {g}")
    plt.grid()
    plt.savefig(f"charts/ml_H/hist_group_{g}.png")
    plt.clf()
    plt.close()
    plt.hist(group_percent_errors[g], bins = 100, range = (0,100), weights = np.ones_like(group_percent_errors[g])/len(group_percent_errors[g]))
    plt.xlabel("Percent Error")
    plt.ylabel("Fraction of Samples")
    plt.title(f"Percent Error Histogram, Group {g}")
    plt.xlim([0,10])
    plt.grid()
    plt.savefig(f"charts/ml_H/full_hist_group_{g}.png")
    plt.clf()
    plt.close()
    plt.hist(phi_group_percent_errors[g], bins = 100, range = (0,10), weights = np.ones_like(phi_group_percent_errors[g])/len(phi_group_percent_errors[g]))
    plt.xlabel("Percent Error")
    plt.ylabel("Fraction of Samples")
    plt.title(f"Percent Error Histogram on Flux, Group {g}")
    plt.xlim([0,10])
    plt.grid()
    plt.savefig(f"charts/ml_H/phi_hist_group_{g}.png")
    plt.clf()
    plt.close()
    plt.hist(phi_group_percent_errors[g], bins = 100, range = (0,100), weights = np.ones_like(phi_group_percent_errors[g])/len(phi_group_percent_errors[g]))
    plt.xlabel("Percent Error")
    plt.ylabel("Fraction of Samples")
    plt.xlim([0,10])
    plt.title(f"Percent Error Histogram on Flux, Group {g}")
    plt.grid()
    plt.savefig(f"charts/ml_H/phi_full_hist_group_{g}.png")
    plt.clf()
    plt.close()
print("End time: "+str(time.time()-start))
