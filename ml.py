import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from scipy.special import roots_legendre
from sklearn.preprocessing import MinMaxScaler
from scipy.special import lpmv
from numpy.polynomial.legendre import leggauss

a = 1 #thickness, not given in xdata
num_groups = 8
num_nodes = 100
num_ordinates = 16
epochs = 50
num_hp_trials = 20
source_order = 4
bc_order = 4
bc_data_in_y = 2*bc_order*num_groups

# load data
xtrain = pd.read_csv("data/xtrain1.csv").to_numpy()
ytrain = pd.read_csv("data/ytrain1.csv").to_numpy()
xtest = pd.read_csv("data/xtest1.csv").to_numpy()
ytest = pd.read_csv("data/ytest1.csv").to_numpy()

#remove first 264 or 776 columns for x data
xtrain, xtest = xtrain[:,776:], xtest[:,776:]

for sample in xtrain:
    if sample[-1] == "reflecting":
        sample[-1] = 0
    elif sample[-1] =="vacuum":
        sample[-1] = 1
    elif sample[-1] =="inflow":
        sample[-1] = 2
for sample in xtest:
    if sample[-1] == "reflecting":
        sample[-1] = 0
    elif sample[-1] =="vacuum":
        sample[-1] = 1
    elif sample[-1] =="inflow":
        sample[-1] = 2

#optionally truncate number of samples for speed.
reduction_factor = 1
train_samples = xtrain.shape[0]
test_samples = xtest.shape[0]
xtrain, ytrain = xtrain[:train_samples//reduction_factor,:], ytrain[:train_samples//reduction_factor,:]
xtest, ytest =  xtest[:test_samples//reduction_factor,:], ytest[:test_samples//reduction_factor,:]
# scale data

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
    'num_layers': np.random.randint(1, 10, 10).tolist(),
    'num_layer_nodes': [16, 32, 64, 128, 256],
    'batch_size': [16, 32, 64, 128]
}

def flux_loss_wrapper(source_order, bc_order, num_nodes, xtest):
    #Returns a loss function that dynamically retrieves L_batch for each batch.
    def loss_fn(ytrue, ypred):
        batch_size = tf.shape(ytrue)[0]  # Get current batch size
        batch_indices = tf.range(batch_size)  # Generate indices for batch
        L_batch = tf.gather(xtest[:, 0], batch_indices)  # Extract L_batch dynamically
        return _loss(ytrue, ypred, L_batch, source_order, bc_order, num_nodes)

    return loss_fn

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

def legendre_expansion(moments, bc_order, num_groups):
    moments = tf.reshape(moments, [-1, bc_order, num_groups]) #shape [batch, bc_order, num_groups]
    weights = tf.cast(tf.expand_dims(((2*tf.range(bc_order)+1)/2), -1), tf.float32)
    weighted_moments = weights*moments  #shape [batch, bc_order, num_groups]
    #direction does not matter since this is only for loss
    angular_flux = tf.matmul(tf.transpose(weighted_moments, perm = [0,2,1]), legendre[:,:num_ordinates//2])  #shape [batch, num_groups, num_ordinates//2]
    angular_flux = tf.transpose(angular_flux, perm = [0,2,1])  #shape [batch, num_ordinates//2, num_groups]
    return angular_flux

def _loss(ytrue, ypred, L_batch, s_order, b_order, num_nodes):
    """
    Computes L2 loss on the Gauss-Legendre integral of the reconstructed spectrum.
    
    - y_true, y_pred: Coefficients to reconstruct the spectrum (shape: [batch, num_coeffs])
    - L_batch: Tensor of shape [batch] containing different L values for each sample.
    - order, num_nodes: Integers used in spectrum reconstruction.
    
    Returns:
    - Scalar loss (L2 norm of integral error between reconstructed spectra).
    """
    # Gauss-Legendre Quadrature (same method as before)
    xi, wi = roots_legendre(num_nodes)
    xi = tf.convert_to_tensor(xi, dtype=tf.float32) 
    wi = tf.convert_to_tensor(wi, dtype=tf.float32)
    L_batch = tf.cast(tf.reshape(L_batch, [-1, 1]), tf.float32)  
    x_batch = (L_batch / 2) * (xi + 1) 
    phi     = groupwise_fourier_expansion(ytrue[:,bc_data_in_y:], s_order, num_nodes, num_groups, x_batch)  
    phi_pred = groupwise_fourier_expansion(ypred[:,bc_data_in_y:], s_order, num_nodes, num_groups, x_batch)  

    # Evaluate loss separately for each group
    phi_loss = 0
    #for i in range(num_groups):
    #    phi_integral     = tf.matmul(phi[:,:,i],     tf.expand_dims(wi, axis=-1))  
    #    phi_pred_integral = tf.matmul(phi_pred[:,:,i], tf.expand_dims(wi, axis=-1))  
    #    phi_loss += tf.reduce_mean(tf.square(phi_integral - phi_pred_integral)) 

        # Compute L2 norm of integral error

    phi_loss += tf.reduce_mean(tf.square(phi - phi_pred)) 
    #for some reason this works better than the commented groupwise method?

    bc_loss = 0
    # bc_l     = legendre_expansion(ytrue[:,:bc_data_in_y//2], b_order, num_groups)
    # bc_l_pred = legendre_expansion(ypred[:,:bc_data_in_y//2], b_order, num_groups)
    # bc_loss += tf.reduce_mean(tf.square(bc_l - bc_l_pred)) 

    # bc_r     = legendre_expansion(ytrue[:,bc_data_in_y//2:bc_data_in_y], b_order, num_groups)
    # bc_r_pred = legendre_expansion(ypred[:,bc_data_in_y//2:bc_data_in_y], b_order, num_groups)
    # bc_loss += tf.reduce_mean(tf.square(bc_r - bc_r_pred)) 
    
    total_loss = phi_loss + bc_loss

    return total_loss

# hp tuning
BEST_PARAMS_FILE = "models/tf_nn_hps.npy"
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

        #model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        model.compile(optimizer=Adam(learning_rate=lr), 
                        loss=flux_loss_wrapper(source_order, bc_order, num_nodes, xtest), 
                        metrics=['mae'])

        history = model.fit(xtrain, 
                ytrain, 
                epochs=int(epochs/10), 
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

model = Sequential()
model.add(Input(shape=(xtrain.shape[1],)))
for _ in range(best_params['num_layers']):
    model.add(Dense(best_params['num_layer_nodes'], activation='relu'))
model.add(Dense(ytrain.shape[1], activation='linear'))
model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), 
                            loss=flux_loss_wrapper(source_order, bc_order, num_nodes, xtest),
                            metrics=['mae'])
history = model.fit(
                xtrain, 
                ytrain, 
                epochs=epochs, 
                batch_size=best_params['batch_size'], 
                validation_data=(xtest, ytest),
                verbose = 1)

# learning curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.yscale("log")
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.yscale("log")
plt.title('Learning Curve')
plt.legend()
plt.grid()
plt.savefig("charts/ml/tf_nn_learning_curve.png")
plt.close()

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
df_pred.to_csv("ypred.csv",index = False)
l2_errors = np.linalg.norm(ytest - ypred, axis=1)
print(f"L2 Error on Output: {np.mean(l2_errors):.5f}")

# dataset L2 error
phi     = sum(fourier_expansion(ytest[:, bc_data_in_y:],source_order,num_nodes,num_groups))
phi_hat = sum(fourier_expansion(ypred[:, bc_data_in_y:],source_order,num_nodes,num_groups))

L2_phi = np.mean(np.linalg.norm(phi - phi_hat, axis = 1))
print(f"L2 error on scalar flux = {np.round(L2_phi,5)}")

# residuals
residuals = np.abs(ytest[:, bc_data_in_y:] - ypred[:, bc_data_in_y:])
plt.figure()
plt.plot(residuals[:, 0], label="A0")
plt.plot(residuals[:, 1], label="A1")
plt.plot(residuals[:, 2], label="B1")
plt.title("Testing Residuals for Fourier Coeffs")
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.yscale("log")
plt.legend()
plt.savefig("charts/ml/tf_residuals.png")
plt.close()

best_idx = np.argmin(np.abs(residuals), axis=0)[0]
worst_idx = np.argmax(np.abs(residuals), axis=0)[0]
print(f"Best Index: {best_idx}, Worst Index: {worst_idx}")
x = np.linspace(0, 1, num_nodes)  # Define x in the same way as fourier_expansion

def plot_flux(x, t, ytest, ypred, flux_order, num_nodes, idx, title):
    """Plot expansion and TT solution."""
    true_fluxes = fourier_expansion(ytest, flux_order, num_nodes, num_groups)
    pred_fluxes = fourier_expansion(ypred, flux_order, num_nodes, num_groups)
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
        plt.title(f"{title} Group {g+1}")
        plt.grid()
        plt.savefig(f"charts/ml/tf_ytest_ypred_{idx}_group{g+1}.png")
        plt.close()

# print(ytest[best_idx, bc_data_in_y:])  # ytest coeffs
# print("\n\n")
# print(ypred[best_idx, bc_data_in_y:])   # ypred_coeffs


plot_flux(
    x,
    a,   # thickness
    ytest[best_idx, bc_data_in_y:],  # ytest coeffs
    ypred[best_idx, bc_data_in_y:],   # ypred_coeffs
    source_order,
    num_nodes,
    best_idx,
    f"Y-Test and Y-Pred vs x, Best Index"
)

plot_flux(
    x,
    a,   # thickness
    ytest[worst_idx, bc_data_in_y:],  # ytest coeffs
    ypred[worst_idx, bc_data_in_y:],   # ypred_coeffs
    source_order,
    num_nodes,
    worst_idx,
    f"Y-Test and Y-Pred vs x, Worst Index"
)