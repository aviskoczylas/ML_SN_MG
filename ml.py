import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from scipy.special import roots_legendre

#TODO: fix data read in
#make sure fourier expansion is being used correctly (correct coeffs/order passed in)

num_groups = 8
epochs = 50
num_hp_trials = 20
source_order = 4
bc_order = 3
bc_data_in_y = 2*(bc_order+1)*num_groups
# load data
#xdata = pd.read_csv("../data/xdata.csv")
#ydata = pd.read_csv("../data/ydata.csv")
xdata = pd.read_csv("data/xtrain_25000.csv")
ydata = pd.read_csv("data/ytrain_25000.csv")
#xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(xdata, ydata, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
#xtrain = pd.read_csv("../data/xtrain.csv")
#ytrain = pd.read_csv("../data/ytrain.csv")
xtrain = xtrain.to_numpy()
ytrain = ytrain.to_numpy()
#xtest = pd.read_csv("../data/xtest.csv").to_numpy()
#ytest = pd.read_csv("../data/ytest.csv").to_numpy()

# scale data
xscaler = MinMaxScaler()
xtrain = xscaler.fit_transform(xtrain)
xtest = xscaler.transform(xtest)
yscaler = MinMaxScaler()
ytrain = yscaler.fit_transform(ytrain)
ytest = yscaler.transform(ytest)

# search space
param_space = {
    'learning_rate': np.random.uniform(0.0001, 0.001, 10).tolist(),
    'num_layers': np.random.randint(1, 6, 10).tolist(),
    'num_nodes': [16, 32, 64, 128, 256],
    'batch_size': [16, 32, 64, 128]
}

def flux_loss_wrapper(order, num_nodes, xtest):
    #Returns a loss function that dynamically retrieves L_batch for each batch.
    def loss_fn(ytrue, ypred):
        batch_size = tf.shape(ytrue)[0]  # Get current batch size
        batch_indices = tf.range(batch_size)  # Generate indices for batch
        L_batch = tf.gather(xtest[:, 0], batch_indices)  # Extract L_batch dynamically

        return _loss(ytrue, ypred, L_batch, order, num_nodes)

    return loss_fn

def groupwise_fourier_expansion(coeffs, order, num_nodes, num_groups, x):
    #groupwise data is initially flattened into coeffs
    x = tf.reshape(x, [1, num_nodes, 1])  # Shape: [num_nodes] -> [1, num_nodes, 1]
    coeffs = tf.reshape(coeffs, [coeffs.shape[0], order*2+1, num_groups])
    # First coefficient (broadcasted across x)
    expansion = tf.expand_dims(coeffs[:, 0, :], axis=1)  # Shape: [batch, 1, num_groups]

    for k in range(order):
        cos_term = tf.expand_dims(coeffs[:, 2 * k + 1, :], axis=1) * tf.cos(np.pi * (k + 1) * x)
        sin_term = tf.expand_dims(coeffs[:, 2 * k + 2, :], axis=1) * tf.sin(np.pi * (k + 1) * x)
        expansion += cos_term + sin_term

    return expansion  # Shape: [batch, num_nodes, num_groups]

def _loss(ytrue, ypred, L_batch, order, num_nodes):
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
    L_batch = tf.reshape(L_batch, [-1, 1])  
    x_batch = (L_batch / 2) * (xi + 1) 
     
    # Reconstruct spectra from Fourier coefficients
    phi     = groupwise_fourier_expansion(ytrue[bc_data_in_y:], order, num_nodes, num_groups, x_batch)  
    phi_pred = groupwise_fourier_expansion(ypred[bc_data_in_y:], order, num_nodes, num_groups, x_batch)  

    # Evaluate loss separately for each group
    total_loss = 0
    for i in range(num_groups):
        phi_integral     = tf.matmul(phi[:,:,i],     tf.expand_dims(wi, axis=-1))  
        phi_pred_integral = tf.matmul(phi_pred[:,:,i], tf.expand_dims(wi, axis=-1))  
        # Compute L2 norm of integral error
        total_loss += tf.reduce_mean(tf.square(phi_integral - phi_pred_integral))

    return total_loss

# hp tuning
BEST_PARAMS_FILE = "../models/tf_nn_hps.npy"
if os.path.exists(BEST_PARAMS_FILE):
    best_params = np.load(BEST_PARAMS_FILE, allow_pickle=True).item()
    print("Loaded best hyperparameters:", best_params)
else:
    best_loss = float("inf")
    best_params = {}

    for _ in range(num_hp_trials):  
        lr = np.random.choice(param_space['learning_rate'])
        num_layers = np.random.choice(param_space['num_layers'])
        num_nodes = np.random.choice(param_space['num_nodes'])
        batch_size = np.random.choice(param_space['batch_size'])

        # Build model----------------------------------------------------------------------------------------
        model = Sequential()
        model.add(Input(shape=(xtrain.shape[1],)))
        for _ in range(num_layers):
            model.add(Dense(num_nodes, activation='relu'))
        model.add(Dense(ytrain.shape[1], activation='linear'))
#-------------------------------------------------

        #model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        model.compile(optimizer=Adam(learning_rate=lr), 
                        loss=flux_loss_wrapper(source_order, num_nodes, xtest), 
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
                    'num_nodes': num_nodes, 
                    'batch_size': batch_size}
    # Save 
    np.save(BEST_PARAMS_FILE, best_params)
    print("Saved best hyperparameters:", best_params)

model = Sequential()
model.add(Input(shape=(xtrain.shape[1],)))
for _ in range(best_params['num_layers']):
    model.add(Dense(best_params['num_nodes'], activation='relu'))
model.add(Dense(ytrain.shape[1], activation='linear'))
model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), 
                            loss=flux_loss_wrapper(source_order, num_nodes, xtest),
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
plt.savefig("../charts/ml/tf_nn_learning_curve.png")
plt.close()

#broadcasting issue?
def group_sum_fourier_expansion(coeffs,order,num_nodes, num_groups):
    x = np.linspace(0, 1, num_nodes)
    coeffs = np.reshape(coeffs, [order*2+1, num_groups])
    # Compute source by Fourier expansion
    return coeffs[0, :] + sum(
        coeffs[2 * k + 1, g] * np.cos(np.pi * (k + 1) * x)
        + coeffs[2 * k + 2, g] * np.sin(np.pi * (k + 1) * x)
        for k in range(order) for g in range(num_groups))

#L2 norm on coeffs
ypred = model.predict(xtest)
ypred_transformed = yscaler.inverse_transform(ypred)
ytest_transformed = yscaler.inverse_transform(ytest)

df_pred = pd.DataFrame(ypred_transformed)
df_pred.to_csv("ypred.csv",index = False)
l2_errors = np.linalg.norm(ytest_transformed - ypred_transformed, axis=1)
print(f"L2 Error on Output: {np.mean(l2_errors):.5f}")

# dataset L2 error
phi = np.zeros((ytest.shape[0],num_nodes))
phi_hat = np.zeros_like(phi)

for ind in range(ytest.shape[0]):
    phi[ind,:]     = group_sum_fourier_expansion(ytest[bc_data_in_y:],source_order,num_nodes,num_groups)
    phi_hat[ind,:] = group_sum_fourier_expansion(ypred[bc_data_in_y:],source_order,num_nodes,num_groups)

L2_phi = np.linalg.norm(phi - phi_hat)
print(f"L2 norm on the scalar flux = {np.round(L2_phi,5)}")

# residuals
residuals = np.abs(ytest[bc_data_in_y:] - ypred[bc_data_in_y:])
plt.figure()
plt.plot(residuals[:, 0], label="A0")
plt.plot(residuals[:, 1], label="A1")
plt.plot(residuals[:, 2], label="B1")
plt.title("Testing Residuals for Fourier Coeffs")
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.yscale("log")
plt.savefig("../charts/ml/tf_residuals.png")
plt.close()

best_idx = np.argmin(np.abs(residuals), axis=0)[0]
worst_idx = np.argmax(np.abs(residuals), axis=0)[0]
print(f"Best Index: {best_idx}, Worst Index: {worst_idx}")
x = np.linspace(0, 1, num_nodes)  # Define x in the same way as fourier_expansion

def plot_flux(x, t, ytest, ypred, flux_order, num_nodes, idx, title):
    """Plot expansion and TT solution."""
    plt.clf()
    plt.plot(
        t * x,
        group_sum_fourier_expansion(ytest[bc_data_in_y:], flux_order, num_nodes, num_groups),
        label="Y-Test",
    )
    plt.plot(
        t * x,
        group_sum_fourier_expansion(ypred[bc_data_in_y:], flux_order, num_nodes, num_groups),
        label="Y-Pred",
    )
    plt.xlabel("$x$")
    plt.ylabel("$\\phi(x)$")
    plt.legend()
    plt.title(title)
    plt.savefig(f"../charts/ml/tf_ytest_ypred_{idx}.png")
    plt.close()

plot_flux(
    x,
    xtest[best_idx, 0],   # thickness
    ytest[best_idx, bc_data_in_y:],  # ytest coeffs
    ypred[best_idx, bc_data_in_y:],   # ypred_coeffs
    source_order,
    num_nodes,
    best_idx,
    f"Y-Test and Y-Pred vs x, Best Index"
)

plot_flux(
    x,
    xtest[worst_idx, 0],   # thickness
    ytest[worst_idx, bc_data_in_y:],  # ytest coeffs
    ypred[worst_idx, bc_data_in_y:],   # ypred_coeffs
    source_order,
    num_nodes,
    worst_idx,
    f"Y-Test and Y-Pred vs x, Worst Index"
)
