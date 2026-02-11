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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quieter logs

SEED = 42
use_stored_model = False
use_hps = True
a = 1 #thickness, not given in xdata
num_groups = 8
num_nodes = 100
num_ordinates = 64
epochs = 250
num_hp_trials = 100
source_order = 4
bc_order = 4
bc_data_in_y = 2*bc_order*num_groups
eps = 1e-10
verbosity = 1 # 0 means not verbose, 1 means verbose
MODEL_FILE = "models/best_model_gpu.model.keras"

np.random.seed(SEED)      # NumPy random
tf.random.set_seed(SEED)  # TensorFlow random

# ---- GPU setup ----
gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs Available:", len(gpus))
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")  # pick GPU:0
        logical = tf.config.list_logical_devices("GPU")
        print("Using GPU:", logical[0].name if logical else "GPU:0")
    except Exception as e: print("GPU selection warning:", e)
else:
    print("WARNING: No GPU found. This will run on CPU.")


# load data
xtrain = pd.read_csv("data/xtrain_25000.csv").to_numpy()
ytrain = pd.read_csv("data/ytrain_25000.csv").to_numpy()
xtest = pd.read_csv("data/xtest_8500.csv").to_numpy()
ytest = pd.read_csv("data/ytest_8500.csv").to_numpy()

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

xscaler = MinMaxScaler()
xtrain = xscaler.fit_transform(xtrain)
xtest = xscaler.transform(xtest)
#yscaler = MinMaxScaler()
#ytrain = yscaler.fit_transform(ytrain)
#ytest = yscaler.transform(ytest)

# tensorflow tensor
xtrain = xtrain.astype("float32")
xtest  = xtest.astype("float32")
ytrain = ytrain.astype("float32")
ytest  = ytest.astype("float32")

def make_ds(x, y, batch_size, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(x), 20000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

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
legendre = tf.constant(legendre,dtype=tf.float32) # gpu compatibility
#legendre = tf.convert_to_tensor(legendre, dtype=tf.float32)

def legendre_expansion(moments, bc_order, num_groups, dir):
    moments = tf.reshape(moments, [-1, bc_order, num_groups]) #shape [batch, bc_order, num_groups]
    weights = tf.cast(tf.expand_dims(((2*tf.range(bc_order)+1)/2), -1), tf.float32)
    weighted_moments = weights*moments  #shape [batch, bc_order, num_groups]
    if dir == -1:
        legendre_values = legendre[:,:num_ordinates//2]
    elif dir == 1:
        legendre_values = legendre[:,num_ordinates//2:]
    #shape [batch, num_groups, num_ordinates//2]
    angular_flux = tf.matmul(tf.transpose(weighted_moments, perm = [0,2,1]), legendre_values)  
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
    _xi, _wi = roots_legendre(num_nodes)
    xi = tf.constant(_xi, dtype=tf.float32)
    wi = tf.constant(_wi, dtype=tf.float32)
    #xi, wi = roots_legendre(num_nodes)
    #xi = tf.convert_to_tensor(xi, dtype=tf.float32) 
    #wi = tf.convert_to_tensor(wi, dtype=tf.float32)
    x_batch = (a / 2) * (xi + 1) 
    phi      = groupwise_fourier_expansion(ytrue[:, bc_data_in_y:], source_order, num_nodes, num_groups, x_batch)
    phi_pred = groupwise_fourier_expansion(ypred[:, bc_data_in_y:], source_order, num_nodes, num_groups, x_batch)

    phi_loss = tf.reduce_mean(tf.square(phi - phi_pred))

    bc_l      = legendre_expansion(ytrue[:, :bc_data_in_y//2], bc_order, num_groups, -1)
    bc_l_pred = legendre_expansion(ypred[:, :bc_data_in_y//2], bc_order, num_groups, -1)
    bc_r      = legendre_expansion(ytrue[:, bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)
    bc_r_pred = legendre_expansion(ypred[:, bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)

    bc_loss = tf.reduce_mean(tf.square(bc_l - bc_l_pred)) + tf.reduce_mean(tf.square(bc_r - bc_r_pred))
    return phi_loss + bc_loss

#    phi     = groupwise_fourier_expansion(ytrue[:,bc_data_in_y:], source_order, num_nodes, num_groups, x_batch)  
#    phi_pred = groupwise_fourier_expansion(ypred[:,bc_data_in_y:], source_order, num_nodes, num_groups, x_batch)  
#
#    # Evaluate loss separately for each group
#    phi_loss = 0
#    phi_loss += tf.reduce_mean(tf.square(phi - phi_pred)) 
#
#    bc_loss = 0
#    bc_l     = legendre_expansion(ytrue[:,:bc_data_in_y//2], bc_order, num_groups, -1)
#    bc_l_pred = legendre_expansion(ypred[:,:bc_data_in_y//2], bc_order, num_groups, -1)
#    bc_loss += tf.reduce_mean(tf.square(bc_l - bc_l_pred)) 
#
#    bc_r     = legendre_expansion(ytrue[:,bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)
#    bc_r_pred = legendre_expansion(ypred[:,bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)
#    bc_loss += tf.reduce_mean(tf.square(bc_r - bc_r_pred)) 
#
#    total_loss = phi_loss + bc_loss
#
#    return total_loss

# hp tuning
BEST_PARAMS_FILE = "models/tf_nn_hps.npy"
if os.path.exists(BEST_PARAMS_FILE) and use_hps:
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
        params = {'learning_rate': lr, 
                    'num_layers': num_layers, 
                    'num_layer_nodes': num_layer_nodes, 
                    'batch_size': batch_size}
        print("Params:", params)
        model = Sequential()
        model.add(Input(shape=(xtrain.shape[1],)))
        for _ in range(num_layers):
            model.add(Dense(num_layer_nodes, activation='relu'))
        model.add(Dense(ytrain.shape[1], activation='linear'))

        model.compile(optimizer=Adam(learning_rate=lr), 
                        loss=loss_func, 
                        metrics=['mae'])

        train_ds = make_ds(xtrain, ytrain, batch_size, training=True)
        val_ds   = make_ds(xtest,  ytest,  batch_size, training=False)
        history = model.fit(train_ds,
                            epochs=int(epochs/10),
                            validation_data=val_ds,
                            verbose=verbosity,
                            )
#        history = model.fit(xtrain, 
#                ytrain, 
#                epochs=int(epochs/10), 
#                batch_size=batch_size, 
#                validation_data=(xtest, ytest), 
#                verbose=1,)

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

    train_ds = make_ds(xtrain, ytrain, best_params["batch_size"], training=True)
    val_ds   = make_ds(xtest,  ytest,  best_params["batch_size"], training=False)
    history = model.fit(train_ds,
                        epochs=epochs,
                        validation_data=val_ds,
                        callbacks = [model_checkpoint_callback],
                        verbose=verbosity,
                        )
#    history = model.fit(
#                    xtrain, 
#                    ytrain, 
#                    epochs=epochs, 
#                    batch_size=best_params['batch_size'], 
#                    validation_data=(xtest, ytest),
#                    callbacks = [model_checkpoint_callback],
#                    verbose = 1)

    # learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.yscale("log")
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    plt.savefig("charts/ml/tf_nn_learning_curve.png")
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
    return np.array(flux)

#L2 norm on coeffs
ypred = model.predict(xtest)

df_pred = pd.DataFrame(ypred)
df_pred.to_csv("ypred.csv",index = False)
#l2_errors = np.linalg.norm(ytest - ypred, axis=1)
#print(f"L2 Error on Output: {np.mean(l2_errors):.5f}")

# dataset L2 error
phi     = fourier_expansion(ytest[:, bc_data_in_y:],source_order,num_nodes,num_groups)
phi_hat = fourier_expansion(ypred[:, bc_data_in_y:],source_order,num_nodes,num_groups)
bc_l = np_legendre(ytest[:,:bc_data_in_y//2], bc_order, num_groups, -1)
bc_l_hat = np_legendre(ypred[:,:bc_data_in_y//2], bc_order, num_groups, -1)
bc_r = np_legendre(ytest[:,bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)
bc_r_hat = np_legendre(ypred[:,bc_data_in_y//2:bc_data_in_y], bc_order, num_groups, 1)

# per group data analysis
# datasets are Groups x Samples x Nodes
x = np.linspace(0, 1, num_nodes)
for g in range(num_groups):
    phi_g     = phi[g]
    phi_hat_g = phi_hat[g]
    bc_l_g     = bc_l[g]
    bc_l_hat_g = bc_l_hat[g]
    bc_r_g     = bc_r[g]
    bc_r_hat_g = bc_r_hat[g]

    # L2 norms
    L2_phi_group  = np.linalg.norm(phi_g - phi_hat_g,ord=2)
    L2_bcl_group  = np.linalg.norm(bc_l_g - bc_l_hat_g,ord=2)
    L2_bcr_group  = np.linalg.norm(bc_r_g - bc_r_hat_g,ord=2)
    print(f"L2_phi (group {g}) = {L2_phi_group:.5e}")
    print(f"L2_lb  (group {g}) = {L2_bcl_group:.5e}")
    print(f"L2_rb  (group {g}) = {L2_bcr_group:.5e}")

    # percent errors
    phi_percent_errors  = 100.0 * np.abs(phi_g  - phi_hat_g)  / (np.abs(phi_g)  + eps)
    bc_l_percent_errors = 100.0 * np.abs(bc_l_g - bc_l_hat_g) / (np.abs(bc_l_g) + eps)
    bc_r_percent_errors = 100.0 * np.abs(bc_r_g - bc_r_hat_g) / (np.abs(bc_r_g) + eps)
    phi_score  = np.mean(phi_percent_errors,  axis=1)  
    bcl_score  = np.mean(bc_l_percent_errors, axis=1)
    bcr_score  = np.mean(bc_r_percent_errors, axis=1)

    # the kth worst prediction
#    phi_rank  = np.argsort(phi_score)[::-1]
#    bcl_rank  = np.argsort(bcl_score)[::-1]
#    bcr_rank  = np.argsort(bcr_score)[::-1]

    print(f"\nBest Predictions per Group {g}")
    best_phi_ind  = np.argmin(phi_score)
    best_bc_l_ind = np.argmin(bcl_score)
    best_bc_r_ind = np.argmin(bcr_score)
    print(best_phi_ind, best_bc_l_ind, best_bc_r_ind)
    print(phi_score[best_phi_ind], bcl_score[best_bc_l_ind], bcr_score[best_bc_r_ind])

    print(f"\nWorst Predictions per Group {g}")
    worst_phi_ind  = np.argmax(phi_score)
    worst_bc_l_ind = np.argmax(bcl_score)
    worst_bc_r_ind = np.argmax(bcr_score)
    print(worst_phi_ind, worst_bc_l_ind, worst_bc_r_ind)
    print(phi_score[worst_phi_ind], bcl_score[worst_bc_l_ind], bcr_score[worst_bc_r_ind])

    # phi plot
    plt.figure(figsize=(8,6))
    plt.plot(x,phi_hat_g[best_phi_ind,:],label="phi-Pred")
    plt.plot(x,phi_g[best_phi_ind,:],label="phi-Test")
    plt.legend()
    plt.grid(which='Both')
    plt.ylabel(fr"$\phi(x,g={g+1})$")
    plt.xlabel("x")
    plt.title(f"Group {g+1} Best Prediction")
    plt.savefig(f"charts/ml/best_phi_g{g+1}.png")
    plt.clf()
    plt.close()

    plt.figure(figsize=(8,6))
    #plt.plot(x,phi_hat_g[worst_phi_ind,:]/ np.sum(phi_hat_g[worst_phi_ind,:]),label="phi-Pred")
    #plt.plot(x,phi_g[worst_phi_ind,:] / np.sum(phi_g[worst_phi_ind,:]),label="phi-Test")
    plt.plot(x,phi_hat_g[worst_phi_ind,:],label="phi-Pred")
    plt.plot(x,phi_g[worst_phi_ind,:],label="phi-Test")
    plt.legend()
    plt.grid(which='Both')
    plt.ylabel(fr"$\phi(x,g={g+1})$")
    plt.xlabel("x")
    plt.title(f"Group {g+1} Worst Prediction")
    plt.savefig(f"charts/ml/worst_phi_g{g+1}.png")
    plt.clf()
    plt.close()

    # bc_l plot
    plt.figure(figsize=(8,6))
    plt.plot(
        quadrature_points[:num_ordinates//2],
        (bc_l_hat_g[best_bc_l_ind,:]),
        label=r"$\psi^{Pred}$",
    )
    plt.plot(
        quadrature_points[:num_ordinates//2],
        (bc_l_g[best_bc_l_ind,:]),
        label=r"$\psi^{Test}$",
    )
    title = fr"$\psi^l(\mu,g={g+1})$"
    plt.xlabel("$\\mu$")
    plt.ylabel("$\\psi(\\mu)$")
    plt.legend()
    plt.title(f"Best {title} vs $\\mu$, Group {g+1}, Left Boundary")
    plt.grid()
    plt.savefig(f"charts/ml/best_psi_l_g{g+1}.png")
    plt.clf()
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(
        quadrature_points[:num_ordinates//2],
        (bc_l_hat_g[worst_bc_l_ind,:]),
        label=r"$\psi^{Pred}$",
    )
    plt.plot(
        quadrature_points[:num_ordinates//2],
        (bc_l_g[worst_bc_l_ind,:]),
        label=r"$\psi^{Test}$",
    )
    plt.xlabel("$\\mu$")
    plt.ylabel("$\\psi(\\mu)$")
    plt.legend()
    plt.title(f"Worst {title} vs $\\mu$, Group {g+1}")
    plt.grid()
    plt.savefig(f"charts/ml/worst_psi_l_g{g+1}.png")
    plt.clf()
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(
        quadrature_points[:num_ordinates//2],
        (bc_l_hat_g[best_bc_r_ind,:]),
        label="psi-Pred",
    )
    plt.plot(
        quadrature_points[:num_ordinates//2],
        (bc_l_g[best_bc_r_ind,:]),
        label="psi-Test",
    )
    title = fr"$\psi^r(\mu,g={g+1})$"
    plt.xlabel("$\\mu$")
    plt.ylabel("$\\psi(\\mu)$")
    plt.legend()
    plt.title(f"Best {title} vs $\\mu$, Group {g+1}")
    plt.grid()
    plt.savefig(f"charts/ml/best_psi_r_g{g+1}.png")
    plt.clf()
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(
        quadrature_points[:num_ordinates//2],
        (bc_l_hat_g[worst_bc_r_ind,:]),
        label="psi-Pred",
    )
    plt.plot(
        quadrature_points[:num_ordinates//2],
        (bc_l_g[worst_bc_r_ind,:]),
        label="psi-Pred",
    )
    plt.xlabel("$\\mu$")
    plt.ylabel("$\\psi(\\mu)$")
    plt.legend()
    plt.title(f"Worst {title} vs $\\mu$, Group {g+1}, Right Boundary")
    plt.grid()
    plt.savefig(f"charts/ml/worst_psi_r_g{g+1}.png")
    plt.clf()
    plt.close()

    # histogram data
    data_cap = 20
    phi_data = np.mean(phi_percent_errors,axis=1)
    num_pts_above_cap = np.sum(phi_data > data_cap)
    print(f"{np.round(100 * num_pts_above_cap/phi_data.size,5)}% of Phi Samples Above {data_cap}% Error, g={g+1}")
    capped_data = np.where(phi_data > data_cap, data_cap, phi_data)
    bins = np.arange(0, data_cap + 1, data_cap/100)
    plt.hist(capped_data, bins=bins, edgecolor='black')
    plt.xlabel('% Error')
    plt.ylabel('Number of Samples')
    plt.xlim([0,data_cap+1])
    plt.title(fr'Sample % Errors, $\phi(x)$, g={g+1}')
    plt.grid(which='Both')
    plt.savefig(f"charts/ml/full_hist_phi_g{g+1}.png")
    plt.close()

    bc_data = np.mean(bc_l_percent_errors,axis=1)
    num_pts_above_cap = np.sum(bc_data > data_cap)
    print(f"\n{np.round(100 * num_pts_above_cap/bc_data.size,5)}% of Psi_l Samples Above {data_cap}% Error, g={g+1}")
    capped_data = np.where(bc_data > data_cap, data_cap, bc_data)
    bins = np.arange(0, data_cap + 1, data_cap/100)
    plt.hist(capped_data, bins=bins, edgecolor='black')
    plt.xlabel('% Error')
    plt.ylabel('Number of Samples')
    plt.xlim([0,data_cap+1])
    plt.title(fr'Sample % Errors, $\psi^l(x,\mu)$, g={g+1}')
    plt.grid(which='Both')
    plt.savefig(f"charts/ml/full_hist_psil_g{g+1}.png")
    plt.close()

    bc_data = np.mean(bc_r_percent_errors,axis=1)
    num_pts_above_cap = np.sum(bc_data > data_cap)
    print(f"{np.round(100 * num_pts_above_cap/bc_data.size,5)}% of Psi_r Samples Above {data_cap}% Error, g={g+1}")
    capped_data = np.where(bc_data > data_cap, data_cap, bc_data)
    bins = np.arange(0, data_cap + 1, data_cap/100)
    plt.hist(capped_data, bins=bins, edgecolor='black')
    plt.xlabel('% Error')
    plt.ylabel('Number of Samples')
    plt.xlim([0,data_cap+1])
    plt.title(fr'Sample % Errors, $\psi^r(x,\mu)$, g={g+1}')
    plt.grid(which='Both')
    plt.savefig(f"charts/ml/full_hist_psir_g{g+1}.png")
    plt.close()

print("End time: "+str(time.time()-start))
