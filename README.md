# Machine Learning Applied to $S_n$ Transport 

Ravi Shastri, Avi Skoczylas, Brian Kiedrowski

Regions, cross sections and boundary conditions are generated.
A 1d $S_n$ solver is used to provide determinisic solutions for the problem with a fixed source.
A machine learning model is trained on the inputs and outputs of the solver and generates flux predictions

# Requirements:
  * Python=3.11.0
  * numpy=2.0.2
  * pandas=2.2.3
  * matplotlib
  * scipy=1.15.1
  * numba=0.62.1
  * seaborn
  * tensorflow=2.18.0
  * scikit-learn=1.7.2

# Usage:
## sn_solver_fixed_source.py:
  * $S_N$ class
    * Generates problem geometry and cross sections if not provided up to specified order
    * Defines fixed source using spatial and groupwise fourier coeficients
  * Hydrogen ratio and boundary conditions are randomized for each solve
  * Solves transport in 1D using discrete ordinates method
    * Outer loop for group-to-group scattering, inner loop for within-group scattering
    * Uses Numba to enhance performance
  * Outputs xs, sources, BCs into csv to be used as xtrain data
  * Outputs moments to be used as ytrain data
  * Plots scalar and angular group flux profiles
  * Other sn_solver files are variations of this solver

## ml.py:
  * Creates neural network using Tensorflow with custom loss function
    * Loss function reconstructs groupwise spectra from fourier coefficients
    * The loss is the sum of the L2 error for each group's reconstruction 
  * Searches for optimal hyperparameters and saves them to np file
  * Trains on input dataset containing xs, sources, and BCs
  * Produces prediction of group flux moments
  * Plots error metrics
    * Residuals
    * Flux reconstruction for best and worst indices
    * Model learning curve
    
Completed Work:
  * Implemented multiple types of $S_n$ solvers
  * Generated training data
  * Trained machine learning model to produce output with MSE error of $< 0.1$

Current work: 
  * Refine machine learning model, particularly the loss function, to reduce error
  * Reduce amount of input data required for single region problems

Future Work:
  * Perform a machine learning sweep to apply results to multi region problems
  * Extend results to higher orders of anisotropy
