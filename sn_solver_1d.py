import time
start_time = time.time()
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import lpmv
print(time.time()-start_time)

#TODO implement vacuum bc
#TODO Write fixed source as Fourier Expansion
#TODO parallelize:
#random numbers for s2, s3.

num_ordinates = 16
dx = 0.5
assert np.abs(1/dx-int(1/dx)) < 1e-6 #each cell must be evenly split
data_dir  = "C:/Users/abrah/Downloads"
data_path = os.path.join(data_dir, "xs_coarse_lattice.csv")
data = np.array(pd.read_csv(data_path))
cell_index = data[:,0]
groups = data[:,1]
num_groups = round(groups[-1])
num_cells = round(cell_index[-1])
total_xs = data[:,2]
total_xs = np.reshape(total_xs,(num_cells, num_groups))
scatter_xs = data[:,3]
scatter_xs = np.reshape(scatter_xs,(num_cells, num_groups))
fission_abs_xs = data[:,4]
fission_abs_xs = np.reshape(fission_abs_xs,(num_cells, num_groups))
#since fission neutrons are only deposited in group 0, we can get rid of the data for other groups, it's all 0. 
fission_prod_xs = data[::8,5]
#rearrange to be in terms of dx rather than cell number for ease of use in calculating fission source
fission_prod_xs = np.repeat(fission_prod_xs, round(1/dx))
g2g_p0_scatter_xs = data[:,6:14]
g2g_p0_scatter_xs = np.reshape(g2g_p0_scatter_xs,(num_cells, num_groups, num_groups))
#form [cell num, current group, group scattered into]
g2g_p1_scatter_xs = data[:,14:]
g2g_p1_scatter_xs = np.reshape(g2g_p1_scatter_xs,(num_cells, num_groups, num_groups))

scatter_matrices = [g2g_p0_scatter_xs, g2g_p1_scatter_xs]

neutrons_per_fission = 2.5
num_sections = round(num_cells/dx)
order = 2 #actually this is order+1, but i don't feel like putting a +1 everywhere.
weight = np.array([0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
          0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541])

#define the gauss-legendre quadrature points
quadrature_points = np.array([0.0950125098376374, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438, 
               0.7554044083550030, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499,
               -0.0950125098376374, -0.2816035507792589, -0.4580167776572274, -0.6178762444026438, 
               -0.7554044083550030, -0.8656312023878318, -0.9445750230732326, -0.9894009349916499,])

 #precalculate the legendre polynomials for the quadrature up to the desired order
#this code assumes that all scattering and flux moments are given to the same order
legendre = np.zeros((order, num_ordinates))
for l in range(order):
    for n in range(num_ordinates):
        legendre[l, n] = lpmv(0, l, quadrature_points[n])
   
def in_group_scatter_expansion(group_flux_moments,g):
    source = np.zeros((num_sections, num_ordinates))
    cell_indices = np.floor(np.arange(num_sections)*dx).astype(int)
    for l in range(order):
        source+=((2*l+1)/2)*np.outer(scatter_matrices[l][cell_indices,g,g]*group_flux_moments[:, l],legendre[l,:])
    return source

def g2g_scatter_expansion(flux_moments, group_scattered_into):
    source = np.zeros((num_sections, num_ordinates))
    cell_indices = np.floor(np.arange(num_sections)*dx).astype(int)
    exclude_current_group = np.ones((num_sections, num_groups))
    exclude_current_group[:,group_scattered_into] = 0
    for l in range(order):
        outscatter_rate = np.sum(scatter_matrices[l][cell_indices, :,group_scattered_into]*flux_moments[:, l, :]*exclude_current_group, axis = 1) #summed across all other groups
        source+=((2*l+1)/2)*np.outer(outscatter_rate , legendre[l,:]) 
    return source

def converged(flux_m, flux_m_plus_1, loop):
    if flux_m is None:
        return False
    diff = np.linalg.norm(flux_m_plus_1 - flux_m)
    norm = np.linalg.norm(flux_m)
    E = 1e-5#1e-6
    l2 = diff / (norm+1e-12)#1e-12 prevents divide by 0
    if loop:
        print("L2,",loop, l2)

    return l2 < E

#reconstruct the flux by evaluating the legendre series at each ordinate
def moments_to_flux(flux_moments):
    weighted_moments = (2*np.arange(order)+1)/2*flux_moments
    angular_flux = weighted_moments @ legendre
    return angular_flux

#decontruct the flux into its legendre moments - should only be used on a partial flux, hence only direction 1 or -1
def flux_to_moments(angular_flux, direction):
    weighted_flux = weight*angular_flux
    if direction == 1: #mu > 0
        flux_moments = legendre[:,:num_ordinates//2] @ weighted_flux
    elif direction == -1: #mu < 0
        flux_moments = legendre[:,num_ordinates//2:] @ weighted_flux
    return flux_moments 

prev_fission_source, prev_g2g_source = None, None

reflecting = True
incoming_moments_l = [0.1,0] #for constant bc, this is a given. For reflecting BC, this is a guess
incoming_moments_r = [0.1,0] #only used for the case of constant bc
in_group_source = np.ones((num_sections, num_ordinates, num_groups))*0.01 #initial guess
g2g_source = np.ones((num_sections, num_ordinates, num_groups))*0.01#initial guess, source is summed over all incoming groups
fission_source = np.ones((num_sections, num_ordinates, num_groups))*0.01 #initial guess
flux_moments = np.zeros((num_sections, order, num_groups))
flux_moments[:, 0, :] = 0.01 #initial guess

left_edge_cell_flux = moments_to_flux(incoming_moments_l)[:num_ordinates//2] #initial guess if reflecting
k=1 #initial guess

#loop until everything converges
while not converged(prev_fission_source, fission_source, "outer loop: "): #outer iteration - fission source
    source = np.zeros((num_sections, num_ordinates, num_groups))
    if prev_fission_source is not None:
        k_new = k * np.sum(fission_source)/np.sum(prev_fission_source) #power iteration
        alpha = 0.7  #arbitrary underrelaxation coefficient to help converge
        k = alpha * k_new + (1 - alpha) * k
    print(k, np.sum(fission_source), np.min(fission_source), np.max(fission_source))
    prev_fission_source = fission_source.copy()
    #chi is approximated as only depositing neutrons in the highest energy group, so no need to iterate over groups
    fission_source = np.zeros((num_sections, num_ordinates, num_groups))
    fission_term = (1/k)*neutrons_per_fission*fission_prod_xs*flux_moments[:,0,0]
    fission_source[:,:,0] = np.transpose(np.tile(fission_term,(num_ordinates,1))) #fission source is isotropic, so it's applied to all ordinates
    while not converged(prev_g2g_source, g2g_source, "middle loop: "): #middle iteration - group to group scattering
        prev_g2g_source = g2g_source.copy()
        for group_scattered_into in range(num_groups):
            g2g_source[:,:,group_scattered_into] = g2g_scatter_expansion(flux_moments, group_scattered_into)
        for group in range(num_groups):
            group_flux_moments = np.zeros((num_sections, order))
            prev_group_flux = None
            while not converged(prev_group_flux, group_flux_moments, 0): #inner iteration - ingroup scattering and transport sweep
                prev_group_flux = group_flux_moments.copy()
                group_flux_moments = np.zeros((num_sections, order))
                #compute internal source from flux moments
                in_group_source[:,:,group] = in_group_scatter_expansion(prev_group_flux, group)
                source[:,:,group] = fission_source[:,:,group]+g2g_source[:,:,group]+in_group_source[:,:,group]
            # 	loop over spatial zones from left boundary to right boundary:
                for i in range(num_sections): 
                    section_index = int(np.floor(i*dx))
            # 		loop over rightward ordinates, these have mu > 0:
                    current_cell_avg_flux = np.zeros(num_ordinates//2)
                    right_edge_cell_flux = np.zeros(num_ordinates//2)
            # 		solve for the cell-average angular flux using inward flux
                    current_cell_avg_flux = (1+(total_xs[section_index,group]*dx)/(2*np.abs(quadrature_points[:num_ordinates//2])))**-1\
                        *(left_edge_cell_flux+dx*source[i,:num_ordinates//2,group]/(2*np.abs(quadrature_points[:num_ordinates//2])))
            # 		solve for the outgoing cell-edge angular flux
                    right_edge_cell_flux = 2*current_cell_avg_flux - left_edge_cell_flux
            # 		if outgoing cell-edge flux negative:
                    if any(right_edge_cell_flux < 0):
            # 			set outgoing cell-edge flux to zero
                        right_edge_cell_flux = np.where(right_edge_cell_flux<0, 0, right_edge_cell_flux)
            # 			recompute cell-average angular flux from particle balance
                        current_cell_avg_flux = source[i,:num_ordinates//2,group]/total_xs[section_index,group]\
                            +np.abs(quadrature_points[:num_ordinates//2])*left_edge_cell_flux/(total_xs[section_index,group]*dx)
                    left_edge_cell_flux = right_edge_cell_flux
                    group_flux_moments[i,:] += flux_to_moments(current_cell_avg_flux,1)
            # 	apply reflecting boundary condition to compute inward right angular fluxes
                if reflecting:
                    right_edge_cell_flux = np.flip(left_edge_cell_flux) 
                else:
                    right_edge_cell_flux = moments_to_flux(incoming_moments_r)[num_ordinates//2:]
            # 	loop over spatial zones from right boundary to left boundary:
                for i in range(num_sections-1,-1,-1): 
                    section_index = int(np.floor(i*dx))
            # 		loop over leftward ordinates, these have mu < 0:
                    current_cell_avg_flux = np.zeros(num_ordinates//2)
                # 	solve for the cell-average angular flux using inward flux
                    current_cell_avg_flux = (1+(total_xs[section_index,group]*dx)/(2*np.abs(quadrature_points[num_ordinates//2:])))**-1\
                        *(right_edge_cell_flux+dx*source[i,num_ordinates//2:,group]/(2*np.abs(quadrature_points[num_ordinates//2:])))                # 			solve for the outgoing cell-edge angular flux
                    left_edge_cell_flux = 2*current_cell_avg_flux - right_edge_cell_flux
        # 			if outgoing cell-edge flux negative:
                    if any(left_edge_cell_flux < 0):
            # 			set outgoing cell-edge flux to zero
                        left_edge_cell_flux = np.where(left_edge_cell_flux<0, 0, left_edge_cell_flux)
                # 		recompute cell-average angular flux from particle balance
                        current_cell_avg_flux = source[i,num_ordinates//2:, group]/total_xs[section_index,group]\
                            +np.abs(quadrature_points[num_ordinates//2:])*right_edge_cell_flux/(total_xs[section_index,group]*dx)
                    right_edge_cell_flux = left_edge_cell_flux
                    group_flux_moments[i,:] += flux_to_moments(current_cell_avg_flux,-1)
            # 	apply reflecting boundary condition to compute inward left angular fluxes
                if reflecting:
                    left_edge_cell_flux = np.flip(right_edge_cell_flux)
                else:
                    left_edge_cell_flux = moments_to_flux(incoming_moments_l)[:num_ordinates//2]
            flux_moments[:,:,group] = group_flux_moments

# report scalar flux in each cell
#TODO are other quantities of interest? If so, it shouldn't be an issue to plot them as well
scalar_flux = flux_moments[:,0,:]
ax = sns.heatmap(np.transpose(scalar_flux), cmap='viridis')
plt.show()