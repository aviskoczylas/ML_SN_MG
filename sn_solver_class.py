import time
start_time = time.time()
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import lpmv
from numba import njit, prange
print(time.time()-start_time)

#TODO implement vacuum bc
#TODO Write fixed source as Fourier Expansion
#random numbers for s2, s3.

class Sn():
    def __init__(self,num_ords, num_groups, num_nodes, NH, dx):
        self.num_ordinates = num_ords
        self.num_groups = num_groups
        self.num_nodes = num_nodes
        self.cell_index = np.linspace(1,self.num_nodes,self.num_nodes)
        self.data_dir = "./data/"
        self.NH = NH
        self.E0 = 1e6
        self.Emin = 1
        #self.E0 = 1e7
        #self.Emin = 1e-2
        self.nbins = num_groups + 1
        self.leg_order = 4
        self.AH = 1
        self.NH = NH
        self.AU = 238
        self.NU = 1
        self.dx = dx
        self.tol = 1e-8
        assert np.abs(1/self.dx-int(1/self.dx)) < self.tol #each cell must be evenly split

        chi35 = pd.read_csv(f'{self.data_dir}chi_u235.txt', sep = '\t',header = 0)
        H1 = pd.read_csv(f'{self.data_dir}xs_h1_T293k.txt', sep  = '\t', header = 0)
        U238 = pd.read_csv(f'{self.data_dir}xs_u238_T293k.txt',sep  = '\t', header = 0)
        sigma_f = pd.read_csv(f'{self.data_dir}xs_u238_fission.csv', sep = ',', dtype=float).to_numpy()

        chi = np.array([chi35['E'],chi35['chi']]).T
        H = np.array([H1['E'],H1['sigma_t'],H1['sigma_s']]).T
        XS38 = np.array([U238['E'],U238['sigma_t'],U238['sigma_s']]).T
        chi = self.get_data(chi,self.nbins)
        H = self.get_data(H,self.nbins)
        H *= self.NH
        XS38 = self.get_data(XS38,self.nbins)
        self.chi = chi[:,1]
        self.boundaries = XS38[:,0]
        self.Evec = self.E0 * np.exp(-self.boundaries)
        self.sig_t_U = XS38[:,1]
        self.sig_s0_U = XS38[:,2]
        self.sig_t_H = H[:,1]
        self.sig_s0_H = H[:,2]
        self.sigma_f = self.get_data(sigma_f,self.nbins)[:,1]

        self.xs_gtg_p0_U = None
        self.xs_gtg_p1_U = None
        self.xs_gtg_p2_U = None
        self.xs_gtg_p3_U = None

        self.xs_gtg_p0_H = None
        self.xs_gtg_p1_H = None
        self.xs_gtg_p2_H = None
        self.xs_gtg_p3_H = None

    def get_data(self,data, gridpoints):
        data = data[(data[:, 0] <= self.E0) & (data[:, 0] >= self.Emin)]
        data[:, 0] = np.log(self.E0 / data[:, 0])

        # Sort
        sort_idx = np.argsort(data[:, 0])
        data = data[sort_idx]
        new_grid = np.linspace(np.log(self.E0/self.E0), np.log(self.E0 / self.Emin), gridpoints)
        out = np.empty((new_grid.size, data.shape[1]), dtype=float)
        out[:, 0] = new_grid

        xp = data[:, 0]
        for i in range(1, data.shape[1]): out[:, i] = np.interp(new_grid, xp, data[:, i])
        return out

    def group_bound(self, A, g, lga): return (self.boundaries.size if A == 1
                else np.searchsorted(self.boundaries, self.boundaries[g] + lga))
                #else 1 + np.searchsorted(self.boundaries, self.boundaries[g] + lga))

    def gmax_vec_fn(self, A, lga):
        gmax_vec = np.zeros_like(self.boundaries, dtype = int)
        for g in range(self.boundaries.size): gmax_vec[g] = self.group_bound(A,g,lga)
        return gmax_vec

    def sigma_gtg_generator(self):
        sigma = self.gen_sig_sn_gtg(self.AU,self.sig_s0_U,self.leg_order, self.boundaries,
                                        self.gmax_vec_fn(self.AU,self.lga_fn(self.alpha_fn(self.AU))),
                                        self.alpha_fn(self.AU), self.tol)
        self.xs_gtg_p0_U = sigma[0,:,:]
        self.xs_gtg_p1_U = sigma[1,:,:]
        self.xs_gtg_p2_U = sigma[2,:,:]
        self.xs_gtg_p3_U = sigma[3,:,:]

        sigma = self.gen_sig_sn_gtg(self.AH,self.sig_s0_H,self.leg_order, self.boundaries,
                                        self.gmax_vec_fn(self.AH,self.lga_fn(self.alpha_fn(self.AH))),
                                        self.alpha_fn(self.AH), self.tol)
        self.xs_gtg_p0_H = sigma[0,:,:]
        self.xs_gtg_p1_H = sigma[1,:,:]
        self.xs_gtg_p2_H = sigma[2,:,:]
        self.xs_gtg_p3_H = sigma[3,:,:]

    @staticmethod
    def alpha_fn(A): return ((A - 1.0)/(A + 1.0)) ** 2

    @staticmethod
    def lga_fn(alpha): return -np.log(alpha) if alpha != 0 else np.inf

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def gen_sig_sn_gtg(A, sig_s0, order, boundaries, gmax_vec, alpha, tol, n_sub = 8):
        """Parallel midpoint integration (no SciPy quad).
        Integrates over x in [c, x2] using n_sub midpoints per base-bin."""
        G = boundaries.size
        du = boundaries[1] - boundaries[0]
        den = (1 - alpha) * du
        lga = -np.log(alpha) if A != 1 else np.inf
        sigma_gtg = np.zeros((order, G - 1, G - 1))
        Am1 = A-1
        Ap1 = A+1

        for l in range(order):
            for gp in prange(G - 1):
                x1 = boundaries[gp]
                x2 = boundaries[gp+1]

                for g in range(gp, min(gmax_vec[gp], G-1)):
                    y1 = boundaries[g]
                    y2 = boundaries[g + 1]
                    c = max(x1, y1 - lga)

                    if (y1 < x1) or (c >= x2) or den == 0.0: continue

                    # choose number of midpoint samples
                    length = x2 - c
                    n_steps_base = int(np.ceil(length / du))
                    if n_steps_base < 1: n_steps_base = 1
                    n_steps = n_steps_base * n_sub
                    dx = length / n_steps

                    # integrate
                    acc = 0
                    for s in range(n_steps):
                        xm = c + (s + .5) * dx
                        a = y1 if xm < y1 else xm
                        bx = xm + lga
                        b = y2 if bx > y2 else bx

                        if l == 0: val = np.exp(-(a - xm)) - np.exp(-(b - xm))

                        elif l == 1:
                            if A == 1:
                                val = ((Ap1) / 3) * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b)))
                            else:
                                val = (((Ap1) / 3) * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b)))
                                  - (Am1) * (np.exp(.5 * (xm - a)) - np.exp(.5 * (xm - b))))


                        elif l == 2:
                            if A == 1:
                                val = .0625 * ((np.exp(xm - 2*a - b) - np.exp(xm - a - 2*b)) *
                                (-4*(3*A*A - 1)*np.exp(a+b) + (3*Ap1*Ap1*(np.exp(a+xm)+np.exp(b+xm)))))
                            else:
                                val = .0625 * (6 * Am1 *  Am1 * (b-a)
                                + (np.exp(xm - 2*a - b) - np.exp(xm - a - 2*b)) *
                                (-4*(3*A*A - 1)*np.exp(a+b) + (3*Ap1*Ap1 * (np.exp(a+xm)+np.exp(b+xm)))))

                        else:
                            if A == 1:
                                val = 0.0625 * (
                                  + 2*(Ap1*Ap1*Ap1) * (np.exp(2.5*(xm - a)) - np.exp(2.5*(xm - b)))
                                  - 2*(Ap1)*(5*A*A - 1) * (np.exp(1.5*(xm - a)) - np.exp(1.5*(xm - b))))
                            else:
                                val = 0.0625 * (
                                    10*(Am1*Am1*Am1) * (np.exp(.5*(a - xm)) - np.exp(.5*(b - xm)))
                                  + 2*(Ap1*Ap1*Ap1) * (np.exp(2.5*(xm - a)) - np.exp(2.5*(xm - b)))
                                  + 6*(Am1)*(5*A*A - 1) * (np.exp(.5*(xm - a)) - np.exp(.5*(xm - b)))
                                  - 2*(Ap1)*(5*A*A - 1) * (np.exp(1.5*(xm - a)) - np.exp(1.5*(xm - b))))

                        acc += val

                    sigma_gtg[l, gp, g] = (sig_s0[gp] * acc * dx ) / den

                    if np.abs(sigma_gtg[l,gp,g]) < tol: break

        return sigma_gtg



sn = Sn(16,8,32,5,.5)

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

scatter_matrices = np.stack((g2g_p0_scatter_xs, g2g_p1_scatter_xs), axis = 0)

neutrons_per_fission = 2.5
num_sections = round(num_cells/dx)
cell_indices = np.floor(np.arange(num_sections)*dx).astype(int)
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

@njit
def in_group_scatter_expansion(group_flux_moments,g):
    source = np.zeros((num_sections, num_ordinates))
    for l in range(order):
        source+=((2*l+1)/2)*np.outer(scatter_matrices[l,cell_indices,g,g]*group_flux_moments[:, l],legendre[l,:])
    return source

@njit
def g2g_scatter_expansion(flux_moments, group_scattered_into):
    source = np.zeros((num_sections, num_ordinates))
    exclude_current_group = np.ones(num_groups)
    exclude_current_group[group_scattered_into] = 0
    for l in range(order):
        outscatter_rate = np.zeros(num_sections)
        for i in range(num_sections):
            outscatter_rate[i] = np.sum(scatter_matrices[l,cell_indices[i],:,group_scattered_into]*flux_moments[i, l, :]*exclude_current_group)
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
@njit
def moments_to_flux(flux_moments):
    weighted_moments = (2*np.arange(order)+1)/2*flux_moments
    angular_flux = weighted_moments @ legendre
    return angular_flux

#decontruct the flux into its legendre moments - should only be used on a partial flux, hence only direction 1 or -1
@njit
def flux_to_moments(angular_flux, direction):
    weighted_flux = weight*angular_flux
    if direction == 1: #mu > 0
        flux_moments = legendre[:,:num_ordinates//2].copy() @ weighted_flux
    elif direction == -1: #mu < 0
        flux_moments = legendre[:,num_ordinates//2:].copy() @ weighted_flux
    return flux_moments 

prev_fission_source, prev_g2g_source = None, None

bc_type = "reflecting" #options are fixed, reflecting, and vacuum
reflecting = True
incoming_moments_l = np.array([0.1,0]) #for constant bc, this is a given. For reflecting BC, this is a guess
incoming_moments_r = np.array([0.1,0]) #only used for the case of constant bc
in_group_source = np.ones((num_sections, num_ordinates, num_groups))*0.01 #initial guess
g2g_source = np.ones((num_sections, num_ordinates, num_groups))*0.01#initial guess, source is summed over all incoming groups
fission_source = np.ones((num_sections, num_ordinates, num_groups))*0.01 #initial guess
flux_moments = np.zeros((num_sections, order, num_groups))
flux_moments[:, 0, :] = 0.01 #initial guess

left_edge_cell_flux = moments_to_flux(incoming_moments_l)[:num_ordinates//2] #initial guess if reflecting
k=1 #initial guess

@njit
def transport_sweep(group, group_source, left_edge_cell_flux):
    group_flux_moments = np.zeros((num_sections, order))
# 	loop over spatial zones from left boundary to right boundary:
    for i in range(num_sections): 
        section_index = int(np.floor(i*dx))
    # 	loop over rightward ordinates, these have mu > 0:
        current_cell_avg_flux = np.zeros(num_ordinates//2)
        right_edge_cell_flux = np.zeros(num_ordinates//2)
# 		solve for the cell-average angular flux using inward flux
        current_cell_avg_flux = (1+(total_xs[section_index,group]*dx)/(2*np.abs(quadrature_points[:num_ordinates//2])))**-1\
            *(left_edge_cell_flux+dx*group_source[i,:num_ordinates//2]/(2*np.abs(quadrature_points[:num_ordinates//2])))
    # 	solve for the outgoing cell-edge angular flux
        right_edge_cell_flux = 2*current_cell_avg_flux - left_edge_cell_flux
# 		if outgoing cell-edge flux negative:
        if np.any(right_edge_cell_flux < 0):
# 			set outgoing cell-edge flux to zero
            right_edge_cell_flux = np.where(right_edge_cell_flux<0, 0, right_edge_cell_flux)
    # 		recompute cell-average angular flux from particle balance
            current_cell_avg_flux = group_source[i,:num_ordinates//2]/total_xs[section_index,group]\
                +np.abs(quadrature_points[:num_ordinates//2])*left_edge_cell_flux/(total_xs[section_index,group]*dx)
        left_edge_cell_flux = right_edge_cell_flux
        group_flux_moments[i,:] += flux_to_moments(current_cell_avg_flux,1)
    # apply reflecting boundary condition to compute inward right angular fluxes
    if bc_type == "reflecting":
        right_edge_cell_flux = np.flip(left_edge_cell_flux) 
    elif bc_type == "fixed":
        right_edge_cell_flux = moments_to_flux(incoming_moments_r)[num_ordinates//2:]
    elif bc_type == "vacuum": 
        right_edge_cell_flux = np.zeros(num_ordinates//2)
# 	loop over spatial zones from right boundary to left boundary:
    for i in range(num_sections-1,-1,-1): 
        section_index = int(np.floor(i*dx))
    # 	loop over leftward ordinates, these have mu < 0:
        current_cell_avg_flux = np.zeros(num_ordinates//2)
    # 	solve for the cell-average angular flux using inward flux
        current_cell_avg_flux = (1+(total_xs[section_index,group]*dx)/(2*np.abs(quadrature_points[num_ordinates//2:])))**-1\
            *(right_edge_cell_flux+dx*group_source[i,num_ordinates//2:]/(2*np.abs(quadrature_points[num_ordinates//2:])))                # 			solve for the outgoing cell-edge angular flux
        left_edge_cell_flux = 2*current_cell_avg_flux - right_edge_cell_flux
# 		if outgoing cell-edge flux negative:
        if np.any(left_edge_cell_flux < 0):
# 			set outgoing cell-edge flux to zero
            left_edge_cell_flux = np.where(left_edge_cell_flux<0, 0, left_edge_cell_flux)
        # 	recompute cell-average angular flux from particle balance
            current_cell_avg_flux = group_source[i,num_ordinates//2:]/total_xs[section_index,group]\
                +np.abs(quadrature_points[num_ordinates//2:])*right_edge_cell_flux/(total_xs[section_index,group]*dx)
        right_edge_cell_flux = left_edge_cell_flux
        group_flux_moments[i,:] += flux_to_moments(current_cell_avg_flux,-1)
# 	apply reflecting boundary condition to compute inward left angular fluxes
    if bc_type == "reflecting":
        left_edge_cell_flux = np.flip(right_edge_cell_flux)
    elif bc_type == "fixed":
        left_edge_cell_flux = moments_to_flux(incoming_moments_l)[:num_ordinates//2]
    elif bc_type == "vacuum":
        left_edge_cell_flux = np.zeros(num_ordinates//2)
    return (group_flux_moments, left_edge_cell_flux)
                


#loop until everything converges
while not converged(prev_fission_source, fission_source, "outer loop: "): #outer iteration - fission source
    source = np.zeros((num_sections, num_ordinates, num_groups))
    if prev_fission_source is not None:
        k_new = k * np.sum(fission_source)/np.sum(prev_fission_source) #power iteration
        alpha = 0.7  #arbitrary underrelaxation coefficient to help converge
        k = alpha * k_new + (1 - alpha) * k
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
                #compute internal source from flux moments
                in_group_source[:,:,group] = in_group_scatter_expansion(prev_group_flux, group)
                source[:,:,group] = fission_source[:,:,group]+g2g_source[:,:,group]+in_group_source[:,:,group]
                group_flux_moments, left_edge_cell_flux = transport_sweep(group,source[:,:,group], left_edge_cell_flux)
            flux_moments[:,:,group] = group_flux_moments

# report scalar flux in each cell
#TODO are other quantities of interest? If so, it shouldn't be an issue to plot them as well
scalar_flux = flux_moments[:,0,:]
ax = sns.heatmap(np.transpose(scalar_flux), cmap='viridis')
plt.show()
print(time.time()-start_time)
