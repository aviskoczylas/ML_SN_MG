import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from numpy.polynomial.legendre import leggauss
from scipy.special import lpmv
from numba import njit, prange
np.random.seed(10)

def np_fourier_expansion(coeffs,order,num_nodes, num_sections):
    x = np.linspace(0, num_nodes, num_sections)
    # Compute source by Fourier expansion
    return coeffs[0] + sum(
        coeffs[2 * k + 1] * np.cos(np.pi * (k + 1) * x)
        + coeffs[2 * k + 2] * np.sin(np.pi * (k + 1) * x)
        for k in range(order))

@njit(fastmath=True)
def _half_moment(leg, w, psi):
    L = leg.shape[0]
    Nh = w.size
    out = np.zeros(L)
    for l in range(L):
        acc = 0.0
        for n in range(Nh):
            acc += leg[l, n] * w[n] * psi[n]
        out[l] = acc
    return out

@njit(cache=True, fastmath=True)
def _project_moments(psi_sn: np.ndarray, leg_w: np.ndarray) -> np.ndarray:
    if psi_sn.ndim == 1:
        out = np.empty(leg_w.shape[0], dtype=psi_sn.dtype)
        for l in range(leg_w.shape[0]):
            out[l] = (psi_sn * leg_w[l]).sum()
        return out
    else:
        S, N = psi_sn.shape
        L = leg_w.shape[0]
        out = np.empty((S, L), dtype=psi_sn.dtype)
        for s in range(S):
            row = psi_sn[s]
            for l in range(L):
                out[s, l] = (row * leg_w[l]).sum()
        return out

@njit(fastmath=True)
def _sweep_sc_one_group(sigt_g, dx, mu_pos, mu_neg, src, leg_pos, leg_neg, w_pos, w_neg, bc_code):
    def _sweep_response(mu_vec, sigt, dx):
        tau = sigt * dx / mu_vec
        e = np.exp(-tau)
        t1 = np.where(tau > 1e-12, (1.0 - e) / tau, 1.0 - 0.5 * tau)
        return e, t1

    S = sigt_g.size
    L = leg_pos.shape[0]
    Nh = mu_pos.size
    mom = np.zeros((S, L))
    psi_in = np.zeros(Nh)

    # Forward sweep 
    for i in range(S):
        sigt = sigt_g[i]
        q    = src[i, :Nh]
        e, t1 = _sweep_response(mu_pos, sigt, dx)
        psi_out = psi_in * e + (q / sigt) * (1.0 - e)
        psi_bar = psi_in * t1 + (q / sigt) * (1.0 - t1)
        mom[i, :] += _half_moment(leg_pos, w_pos, psi_bar)
        psi_in = psi_out

    # Set inflow for negative mu at right boundary
    if bc_code == 0: psi_in = psi_out[::-1]
    elif bc_code == 1: psi_in = np.zeros(Nh)

    # Backward sweep 
    for i in range(S - 1, -1, -1):
        sigt = sigt_g[i]
        q    = src[i, Nh:]
        e, t1 = _sweep_response(mu_neg, sigt, dx)
        psi_out = psi_in * e + (q / sigt) * (1.0 - e)
        psi_bar = psi_in * t1 + (q / sigt) * (1.0 - t1)
        mom[i, :] += _half_moment(leg_neg, w_neg, psi_bar)
        psi_in = psi_out

    return mom

@njit(parallel=True, fastmath=True, cache=True)
def _outer_iteration_parallel_fixed(flux_prev, sigt_sec, scatter_mats_sec,
                                    Qmom, dx, mu, w, leg_full, bc_type, 
                                    theta_relax, max_inner, eps_inner):
    S, L, G = flux_prev.shape
    N = mu.size
    Nh = N // 2

    mu_pos = np.abs(mu[:Nh]); mu_neg = np.abs(mu[Nh:])
    w_pos  = w[:Nh];           w_neg  = w[Nh:]
    leg_pos = leg_full[:, :Nh]
    leg_neg = leg_full[:, Nh:]

    # Precompute off-diagonal scatter contribution per (s,g,l)
    g2g_amp = np.zeros((S, G, L))
    for s in prange(S):
        for g in range(G):
            for l in range(L):
                acc = 0.0
                for gp in range(G):
                    if gp != g:
                        acc += scatter_mats_sec[l, s, gp, g] * flux_prev[s, l, gp]
                g2g_amp[s, g, l] = acc

    # In-group diagonal σ_l(s,g,g)
    sig_diag = np.zeros((L, S, G))
    for l in prange(L):
        for s in range(S):
            for g in range(G):
                sig_diag[l, s, g] = scatter_mats_sec[l, s, g, g]

    flux_next = np.zeros_like(flux_prev)

    # Group-by-group transport sweep with inner iteration on moments
    for g in prange(G):
        group_mom = flux_prev[:, :, g].copy()
        prev_mom  = group_mom.copy()

        for _ in range(max_inner):
            total_src = np.zeros((S, N))

            for l in range(L):
                coeff = 0.5 * (2 * l + 1)
                for s in range(S):
                    amp = coeff * Qmom[s, l, g]
                    for n in range(N):
                        total_src[s, n] += amp * leg_full[l, n]

            # Scattering source (in-group + g->g’)
            for l in range(L):
                coeff = 0.5 * (2 * l + 1)
                for s in range(S):
                    amp = coeff * (sig_diag[l, s, g] * group_mom[s, l] + g2g_amp[s, g, l])
                    for n in range(N):
                        total_src[s, n] += amp * leg_full[l, n]

            # Sweep this energy group
            sigt_g = sigt_sec[:, g]
            new_mom = _sweep_sc_one_group(sigt_g, dx, mu_pos, mu_neg, total_src,
                              leg_pos, leg_neg, w_pos, w_neg, bc_type)

            # Inner convergence check on group moments
            num = 0.0; den = 0.0
            for s in range(S):
                for l in range(L):
                    d = new_mom[s, l] - prev_mom[s, l]
                    num += d * d
                    den += prev_mom[s, l] * prev_mom[s, l]
            l2 = np.sqrt(num / (den + 1e-30))

            # Under-relaxation
            for s in range(S):
                for l in range(L):
                    group_mom[s, l] = theta_relax * new_mom[s, l] + (1.0 - theta_relax) * group_mom[s, l]

            prev_mom[:, :] = new_mom
            if l2 < eps_inner:
                break

        flux_next[:, :, g] = group_mom

    return flux_next

def choose_bc_type():
    rn = np.random.rand()
    if rn < .5: bc_type = "reflecting"
    else: bc_type = "vacuum"
    return bc_type

def choose_NH(): return np.round(.1 * np.random.rand(),5)

class Sn:
    def __init__(self, leg_order, num_ords, num_groups, num_nodes, NH, dx, bc_type, source_order, run):
        self.num_ordinates = int(num_ords)
        assert self.num_ordinates % 2 == 0
        self.num_groups = int(num_groups)
        self.num_nodes = int(num_nodes)
        self.dx = float(dx)
        self.tol = 1e-6
        self.run = run

        assert abs(1.0/self.dx - int(1.0/self.dx)) < self.tol
        self.sections_per_cell = int(round(1.0/self.dx))
        self.num_sections = self.num_nodes * self.sections_per_cell
        self.section_index_to_cell = (np.arange(self.num_sections) // self.sections_per_cell).astype(int)

        self.AH, self.AU = 1., 238.
        self.NH = NH
        self.NU35 = .0125
        self.NU38 = .0875

        self.E0, self.Emin = 1e7, 1
        self.nbins = self.num_groups + 1
        self.boundaries = None
        self.Evec = None

        self.nu = 2.43
        self.bc_type = bc_type.lower()
        self.data_dir = "./data/"

        self.leg_order = leg_order
        self.legendre = np.zeros((self.leg_order, self.num_ordinates))
        x, w = leggauss(self.num_ordinates)
        self.weight = np.flip(w)
        self.mu = np.flip(x)
        self._build_legendre_matrix()
        self.source_order = source_order

        self.sig_t_H = None; self.sig_s0_H = None
        self.sig_t_U = None; self.sig_s0_U = None
        self.sigma_f = None
        self.chi = None

        self.xsH_gtg = None
        self.xsU_gtg = None

        self.sigt_sec = None
        self.scatter_mats_sec = None

        self._load_or_build_data()
        self.build_geometry(seed=(self.run), p_U=0.6, verbose=True)

    @staticmethod
    def _safe_load_csv(path, sep, cols=None):
        df = pd.read_csv(path, sep=sep, header=0)
        if cols is not None:
            try:
                df = df.loc[:, cols]
            except Exception:
                pass
        return df.to_numpy(dtype=float)

    @staticmethod
    def get_data(data, gridpoints, E0, Emin):
        data = data[(data[:, 0] <= E0) & (data[:, 0] >= Emin)].copy()
        data[:, 0] = np.log(E0 / data[:, 0])
        data = data[np.argsort(data[:, 0])]
        new_grid = np.linspace(np.log(E0/E0), np.log(E0/Emin), gridpoints)
        out = np.empty((new_grid.size, data.shape[1]), dtype=float)
        out[:, 0] = new_grid
        xp = data[:, 0]
        for i in range(1, data.shape[1]):
            out[:, i] = np.interp(new_grid, xp, data[:, i])
        return out

    def _load_or_build_data(self):
        chi35 = self._safe_load_csv(f'{self.data_dir}chi_u235.txt', '\t', cols=['E','chi'])
        H1   = self._safe_load_csv(f'{self.data_dir}xs_h1_T293k.txt', '\t', cols=['E','sigma_t','sigma_s'])
        U238 = self._safe_load_csv(f'{self.data_dir}xs_u238_T293k.txt', '\t', cols=['E','sigma_t','sigma_s'])
        sigma_f_raw = self._safe_load_csv(f'{self.data_dir}xs_u238_fission.csv', ',', cols=None)
        U235 = self._safe_load_csv(f'{self.data_dir}xs_u235.csv', sep=';', cols=None)

        chi = self.get_data(chi35, self.nbins, self.E0, self.Emin)
        H   = self.get_data(H1,   self.nbins, self.E0, self.Emin)
        U   = self.get_data(U238, self.nbins, self.E0, self.Emin)
        U35 = self.get_data(U235, self.nbins, self.E0, self.Emin)

        self.boundaries = U[:, 0]
        self.Evec = self.E0 * np.exp(-self.boundaries)

        self.sig_t_H  = H[:-1, 1] * self.NH
        self.sig_s0_H = H[:-1, 2] * self.NH
        self.sig_t_U  = U[:-1, 1] * self.NU38 + U35[:-1,1] * self.NU35
        self.sig_s0_U = U[:-1, 2] * self.NU38 + U35[:-1,2] * self.NU35
        self.sigma_f  = self.get_data(sigma_f_raw,self.nbins,self.E0,self.Emin)[:-1,1] * self.NU38
        self.sigma_f += U35[:-1,3] * self.NU35
        self.sigma_f *= self.nu

        self.chi = chi[:-1, 1]
        s = self.chi.sum()
        if s > 0: self.chi /= s

        self.xsH_gtg = self._build_sigma_gtg(self.AH, self.sig_s0_H)
        self.xsU_gtg = self._build_sigma_gtg(self.AU, self.sig_s0_U)

    def _build_sigma_gtg(self, A, sig_s0):
        gmax_vec = self._gmax_vec_fn(A, self._lga_fn(self._alpha_fn(A)))
        return self._gen_sig_sn_gtg(A, sig_s0, self.leg_order, self.boundaries, gmax_vec,
                                    self._alpha_fn(A), self.tol)

    def build_geometry(self, layout=None, p_U=0.5, seed=10, verbose=True):
        if layout is not None:
            arr = np.asarray(layout, dtype=int)
            assert arr.shape == (self.num_nodes,)
            assert np.all((arr == 0) | (arr == 1))
            self.cell_layout = arr.copy()
        else:
            # choose geometry from rng
            rng = np.random.default_rng(seed=seed)
            if self.num_nodes == 1: self.cell_layout = [1]
            else: self.cell_layout = (rng.random(self.num_nodes) < float(p_U)).astype(int)

        # apply discritization
        self.mat_per_section = np.repeat(self.cell_layout, self.sections_per_cell)
        if verbose:
            print(f"Geometry H=0,U=1: {self.cell_layout}")
            print(f"Number of Uranium Cells: {sum(self.cell_layout)}")

        # stitch everything together
        self._build_per_section_xs_from_layout()

    def _build_per_section_xs_from_layout(self):
        S, G = self.num_sections, self.num_groups
        self.sigt_sec = np.zeros((S, G))
        self.scatter_mats_sec = np.zeros((self.leg_order, S, G, G))
        for s in range(S):
            if self.mat_per_section[s] == 0:
                self.sigt_sec[s, :] = self.sig_t_H
                self.scatter_mats_sec[:, s, :, :] = self.xsH_gtg
            else:
                self.sigt_sec[s, :] = self.sig_t_U
                self.scatter_mats_sec[:, s, :, :] = self.xsU_gtg

    def _build_legendre_matrix(self):
        for l in range(self.leg_order):
            self.legendre[l, :] = lpmv(0, l, self.mu)

    @staticmethod
    def _alpha_fn(A): return ((A - 1.0) / (A + 1.0)) ** 2

    @staticmethod
    def _lga_fn(alpha): return -np.log(alpha) if alpha != 0 else np.inf

    def _gmax_vec_fn(self, A, lga):
        def _group_bound(boundaries, A, g, lga):
            if A == 1.0: return boundaries.size
            return int(np.searchsorted(boundaries, boundaries[g] + lga)) + 1
    
        G = self.boundaries.size
        gmax_vec = np.zeros(G, dtype=np.int64)
        for g in range(G): gmax_vec[g] = _group_bound(self.boundaries, A, g, lga)
        return gmax_vec

    @staticmethod
    @njit(parallel=True, fastmath=True,cache=True)
    def _gen_sig_sn_gtg(A, sig_s0, order, boundaries, gmax_vec, alpha, tol, n_sub=32):
        if order > 4: raise ValueError("Legendre Expansion Order Not Supported")
        G = boundaries.size
        du = boundaries[1] - boundaries[0]
        den = (1 - alpha) * du
        lga = -np.log(alpha) if A != 1 else np.inf
        sigma_gtg = np.zeros((order, G - 1, G - 1))
        Am1 = A - 1
        Ap1 = A + 1

        for l in range(order):
            for gp in prange(G - 1):
                x1 = boundaries[gp]
                x2 = boundaries[gp + 1]
                gmax = gmax_vec[gp] if gmax_vec[gp] < G else (G - 1)
                for g in range(gp, gmax):
                    y1 = boundaries[g]
                    y2 = boundaries[g + 1]
                    c = x1 if x1 > (y1 - lga) else (y1 - lga)
                    if (y1 < x1) or (c >= x2) or den == 0.0: continue

                    length = x2 - c
                    n_steps_base = int(np.ceil(length / du))
                    if n_steps_base < 1:
                        n_steps_base = 1
                    n_steps = n_steps_base * n_sub
                    dx = length / n_steps

                    acc = 0.0
                    for s in range(n_steps):
                        xm = c + (s + 0.5) * dx
                        a = y1 if xm < y1 else xm
                        bx = xm + lga
                        b = y2 if bx > y2 else bx

                        if l == 0: val = np.exp(-(a - xm)) - np.exp(-(b - xm))
                        elif l == 1:
                            if A == 1.0: val = (Ap1 / 3.0) * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b)))
                            else: val = ((Ap1 / 3.0) * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b)))
                                       - (Am1) * (np.exp(0.5 * (xm - a)) - np.exp(0.5 * (xm - b))))
                        elif l == 2:
                            if A == 1.0: val = 0.0625 * ((np.exp(xm - 2 * a - b) - np.exp(xm - a - 2 * b)) *
                                                (-4.0 * (3.0 * A * A - 1.0) * np.exp(a + b)
                                                 + (3.0 * Ap1 * Ap1 * (np.exp(a + xm) + np.exp(b + xm)))))
                            else: val = 0.0625 * (6.0 * Am1 * Am1 * (b - a)
                                                + (np.exp(xm - 2 * a - b) - np.exp(xm - a - 2 * b)) *
                                                (-4.0 * (3.0 * A * A - 1.0) * np.exp(a + b)
                                                 + (3.0 * Ap1 * Ap1 * (np.exp(a + xm) + np.exp(b + xm)))))
                        elif l == 3:
                            if A == 1.0: val = 0.0625 * (
                                    2.0 * (Ap1 * Ap1 * Ap1) * (np.exp(2.5 * (xm - a)) - np.exp(2.5 * (xm - b)))
                                    - 2.0 * (Ap1) * (5 * A * A - 1) * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b))))

                            else: val = 0.0625 * (
                                    10*(Am1*Am1*Am1) * (np.exp(.5*(a - xm)) - np.exp(.5*(b - xm)))
                                  + 2*(Ap1*Ap1*Ap1) * (np.exp(2.5*(xm - a)) - np.exp(2.5*(xm - b)))
                                  + 6*(Am1)*(5*A*A - 1) * (np.exp(.5*(xm - a)) - np.exp(.5*(xm - b)))
                                  - 2*(Ap1)*(5*A*A - 1) * (np.exp(1.5*(xm - a)) - np.exp(1.5*(xm - b))))
                        acc += val
                    sigma_gtg[l, gp, g] = (sig_s0[gp] * acc * dx) / den
        return sigma_gtg

    def sample_source_from_geometry(self, geom_layout, source_norm=1.0, max_resamples=20, damp=0.7):
        """
        Build per-group Fourier coefficients gated by geometry mask (0=no source, 1=source).
        Cases (inferred from mask):
          2: all zeros (no source) → coeffs=0
          1: mask touches either boundary → allow DC + all modes
          0: interior slab (ones exist; both ends zero) → no DC, oscillatory modes only
        """
        geom_layout = np.asarray(geom_layout, dtype=float)
        if geom_layout.ndim == 1:
            geom_layout = geom_layout[None, :]
        G, num_nodes = geom_layout.shape
        num_sections = self.num_sections
        x = np.linspace(0.0, 1.0, num_sections)  # integral in section space
        coeffs = np.zeros((1 + 2 * self.source_order, G), dtype=float)
        statuses = np.zeros(G, dtype=int)
    
        # Inflate node mask to section mask
        M_sections = np.repeat((geom_layout > 0.5).astype(float), self.sections_per_cell, axis=1)
    
        def infer_status(mask_nodes):
            if np.all(mask_nodes == 0.0):
                return 2
            left, right = mask_nodes[0] == 1.0, mask_nodes[-1] == 1.0
            return 1 if (left or right) else 0
    
        total_integral = 0.0
    
        for g in range(G):
            M = M_sections[g]
            status = infer_status(geom_layout[g])
            statuses[g] = status
            if status == 2:
                continue
    
            c = np.zeros(1 + 2 * self.source_order, dtype=float)
            if status == 0:
                if self.source_order > 0:
                    c[2::2] = np.random.uniform(-1.0, 1.0, size=self.source_order)
            else:  # status == 1
                c[0] = np.random.uniform(0.0, 1.0)
                if self.source_order > 0:
                    c[1:] = np.random.uniform(-1.0, 1.0, size=2 * self.source_order)
    
            for _ in range(max_resamples + 1):
                Q_unmasked = np_fourier_expansion(c, self.source_order, self.num_nodes, self.num_sections)
                Q_masked = M * Q_unmasked
                if np.all(Q_masked >= 0.0): break
                if self.source_order > 0:
                    c[1:] *= damp
                    c[0] = max(c[0], 0.0)
            else:
                Q_unmasked = np_fourier_expansion(c, self.source_order, self.num_nodes, self.num_sections)
                Q_masked = np.maximum(0.0, M * Q_unmasked)
    
            total_integral += np.trapz(Q_masked, x)
            coeffs[:, g] = c
    
        if total_integral <= 0.0:
            return coeffs, statuses, np.zeros((self.num_sections, self.leg_order, self.num_groups), dtype=float)
    
        scale = source_norm / total_integral
        coeffs *= scale
    
        # Build Qmom (l=0 only)
        Qmom = np.zeros((self.num_sections, self.leg_order, self.num_groups), dtype=float)
        for g in range(G):
            if not np.any(coeffs[:, g]): continue
            Qg = np_fourier_expansion(coeffs[:, g], self.source_order, self.num_nodes, self.num_sections)
            Qg *= M_sections[g]
            Qmom[:, 0, g] = Qg
    
        self.source_coeffs = coeffs
        return coeffs, statuses, Qmom

    def moments_to_flux(self, flux_moments):
        L = self.leg_order
        coeff = (2*np.arange(L)+1)/2.0
        return (flux_moments * coeff[None, :]) @ self.legendre

    def flux_to_moments(self, phi, direction):
        if direction == 1:
            leg = self.legendre[:, :self.num_ordinates//2]
            w = self.weight[:self.num_ordinates//2]
        else:
            leg = self.legendre[:, self.num_ordinates//2:]
            w = self.weight[self.num_ordinates//2:]
        return leg @ (w * phi)

    def _build_chi_per_section(self):
        if not hasattr(self, "mat_per_section"):
            raise RuntimeError("Call build_geometry() first.")
    
        S, G = self.num_sections, self.num_groups
        chi_sec = np.zeros((S, G), dtype=np.float64)
    
        chi_vec = np.array(self.chi, dtype=np.float64).ravel()
        if chi_vec.size != G or chi_vec.sum() <= 0:
            chi_vec = np.zeros(G, dtype=np.float64); chi_vec[0] = 1.0
        else: chi_vec /= chi_vec.sum()
    
        chi_sec[self.mat_per_section == 1, :] = chi_vec[None, :]
        return chi_sec

    def in_group_scatter_expansion(self, group_flux_moments, g):
        S, N = self.num_sections, self.num_ordinates
        src = np.zeros((S, N))
        for l in range(self.leg_order):
            sig_ll = self.scatter_mats_sec[l, :, g, g]
            term = ((2*l+1)/2.0) * (sig_ll * group_flux_moments[:, l])
            src += term[:, None] * self.legendre[l, :][None, :]
        return src

    def g2g_scatter_expansion(self, flux_moments, g):
        S, N = self.num_sections, self.num_ordinates
        src = np.zeros((S, N))
        for l in range(self.leg_order):
            sig_mat = self.scatter_mats_sec[l, :, :, :]
            col_g = sig_mat[:, :, g]
            contrib = np.sum(col_g * flux_moments[:, l, :], axis=1) - col_g[:, g] * flux_moments[:, l, g]
            src += ((2*l+1)/2.0) * contrib[:, None] * self.legendre[l, :][None, :]
        return src

    @staticmethod
    def converged(prev, curr, label=None, eps=1e-5):
    #def converged(prev, curr, label=None, eps=1e-6):
        if prev is None: return False
        diff = np.linalg.norm(curr - prev)
        norm = np.linalg.norm(prev)
        l2 = diff / (norm + 1e-12)
        if label: print(f"{label} L2: {l2:.3e}")
        return l2 < eps

    def run_fixed_source(self, A_coeffs=None, B_coeffs=None,
                     max_outer=20000, alpha_underrelax=0.7, eps_outer=1e-5,
                     theta_relax=0.6, max_inner=20000, eps_inner=1e-4):
        """Solve S_N transport for a prescribed spatial Fourier external source of order N."""
        S, G, L = self.num_sections, self.num_groups, self.leg_order
        Nmu = self.num_ordinates
    
        bc_map = {"reflecting": 0, "vacuum": 1}
        bc_code = bc_map[self.bc_type]
    
        # Cast data
        sigt_sec = self.sigt_sec
        scat     = self.scatter_mats_sec
        leg_full = self.legendre
        mu       = self.mu
        w        = self.weight
        dx       = self.dx
    
        # Build source from geometry layout (node mask = cell_layout)
        geom_mask_nodes = np.tile(self.cell_layout, (G, 1))  
        _, _, Qmom = self.sample_source_from_geometry(geom_mask_nodes)

        # Initial guess
        flux = np.ones((S, L, G), dtype=np.float64)
        
        for it in range(max_outer):
            flux_new = _outer_iteration_parallel_fixed(
                flux, sigt_sec, scat, Qmom, dx, mu, w, leg_full,
                bc_code, theta_relax, max_inner, eps_inner)
            num = np.linalg.norm(flux_new - flux)
            den = np.linalg.norm(flux) + 1e-30
            l2 = num / den
            if it % 10 == 0: print(f"[outer {it}] L2={l2:.5e}")
            flux = alpha_underrelax * flux_new + (1.0 - alpha_underrelax) * flux
            if l2 < eps_outer: break

        return flux
    
# run
leg_order = 4
source_order = 4
num_ordinates = 64
num_groups = 8
num_nodes = 1
dx = .001

runs = 10
xtrain = []
ytrain = []

print(f"Order: {leg_order}")
print(f"Ordinates: {num_ordinates}")
print(f"Groups: {num_groups}")
print(f"Nodes: {num_nodes}")
print(f"dx: {dx}")

start = time.time()
for run in range(runs):    
    stt = time.time()
    # choose variable parameters
    NH = choose_NH()
    bc_type = choose_bc_type()
    print(f"Run Number {run+1}, {bc_type} BC, NH = {NH}")
    sn = Sn(leg_order,num_ordinates, num_groups, num_nodes, NH, dx, bc_type, source_order, run)
    flux_moments = sn.run_fixed_source()
    phi = np.sum(flux_moments,axis=1)
    print("Elapsed:", np.round(time.time() - stt,6), "s")

    # save to df for each run
    # xtrain
#    row_num = np.concatenate([
#        sn.sig_t_H.ravel(),
#        sn.xsH_gtg.ravel(),
#        sn.source_coeffs.ravel(),
#        np.array([NH], dtype=np.float64),
#    ])
#    
#    row = np.concatenate([row_num.astype(object), np.array([bc_type], dtype=object)])
#    xtrain.append(row)

    # ytrain
#    phi_moments = np.zeros_like(sn.source_coeffs)
#    print(phi_moments.shape)
#    phi_foo = np_fourier_expansion(sn.source_coeffs[:,0], source_order, num_groups, sn.num_sections)
#    print(phi_foo.shape)
#
#    for i in range(num_groups): phi_moments[:,i] = (np_fourier_expansion(sn.source_coeffs[:,i], 
#                                                    source_order, num_groups, sn.num_sections))
#    ytrain.append(phi_moments.ravel())

#df = pd.DataFrame(xtrain)
#df.to_csv(f"data/xtrain_{runs}.csv",index=False)

#df = pd.DataFrame(ytrain)
#df.to_csv(f"data/ytrain_{runs}.csv",index=False)

print(f"Total Time = {np.round(time.time() - start,6)} s")

plotting = True
if plotting:
    S, N, G = sn.num_sections, sn.num_ordinates, sn.num_groups
    save_dir = "charts"
    def shade_material(ax):
        color_map = {0: ("blue", 0.12), 1: ("red",  0.12)}
        A = sn.cell_layout      
        n = sn.num_nodes
        i = 0
        while i < n:
            a_i = A[i]
            j = i + 1
            while j < n and A[j] == a_i: j += 1
            color, alpha = color_map.get(a_i, ("0.9", 0.10))
            ax.axvspan(i, j, facecolor=color, alpha=alpha, edgecolor="none", zorder=0)
            i = j
        ax.set_xlim(0, n)

    for g in range(G):
        # Angular flux colorplot
        ang = sn.moments_to_flux(flux_moments[:, :, g])  
        fig, ax = plt.subplots(figsize=(6, 3.8))
        im = ax.imshow(ang.T, aspect='auto', origin='lower',
                       extent=[0, sn.num_nodes, -1, 1], interpolation='nearest')
        ax.set_title(fr"Group {g} — $\psi (x)$")
        ax.set_xlabel("Section")
        ax.set_ylabel(r"$\mu$")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(r"$\psi$")
        fig.savefig(f"{save_dir}/psi_g{g}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    
        # Scalar flux scatter with H/U shading
        x = np.linspace(0,sn.num_nodes,sn.num_sections)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        #shade_material(ax)
        ax.plot(x, phi[:,g])
        ax.set_title(fr"Group {g} — $\phi (x)$")
        ax.set_xlabel("Section")
        ax.set_ylabel(r"$\phi$")
        # optional legend
        legend_patches = [Patch(facecolor='blue', alpha=0.12, label='A=1 (H)'),
                          Patch(facecolor='red',  alpha=0.12, label='A=238 (U)')]
        ax.legend(handles=legend_patches, loc='upper right', frameon=False)
        fig.savefig(f"{save_dir}/phi_g{g}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
